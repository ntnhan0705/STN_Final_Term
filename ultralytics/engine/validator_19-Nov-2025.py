# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Base validator - compatible with YOLOv8 + custom STN/SupCon pipeline.

Highlights:
- Safe val-loss path during training (call train graph without updating BN).
- Robust feature extraction from arbitrary train forward outputs.
- Prefers detect head maps ('p' in raw) over generic 4D scan; otherwise sorts by H*W.
- Channel filter equals 'no' (==) to avoid mixing backbone maps with detect head.
- Flexible criterion call: passes SupCon/STN context if criterion.accepts_ctx is True,
  otherwise falls back to old API and stashes raw into criterion._external_ctx when available.
- Clear logging when val-loss is computed or skipped (and why).
"""

import time
from pathlib import Path
import numbers

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis, DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode


class BaseValidator:
    """
    A base class for creating validators with robust val-loss support during training.
    """

    def __init__(self, args=None, dataloader=None, save_dir=None, pbar=None):
        super().__init__()
        self.args = args or get_cfg(DEFAULT_CFG)
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.pbar = pbar
        self.stride = 32
        self.data = None
        self.model = None
        self.metrics = None
        self.callbacks = callbacks.get_default_callbacks()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.loss = torch.zeros(3)
        self.skip_loss = False
        self.jdict = []
        self.plots = {}
        self.on_plot = self.on_plot
        self.pbar_desc = self.get_desc()

        # Accumulators for VAL loss (box, cls, dfl, sup)
        self._val_items_sum = None  # tensor(4)
        self._val_items_count = 0

        # Quiet/once-only warnings & flags
        self._warned_no_criterion = False
        self._warned_feat_shortage = False
        self._warned_align_fallback = False
        self._logged_val_loss_setup = False

    # ---------- Minimal compat for standalone validate ----------
    def get_data(self):
        d = getattr(self.args, "data", None)
        if isinstance(d, (str, Path)):
            try:
                return check_det_dataset(d)
            except Exception:
                try:
                    return check_cls_dataset(d)
                except Exception:
                    return d
        return d if d is not None else getattr(self, "data", None)

    # ---------------------- Helpers ----------------------
    @staticmethod
    def _flatten_to_float_list(x):
        if torch.is_tensor(x):
            return [] if x.numel() == 0 else [float(t) for t in x.detach().flatten().tolist()]
        if isinstance(x, dict):
            out = []
            for v in x.values():
                out.extend(BaseValidator._flatten_to_float_list(v))
            return out
        if isinstance(x, (list, tuple)):
            out = []
            for v in x:
                out.extend(BaseValidator._flatten_to_float_list(v))
            return out
        if isinstance(x, numbers.Number):
            return [float(x)]
        return []

    @staticmethod
    def _items_to_vec4(items, device):
        """
        Normalize 'items' (from criterion) into tensor[4] = [box, cls, dfl, sup].
        Accepts dict or any nested structure, with common aliases.
        """
        def _first_num(val):
            fl = BaseValidator._flatten_to_float_list(val)
            return fl[0] if fl else 0.0

        if isinstance(items, dict):
            aliases = {
                "box_loss": ["box_loss", "box", "loss_box"],
                "cls_loss": ["cls_loss", "cls", "loss_cls"],
                "dfl_loss": ["dfl_loss", "dfl", "loss_dfl"],
                "supcon_loss": ["supcon_loss", "sup", "contrastive", "loss_supcon", "sup_loss"],
            }
            out = []
            for _, names in aliases.items():
                v = 0.0
                for name in names:
                    if name in items:
                        vv = items[name]
                        if torch.is_tensor(vv):
                            v = 0.0 if vv.numel() == 0 else float(vv.detach().view(-1)[0].item())
                        else:
                            v = _first_num(vv)
                        break
                out.append(v)
            return torch.tensor(out, device=device, dtype=torch.float32)

        flat = BaseValidator._flatten_to_float_list(items)
        vec = torch.zeros(4, device=device, dtype=torch.float32)
        n = min(4, len(flat))
        if n:
            vec[:n] = torch.tensor(flat[:n], device=device, dtype=torch.float32)
        return vec

    @staticmethod
    def _gather_all_4d_tensors(obj, out_list):
        """Collect all 4D tensors (B,C,H,W) with H>0,W>0 in insertion order."""
        if torch.is_tensor(obj):
            if obj.dim() == 4 and obj.shape[2] > 0 and obj.shape[3] > 0:
                out_list.append(obj)
            return
        if isinstance(obj, (list, tuple)):
            for v in obj:
                BaseValidator._gather_all_4d_tensors(v, out_list)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                BaseValidator._gather_all_4d_tensors(v, out_list)
            return

    @staticmethod
    def _extract_feats_4d(raw):
        """
        Return a list of 4D feature maps (B,C,H,W) for detection loss.
        Prefer 'known' keys if present; otherwise, scan the entire structure.
        """
        if isinstance(raw, dict):
            for k in ["feats", "features", "p", "out", "outputs", "y", "maps", "raw"]:
                if k in raw:
                    lst = []
                    BaseValidator._gather_all_4d_tensors(raw[k], lst)
                    if lst:
                        return lst
        lst = []
        BaseValidator._gather_all_4d_tensors(raw, lst)
        return lst

    @staticmethod
    def _align_pyramid(feats4d, expected_n):
        """
        Sort by spatial area (H*W) desc and pick exactly expected_n maps.
        Returns (ok, feats_aligned). ok=False if insufficient maps.
        """
        if not feats4d:
            return False, []
        feats4d_sorted = sorted(feats4d, key=lambda t: int(t.shape[2]) * int(t.shape[3]), reverse=True)
        if len(feats4d_sorted) < expected_n:
            return False, feats4d_sorted
        return True, feats4d_sorted[:expected_n]

    @staticmethod
    def _call_criterion_safely(criterion_fn, *args, **kwargs):
        """Wrap criterion call (caller handles try/except)."""
        return criterion_fn(*args, **kwargs)

    @staticmethod
    def _collect_supcon_ctx(raw):
        """Pick common SupCon/STN context keys from raw."""
        ctx = {}
        if isinstance(raw, dict):
            for k in ("stn_feat", "supcon_feat", "theta", "stn_theta", "state", "aux"):
                if k in raw:
                    ctx[k] = raw[k]
        return ctx

    # ---------------------- Main entry ----------------------
    @torch.no_grad()
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Validate during training (preferred) or standalone."""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)

        if self.training:
            # ===== VALIDATE DURING TRAIN =====
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = False
            self.save_dir = trainer.save_dir

            self._val_items_sum = torch.zeros(4, device=self.device, dtype=torch.float32)
            self._val_items_count = 0

            model_infer = trainer.ema.ema or trainer.model
            model_infer = model_infer.float().eval()
            self.model = trainer.model  # training model for val-loss
            model = model_infer

            if not self.dataloader:
                self.dataloader = self.get_dataloader(self.data.get(self.args.split), self.args.batch)
            if self.pbar is None:
                self.pbar = TQDM(self.dataloader, desc=self.pbar_desc, total=len(self.dataloader))

            if not self._logged_val_loss_setup:
                LOGGER.info("[Validator] Training-mode validation: will compute val-loss from train graph without BN updates.")
                self._logged_val_loss_setup = True

        else:
            # ===== STANDALONE VALIDATE =====
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_val_start")
            assert model is not None, "Either trainer or model is required for validation"

            self.device = select_device(self.args.device, self.args.batch)
            self.args.half &= self.device.type != "cpu"

            if isinstance(model, (str, Path)):
                model = AutoBackend(
                    model,
                    device=self.device,
                    dnn=getattr(self.args, "dnn", False),
                    fp16=self.args.half,
                )
            elif isinstance(model, torch.nn.Module):
                if self.args.half and self.device.type != "cpu" and hasattr(model, "half"):
                    model = model.half()
                else:
                    model = model.float()
            else:
                raise TypeError(f"validator: unsupported model type {type(model)}")

            self.model = model.eval()
            self.data = self.get_data()
            if self.device.type == "cpu":
                self.args.workers = 0
            if not self.dataloader:
                self.dataloader = self.get_dataloader(self.data.get(self.args.split), self.args.batch)
            if self.pbar is None:
                self.pbar = TQDM(self.dataloader, desc=self.pbar_desc, total=len(self.dataloader))

            # Optional warmup
            imsz = self.args.imgsz
            if isinstance(imsz, int):
                imsz = (imsz, imsz)
            bs = 1 if getattr(self.args, "pt", True) else getattr(self.args, "batch", 1)
            warm = getattr(model, "warmup", None)
            if callable(warm):
                try:
                    warm(imgsz=(bs, 3, *imsz))
                except Exception:
                    pass
            else:
                try:
                    dev = next(model.parameters()).device if isinstance(model, torch.nn.Module) else self.device
                    dummy = torch.zeros(bs, 3, imsz[0], imsz[1], device=dev)
                    model(dummy)
                except Exception:
                    pass

        # === Validation loop ===
        from ultralytics.utils.ops import Profile
        dt = (Profile(), Profile(), Profile(), Profile())
        bar = self.pbar

        # names
        if isinstance(self.data, dict) and "names" in self.data and model is not None:
            model.names = self.data["names"]
        elif hasattr(self.model, "names") and model is not None:
            model.names = self.model.names

        self.init_metrics(de_parallel(model if isinstance(model, torch.nn.Module) else self.model))
        self.jdict = []

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i

            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference path (eval) for metrics
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # ---------- val-loss via train-path (quiet & robust) ----------
            with dt[2]:
                if self.training:
                    bn_states = []
                    try:
                        model_train = self.model
                        was_training = getattr(model_train, "training", False)
                        model_train.train()  # to produce raw detect head outputs (train path)

                        # Freeze BN updates during val-loss
                        for m in model_train.modules():
                            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                                bn_states.append((m, m.training))
                                m.eval()

                        # Forward on training graph (no grads due to no_grad)
                        raw = model_train(batch["img"], augment=False, profile=False)

                        # Extract 4D detection maps
                        feats4d_all = self._extract_feats_4d(raw)

                        # ===== Find detection criterion (v8DetectionLoss or wrapper) =====
                        criterion_obj, crit_src = None, "N/A"

                        def _is_valid_crit(cand):
                            # chỉ chấp nhận object callable, không phải Tensor/number
                            return (
                                cand is not None
                                and callable(cand)
                                and not isinstance(cand, (torch.Tensor, numbers.Number))
                            )

                        # 1) Ưu tiên validator.criterion (có thể được EnableValLoss gán)
                        cand = getattr(self, "criterion", None)
                        if _is_valid_crit(cand):
                            criterion_obj, crit_src = cand, "validator.criterion"

                        # 2) trainer.criterion (YOLO gốc + ReinforceSupConToLoss patch ở đây)
                        if criterion_obj is None and trainer is not None:
                            cand = getattr(trainer, "criterion", None)
                            if _is_valid_crit(cand):
                                criterion_obj, crit_src = cand, "trainer.criterion"

                        # 3) model.criterion
                        if criterion_obj is None and self.model is not None:
                            cand = getattr(self.model, "criterion", None)
                            if _is_valid_crit(cand):
                                criterion_obj, crit_src = cand, "model.criterion"

                        # 4) model.loss (một số pipeline custom gán ở đây)
                        if criterion_obj is None and self.model is not None:
                            cand = getattr(self.model, "loss", None)
                            if _is_valid_crit(cand):
                                criterion_obj, crit_src = cand, "model.loss"

                        # Không tìm được criterion hợp lệ
                        if criterion_obj is None:
                            if not self._warned_no_criterion:
                                LOGGER.warning(
                                    "[Validator] Val-loss disabled: no detection criterion found on "
                                    "validator/trainer/model."
                                )
                                self._warned_no_criterion = True
                            raise RuntimeError("no detection criterion")

                        # Debug 1 lần xem nó lấy crit từ đâu
                        if batch_i == 0 and not hasattr(self, "_logged_criterion_source"):
                            try:
                                cname = criterion_obj.__class__.__name__
                            except Exception:
                                cname = str(type(criterion_obj))
                            LOGGER.info(f"[Validator] Using detection criterion from {crit_src}: {cname}")
                            self._logged_criterion_source = True

                        # ===== Expected nl & channels from detection criterion =====
                        expected_n = 0
                        if hasattr(criterion_obj, "stride"):
                            try:
                                expected_n = len(criterion_obj.stride)
                            except Exception:
                                expected_n = 0
                        if not expected_n:
                            expected_n = 3  # fallback an toàn

                        # Channel filter: keep maps whose channels exactly match criterion.no
                        min_no = 0
                        try:
                            min_no = int(getattr(criterion_obj, "no", 0) or 0)
                        except Exception:
                            min_no = 0
                        feats4d = feats4d_all
                        if min_no > 0:
                            kept = [t for t in feats4d_all if int(t.shape[1]) == min_no]
                            if kept:
                                feats4d = kept
                            else:
                                if not self._warned_align_fallback:
                                    LOGGER.warning(
                                        f"[Validator] No maps with channels == no ({min_no}). "
                                        f"Falling back to unfiltered 4D maps (may pick backbone)."
                                    )
                                    self._warned_align_fallback = True

                        # Prefer canonical detect head order if available
                        if isinstance(raw, dict) and "p" in raw and isinstance(raw["p"], (list, tuple)):
                            feats_list = list(raw["p"])[:expected_n]
                            if batch_i == 0:
                                shapes = [tuple(t.shape) for t in feats_list]
                                LOGGER.info(
                                    f"[Validator] Using raw['p'] for detect maps (nl={len(feats_list)}): {shapes}"
                                )
                        else:
                            ok, feats_list = self._align_pyramid(feats4d, expected_n)
                            if not ok:
                                if not self._warned_feat_shortage:
                                    LOGGER.warning(
                                        f"[Validator] Val-loss skipped: need {expected_n} maps, "
                                        f"got {len(feats4d)} (total_4d={len(feats4d_all)})."
                                    )
                                    self._warned_feat_shortage = True
                                raise RuntimeError("insufficient feature maps for detection loss")
                            if batch_i == 0 and not self._warned_align_fallback:
                                shapes = [tuple(t.shape) for t in feats_list]
                                LOGGER.info(f"[Validator] Using H*W sort fallback for detect maps: {shapes}")
                                self._warned_align_fallback = True

                        # Use the detection criterion (đã đảm bảo callable ở _is_valid_crit)
                        criterion_fn = criterion_obj

                        # Build SupCon/STN context
                        ctx = {"feats": feats_list}
                        sup_ctx = self._collect_supcon_ctx(raw)
                        ctx.update(sup_ctx)

                        # Criterion call: prefers context API if supported
                        try:
                            if getattr(criterion_obj, "accepts_ctx", False):
                                if batch_i == 0:
                                    LOGGER.info("[Validator] Calling criterion(ctx, batch) with SupCon/STN context.")
                                total, items = self._call_criterion_safely(criterion_fn, ctx, batch)
                            else:
                                # Back-compat: stash raw for old-style criterions that sniff external ctx
                                try:
                                    setattr(criterion_obj, "_external_ctx", raw)
                                except Exception:
                                    pass
                                if batch_i == 0:
                                    LOGGER.info("[Validator] Calling criterion(feats, batch) (legacy path).")
                                total, items = self._call_criterion_safely(criterion_fn, feats_list, batch)
                        except Exception as e:
                            LOGGER.warning(
                                f"[Validator] Val-loss criterion error at batch {batch_i}: {e}. Skipping this batch."
                            )
                            raise

                        # items -> vec4 and accumulate
                        items_t = self._items_to_vec4(items, self.device)
                        if self._val_items_sum is None:
                            self._val_items_sum = torch.zeros(4, device=self.device, dtype=torch.float32)
                            self._val_items_count = 0
                        self._val_items_sum += items_t
                        self._val_items_count += 1

                        # Update progress bar (avg val-loss components)
                        if self.pbar is not None and self._val_items_count > 0:
                            avg = (self._val_items_sum / max(1, self._val_items_count)).detach().float().cpu().tolist()
                            vbox, vcls, vdfl, vsup = (avg + [0.0, 0.0, 0.0, 0.0])[:4]
                            try:
                                self.pbar.set_postfix_str(
                                    f"vbox={vbox:.3f} vcls={vcls:.3f} vdfl={vdfl:.3f} vsup={vsup:.3f} "
                                    f"vtot={(vbox + vcls + vdfl + vsup):.3f}",
                                    refresh=False,
                                )
                            except Exception:
                                pass

                    except Exception:
                        # Quietly skip this batch val-loss (already logged why)
                        pass
                    finally:
                        # Restore BN + train/eval state
                        try:
                            for m, prev in bn_states:
                                m.train(prev)
                        except Exception:
                            pass
                        try:
                            model_train.train(was_training)
                        except Exception:
                            pass


            # Postprocess & metrics
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        # ---- Finalize ----
        stats = self.get_stats()
        self.check_stats(stats)
        n = max(1, len(self.dataloader))
        self.speed = {
            "preprocess": dt[0].dt * 1e3 / n,
            "inference": dt[1].dt * 1e3 / n,
            "loss": dt[2].dt * 1e3 / n,
            "postprocess": dt[3].dt * 1e3 / n,
        }
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")

        # ---- Write aggregated val-loss to stats (during training only) ----
        if self.training:
            if self._val_items_count > 0:
                avg_items = (self._val_items_sum / max(1, self._val_items_count)).detach().float().cpu()

                def _get(i):
                    return float(avg_items[i].item()) if i < avg_items.numel() else 0.0

                box_m = _get(0)
                cls_m = _get(1)
                dfl_m = _get(2)
                sup_m = _get(3)

                stats["val/box_loss"] = box_m
                stats["val/cls_loss"] = cls_m
                stats["val/dfl_loss"] = dfl_m
                stats["val/sup_loss"] = sup_m
                stats["val/loss"] = box_m + cls_m + dfl_m + sup_m

                if hasattr(self, "metrics") and hasattr(self.metrics, "results_dict"):
                    self.metrics.results_dict["val/box_loss"] = box_m
                    self.metrics.results_dict["val/cls_loss"] = cls_m
                    self.metrics.results_dict["val/dfl_loss"] = dfl_m
                    self.metrics.results_dict["val/sup_loss"] = sup_m

                LOGGER.info(
                    f"[VAL/LOSS] box={box_m:.4f} | cls={cls_m:.4f} | dfl={dfl_m:.4f} | sup={sup_m:.4f} | "
                    f"total={box_m+cls_m+dfl_m+sup_m:.4f}"
                )
            else:
                zeros = {"val/box_loss": 0.0, "val/cls_loss": 0.0, "val/dfl_loss": 0.0, "val/sup_loss": 0.0}
                stats.update({**zeros, "val/loss": 0.0})
                if hasattr(self, "metrics") and hasattr(self.metrics, "results_dict"):
                    self.metrics.results_dict.update(zeros)
                LOGGER.warning("[Validator] No val-loss batches were aggregated (all skipped).")

            stats.setdefault(
                "fitness",
                float(stats.get("metrics/mAP50-95", stats.get("mAP50-95", stats.get("map50-95", 0.0)))),
            )

        return stats

    # ---------------------- Stubs/boilerplate ----------------------
    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        return batch

    def postprocess(self, preds):
        return preds

    def init_metrics(self, model):
        pass

    def update_metrics(self, preds, batch):
        pass

    def finalize_metrics(self, *args, **kwargs):
        pass

    def get_stats(self):
        return {}

    def check_stats(self, stats):
        pass

    def print_results(self):
        pass

    def get_desc(self):
        return "Validating"

    @property
    def metric_keys(self):
        return []

    def on_plot(self, name, data=None):
        if not hasattr(self, "plots"):
            self.plots = {}
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    def plot_val_samples(self, batch, ni):
        pass

    def plot_predictions(self, batch, preds, ni):
        pass

    def pred_to_json(self, preds, batch):
        pass

    def eval_json(self, stats):
        pass

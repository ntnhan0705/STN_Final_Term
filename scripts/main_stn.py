# main_stn.py (slim v3) — compatible with stn_utils.py (and the slim version you have)
from __future__ import annotations
import argparse, random, sys, os
from pathlib import Path
from types import SimpleNamespace
import logging

import numpy as np, torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# pairing + supcon + stn utilities
from ultralytics.utils.stn_pairing import UsePairedLoader
from ultralytics.utils.stn_utils import (
    # core + debug
    STNControl, DebugImages, DebugBgPairROIs,
    # supcon plumbing
    InjectSupConArgsMinimal, ReinforceSupConToLoss, SupConScheduler,
    LinkTrainerToLoss, SyncEpochToLoss, TapSTNFeat, SupConPercentLogger,
    AttachSupConProjToOptim, LossNaNGuard, BatchSanityFilter,
    supcon_register_projector,
    PublishThetaToStateV2,
    DumpModelWiringOnVal,
    WireThetaToSDTNHead,
    EnsureThetaOnValBatch,
    debug_val_sample,
)

# ---------------------- (Optional) DEMO PATHS ----------------------
DEMO_YAML   = r"C:/OneDrive/Study/AI/STN_Final_Term/dataset.yaml"
DEMO_MODEL  = r"C:/OneDrive/Study/AI/STN_Final_Term/models/yolo11m_stn.pt"
DEMO_BGPAIR = r"C:/OneDrive/Study/AI/STN_Final_Term/scripts/bgpair_map.json"

# ---------------------- Silence ONLY [SupConDbg] ----------------------
def _debug_pred_boxes_on_val(trainer, *a, **k):
    if not hasattr(trainer, "batch") or not hasattr(trainer, "pred"):
        return
    batch = trainer.batch
    pred = trainer.pred
    if not isinstance(pred, (list, tuple)) or len(pred) < 2:
        return
    pred_scores, pred_bboxes = pred[0], pred[1]
    images = batch.get("img", None)
    if images is None:
        return
    debug_val_sample(images, pred_scores, pred_bboxes, save_dir=f"{trainer.save_dir}/val_pred")

class _DenySupConDbg(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return "[SupConDbg]" not in record.getMessage()
        except Exception:
            return True

_try_filter = _DenySupConDbg()
try:
    for h in list(LOGGER.handlers):
        h.addFilter(_try_filter)
    logging.getLogger("ultralytics").addFilter(_try_filter)
    logging.getLogger().addFilter(_try_filter)
except Exception:
    pass

# ---------------------- Small helpers ----------------------
class LRGuard:
    """Ensure every optimizer param_group has 'initial_lr' (avoids scheduler KeyError)."""
    def _fix(self, trainer):
        opt = getattr(trainer, "optimizer", None)
        if opt is None:
            return
        fixed = 0
        for g in opt.param_groups:
            if "initial_lr" not in g:
                g["initial_lr"] = g.get("lr", 0.0)
                fixed += 1
        if fixed:
            LOGGER.info(f"[LRGuard] injected 'initial_lr' into {fixed} optimizer param groups")
    def on_train_start(self, trainer, *_, **__): self._fix(trainer)
    def on_train_epoch_start(self, trainer, *_, **__): self._fix(trainer)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------- SafeTSNE (lightweight t-SNE, local impl) ----------------------
class SafeTSNE:
    """Collect GAP vectors (CPU) via a forward hook and export t-SNE every N epochs."""
    def __init__(self, every=10, max_samples=4000, per_class_cap=300, seed=0):
        self.every = int(every)
        self.max_samples = int(max_samples)
        self.per_class_cap = int(per_class_cap)
        self.seed = int(seed)
        self.hook = None
        self._feat = None
        self._xs, self._ys = [], []

    def _cfg(self, trainer, key, default=None):
        cfg = getattr(trainer, "_supcon_cfg", None)
        if cfg is None:
            return default
        return (cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default))

    @torch.no_grad()
    def _hook_fn(self, module, _in, out):
        if torch.is_tensor(out):
            x = F.adaptive_avg_pool2d(out, (1,1)).flatten(1) if out.dim()==4 else out
            if x.dim()==2:
                self._feat = x.detach().to("cpu", non_blocking=True).contiguous()

    def _resolve_module(self, model, idx, name):
        modlist = getattr(model, "model", None)
        if modlist is None:
            return None, None
        if isinstance(idx, int) and 0 <= idx < len(modlist):
            return modlist[idx], f"model.model[{idx}]"
        if name:
            low = str(name).lower()
            for i, m in enumerate(modlist):
                if low in m.__class__.__name__.lower():
                    return m, f"model.model[{i}]<{m.__class__.__name__}>"
        for i, m in enumerate(modlist):
            if hasattr(m, "forward"):
                return m, f"model.model[{i}]<{m.__class__.__name__}>"
        return None, None

    def on_train_start(self, trainer, *a, **k):
        idx = self._cfg(trainer, "supcon_out", None); name = self._cfg(trainer, "supcon_name", None)
        m, where = self._resolve_module(trainer.model, idx, name)
        if m is None:
            LOGGER.info("[SafeTSNE] could not attach hook -> skip")
            return
        if self.hook is not None:
            try: self.hook.remove()
            except Exception: pass
        self.hook = m.register_forward_hook(self._hook_fn)
        self._feat = None; self._xs.clear(); self._ys.clear()
        LOGGER.info(f"[SafeTSNE] hook at {where}; every={self.every}, max={self.max_samples}")

    @torch.no_grad()
    def on_train_batch_end(self, trainer, *a, **k):
        if self._feat is None:
            return
        x = self._feat; self._feat = None
        y = None
        b = getattr(trainer, "batch", None)
        if isinstance(b, dict) and torch.is_tensor(b.get("cls", None)):
            y = b["cls"].view(-1).to("cpu", non_blocking=True).long()
            if y.numel() != x.size(0):
                y = None
        if y is None:
            y = torch.full((x.size(0),), -1, dtype=torch.long)
        if self.per_class_cap > 0:
            from collections import Counter
            cnt = Counter(self._ys)
            for xi, yi in zip(x, y):
                yi = int(yi.item())
                if cnt[yi] < self.per_class_cap:
                    self._xs.append(xi.numpy()); self._ys.append(yi); cnt[yi] += 1
                if len(self._xs) >= self.max_samples:
                    break
        else:
            for xi, yi in zip(x, y):
                self._xs.append(xi.numpy()); self._ys.append(int(yi.item()))
                if len(self._xs) >= self.max_samples:
                    break
        if len(self._xs) > self.max_samples:
            self._xs = self._xs[:self.max_samples]; self._ys = self._ys[:self.max_samples]

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, *a, **k):
        ep = int(getattr(trainer, "epoch", 0))
        if self.every <= 0 or (ep % self.every) != 0:
            return
        if len(self._xs) < 50:
            LOGGER.info(f"[SafeTSNE] skip epoch {ep}: few samples ({len(self._xs)})")
            return
        xs = np.asarray(self._xs, dtype=np.float32)
        ys = np.asarray(self._ys, dtype=np.int64) if len(self._ys)==len(self._xs) else None
        try:
            from sklearn.manifold import TSNE; import time
            t0 = time.time()
            tsne = TSNE(
                n_components=2,
                perplexity=min(30, max(5, xs.shape[0]//100)),
                init="pca",
                learning_rate="auto",
                random_state=self.seed,
                n_iter=1000,
                verbose=0
            )
            emb = tsne.fit_transform(xs); _ = time.time()-t0
        except Exception as e:
            LOGGER.warning(f"[SafeTSNE] TSNE failed: {e}")
            return
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6,5), dpi=160)
            if ys is None or (ys < 0).all():
                plt.scatter(emb[:,0], emb[:,1], s=3)
            else:
                num_cls = int(np.clip(len(np.unique(ys[ys>=0])), 1, 20))
                plt.scatter(emb[:,0], emb[:,1], c=np.clip(ys, 0, num_cls-1), s=3, cmap="tab20")
            plt.title(f"t-SNE epoch {ep} (n={xs.shape[0]})"); plt.tight_layout()
            out_dir = os.path.join(str(getattr(trainer, "save_dir", "runs")), "embeddings"); os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"tsne_epoch_{ep:03d}.png"); fig.savefig(out_path); plt.close(fig)
            LOGGER.info(f"[SafeTSNE] saved {out_path} (n={xs.shape[0]})")
        except Exception as e:
            LOGGER.warning(f"[SafeTSNE] plot failed: {e}")

    def on_train_end(self, trainer, *a, **k):
        if self.hook is not None:
            try: self.hook.remove()
            except Exception: pass
        self.hook = None; self._feat = None; self._xs.clear(); self._ys.clear()

# ---------------------- Training routine ----------------------
def train_one(args: argparse.Namespace, run_id: int):
    set_seed(run_id)
    LOGGER.info(f"===== RUN {run_id}/{args.runs} =====")

    yolo = YOLO(args.model)
    save_dir = Path(args.output) / f"{Path(args.model).stem}_run{run_id}"

    # ---- SupCon config (early -> model.args) ----
    supcon_cfg = dict(
        supcon_on=args.supcon_on,
        supcon_feat=args.supcon_feat,
        supcon_warp_gt=args.supcon_warp_gt,
        supcon_out=args.supcon_out,
        supcon_min_box=args.supcon_min_box,
        supcon_max_per_class=args.supcon_max_per_class,
        supcon_gain=args.supcon_gain,
        supcon_loss_weight=(args.supcon_loss_weight if args.supcon_loss_weight is not None else args.supcon_gain),
        supcon_temp=args.supcon_temp,
        supcon_warmup=args.supcon_warmup,
        supcon_log=args.supcon_log,
        supcon_use_mem=args.supcon_use_mem,
        supcon_queue=args.supcon_queue,
        supcon_schedule=args.supcon_schedule,
        supcon_proj_dim=args.supcon_proj_dim,
        supcon_proj_hidden=args.supcon_proj_hidden,
        supcon_proj_bn=args.supcon_proj_bn,
    )
    if not isinstance(getattr(yolo.model, "args", None), SimpleNamespace):
        yolo.model.args = SimpleNamespace()
    for k, v in supcon_cfg.items():
        setattr(yolo.model.args, k, v)
    LOGGER.info(f"[SupCon/EARLY] supcon_feat={yolo.model.args.supcon_feat}, on={yolo.model.args.supcon_on}")

    # ===================== CALLBACKS =====================
    # 0) inject supcon args vào trainer/loss
    inj = InjectSupConArgsMinimal(**supcon_cfg)
    yolo.add_callback("on_pretrain_routine_start", inj.on_pretrain_routine_start)
    yolo.add_callback("on_pretrain_routine_end",   inj.on_pretrain_routine_end)
    yolo.add_callback("on_train_start",            inj.on_train_start)

    # 1) lịch bật/tắt + reinforce hyper theo epoch
    yolo.add_callback("on_train_epoch_start", SupConScheduler(args.supcon_schedule, default_on=args.supcon_on).on_train_epoch_start)
    yolo.add_callback("on_train_epoch_start", ReinforceSupConToLoss(supcon_cfg.keys()).on_train_epoch_start)

    # 2) liên kết trainer<->loss + sync epoch/batch
    yolo.add_callback("on_train_start", LinkTrainerToLoss().on_train_start)
    syncer = SyncEpochToLoss()
    yolo.add_callback("on_train_epoch_start", syncer.on_train_epoch_start)
    yolo.add_callback("on_train_batch_start", syncer.on_train_batch_start)

    # 3) projector + safety guards  (be more stubborn when attaching)
    attacher  = AttachSupConProjToOptim()
    nan_guard = LossNaNGuard(stop_on_nan=True, save_bad_batch=True)

    # cố gắng gắn ở nhiều điểm hơn (NEW lines)
    yolo.add_callback("on_pretrain_routine_end", attacher.on_train_start)  # sau khi optimizer được tạo
    yolo.add_callback("on_train_start",          attacher.on_train_start)
    yolo.add_callback("on_fit_epoch_start",      attacher.on_train_start)  # mỗi epoch thử lại 1 lần
    yolo.add_callback("on_train_batch_start",    attacher.on_train_batch_start)
    yolo.add_callback("on_train_batch_end",      attacher.on_train_batch_end)

    yolo.add_callback("on_train_batch_end",      nan_guard.on_train_batch_end)

    # 3b) lọc batch & LR guard
    yolo.add_callback("on_train_batch_start", BatchSanityFilter(eps=1e-6).on_train_batch_start)
    _lr_guard = LRGuard()
    yolo.add_callback("on_train_start",       _lr_guard.on_train_start)
    yolo.add_callback("on_train_epoch_start", _lr_guard.on_train_epoch_start)
    yolo.add_callback("on_val_batch_end", _debug_pred_boxes_on_val)

    # (optional) discover projector by alias then register globally so attacher can find it
    class _DiscoverSupConProj:
        def __init__(self): self.done=False
        def _try_register(self, trainer):
            if self.done:
                return
            # quét trong nhiều “nguồn” có thể chứa projector
            sources = [
                getattr(trainer, "criterion", None),
                getattr(trainer, "loss", None),
                getattr(getattr(trainer, "model", None), "criterion", None),
            ]
            aliases = ("_supcon_proj", "supcon_projector", "projector", "supcon_proj")
            for src in sources:
                if src is None:
                    continue
                for name in aliases:
                    proj = getattr(src, name, None)
                    if proj is not None:
                        try:
                            supcon_register_projector(proj)
                            self.done = True
                            LOGGER.info(f"[DiscoverSupConProj] projector registered from {type(src).__name__}.{name}")
                            return
                        except Exception:
                            pass

        # gọi lại ở nhiều mốc để chắc chắn “bắt” được projector khi nó sinh ra trễ
        def on_pretrain_routine_end(self, trainer, *a, **k): self._try_register(trainer)
        def on_train_start(self, trainer, *a, **k):          self._try_register(trainer)
        def on_fit_epoch_start(self, trainer, *a, **k):      self._try_register(trainer)
        def on_train_batch_end(self, trainer, *a, **k):      self._try_register(trainer)

    _discover = _DiscoverSupConProj()
    yolo.add_callback("on_pretrain_routine_end", _discover.on_pretrain_routine_end)
    yolo.add_callback("on_train_start",          _discover.on_train_start)
    yolo.add_callback("on_fit_epoch_start",      _discover.on_fit_epoch_start)
    yolo.add_callback("on_train_batch_end",      _discover.on_train_batch_end)

    # 4) STN trust-region + warmup (STNControl tự bypass IDENTITY khi validation)
    freeze_ep  = int(args.freeze_epochs)
    stn_warmup = max(12, int(args.supcon_warmup) if int(args.supcon_warmup) > 0 else 0, freeze_ep // 2)
    ctrl = STNControl(freeze_epochs=freeze_ep, stn_warmup=stn_warmup, tmax=0.20, smin=0.90, smax=1.10, log=True)
    def _relax_stn_bounds(trainer):
        e = int(getattr(trainer, "epoch", 0))
        if e == 20: ctrl.tmax, ctrl.smin, ctrl.smax = 0.25, 0.85, 1.15
        if e == 40: ctrl.tmax, ctrl.smin, ctrl.smax = 0.30, 0.80, 1.20
    yolo.add_callback("on_train_epoch_start", _relax_stn_bounds)
    yolo.add_callback("on_train_epoch_start", ctrl.on_train_epoch_start)

    # theta publish + wiring (VAL)
    theta_pub = PublishThetaToStateV2(verbose=True)
    yolo.add_callback("on_train_start", theta_pub.on_train_start)
    yolo.add_callback("on_train_end",   theta_pub.on_train_end)
    yolo.add_callback("on_val_start",   theta_pub.on_val_start)
    yolo.add_callback("on_val_end",     theta_pub.on_val_end)

    wire_dump = DumpModelWiringOnVal(verbose=True, max_lines=32)
    yolo.add_callback("on_val_start", wire_dump.on_val_start)

    wire_theta = WireThetaToSDTNHead(verbose=True)
    yolo.add_callback("on_val_start",       wire_theta.on_val_start)
    yolo.add_callback("on_val_batch_start", wire_theta.on_val_batch_start)

    # NEW: đảm bảo theta có mặt trong từng batch VAL (khớp import EnsureThetaOnValBatch)
    ensure_theta = EnsureThetaOnValBatch(verbose=True)
    yolo.add_callback("on_val_batch_start", ensure_theta.on_val_batch_start)
    # Fix pred format for YOLOv8 validator
    yolo.add_callback("on_val_batch_end", fix_val_prediction_format)

    # 5) Hook STN feature (GAP vector)
    tap = TapSTNFeat()
    yolo.add_callback("on_train_start", tap.on_train_start)
    yolo.add_callback("on_train_end",   tap.on_train_end)

    # 6) Debug ảnh + preview FG|BG pair (opt-in)
    if args.debug_images:
        yolo.add_callback("on_train_epoch_end", DebugImages(epochs=range(args.epochs + 1), max_images=5).on_train_epoch_end)
        yolo.add_callback("on_train_epoch_end", DebugBgPairROIs(epochs=range(args.epochs + 1), max_pairs=6).on_train_epoch_end)

    # 7) TSNE embeddings (safe & light). Tắt nếu tsne_every<=0
    if args.tsne_every > 0:
        _safe_tsne = SafeTSNE(
            every=args.tsne_every,
            max_samples=min(args.tsne_per_class * 4, 4000),
            per_class_cap=args.tsne_per_class,
            seed=0
        )
        yolo.add_callback("on_train_start",     _safe_tsne.on_train_start)
        yolo.add_callback("on_train_batch_end", _safe_tsne.on_train_batch_end)
        yolo.add_callback("on_train_epoch_end", _safe_tsne.on_train_epoch_end)
        yolo.add_callback("on_train_end",       _safe_tsne.on_train_end)

    # 8) Log % supcon
    pct_logger = SupConPercentLogger()
    yolo.add_callback("on_train_batch_end", pct_logger.on_train_batch_end)
    yolo.add_callback("on_train_epoch_end", pct_logger.on_train_epoch_end)

    # 9) Paired loader (BG pairing / negatives)
    if args.pairing:
        if args.batch % 2 != 0:
            LOGGER.warning(f"[Pairing] batch={args.batch} phải CHẴN; tự động giảm còn {args.batch - 1}.")
            args.batch -= 1

        pl = UsePairedLoader(data_yaml=args.yaml,
                             bgpair_map=args.bgpair_map,
                             batch_size=args.batch,
                             seed=run_id)

        # luôn có on_train_start
        yolo.add_callback("on_train_start", pl.on_train_start)

        # một số phiên bản có on_fit_epoch_start, số khác dùng on_train_epoch_start, hoặc không có
        if hasattr(pl, "on_fit_epoch_start"):
            yolo.add_callback("on_fit_epoch_start", pl.on_fit_epoch_start)
            LOGGER.info("[Pairing] hook=on_fit_epoch_start attached")
        elif hasattr(pl, "on_train_epoch_start"):
            yolo.add_callback("on_train_epoch_start", pl.on_train_epoch_start)
            LOGGER.info("[Pairing] hook=on_train_epoch_start attached")
        else:
            LOGGER.warning("[Pairing] No per-epoch hook found on UsePairedLoader "
                           "(sampler.set_epoch won’t be called; vẫn chạy OK).")

        LOGGER.info(f"[Pairing] Enabled with bgpair_map='{args.bgpair_map}', batch={args.batch}")


    # ---------- Train ----------
    device_arg = "0" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    # ép cấu hình “an toàn khỏi NaN/độ bất ổn” (giữ nguyên AMP theo CLI)
    train_overrides = dict(mixup=0.0, cutmix=0.0)

    yolo.train(
        data=args.yaml,
        epochs=int(args.epochs),
        batch=int(args.batch),
        imgsz=int(args.imgsz),
        device=device_arg,
        resume=bool(args.resume),
        project=str(save_dir.parent),
        name=save_dir.name,
        patience=int(args.patience),
        val=True,
        verbose=True,
        seed=run_id,
        save=bool(args.save),
        save_period=int(args.save_period),
        amp=bool(args.amp),
        **train_overrides,
    )
    LOGGER.info("✔ Training finished\n")

# ---------------------- Arg parsing ----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO-STN + SupCon training (slim)")

    # training common
    p.add_argument("--yaml", help="Path to dataset.yaml for Ultralytics YOLO.")
    p.add_argument("--model", help="Path to model weights .pt or model YAML.")
    p.add_argument("--output", default="runs_stn", help="Output root directory for runs (logs, ckpts, images).")
    p.add_argument("--epochs", type=int, default=100, help="Total training epochs.")
    p.add_argument("--batch", type=int, default=8, help="Global batch size.")
    p.add_argument("--imgsz", type=int, default=640, help="Training/validation image size (square).")
    p.add_argument("--runs", type=int, default=1, help="Number of repeated runs (seeds).")
    p.add_argument("--resume", action="store_true", help="Resume from the last checkpoint in the run folder.")
    p.add_argument("--device", default="auto", help="Device selector: 'auto', 'cpu', '0', '0,1', …")
    p.add_argument("--patience", type=int, default=100, help="Early-stopping patience.")
    p.add_argument("--freeze_epochs", type=int, default=30, help="Freeze/disable STN for first N epochs.")
    p.add_argument("--debug_images", action="store_true", help="Dump STN debug images.")
    p.add_argument("--tsne_every", type=int, default=0, help="Export t-SNE every N epochs (0 to disable).")
    p.add_argument("--tsne_per_class", type=int, default=300, help="Max samples per class for t-SNE.")
    p.add_argument("--amp", type=int, default=1, help="Automatic Mixed Precision: 1=enable, 0=disable.")
    p.add_argument("--save", type=int, default=1, help="1=save weights, 0=do not save")
    p.add_argument("--save_period", type=int, default=-1, help="-1=only best/last; >0=save every N epochs")

    # SupCon (internal)
    p.add_argument("--supcon_on", type=int, default=1)
    p.add_argument("--supcon_feat", type=str, default="stn")
    p.add_argument("--supcon_warp_gt", type=int, default=0)
    p.add_argument("--supcon_out", type=int, default=7)
    p.add_argument("--supcon_min_box", type=int, default=1)
    p.add_argument("--supcon_max_per_class", type=int, default=0)
    p.add_argument("--supcon_gain", type=float, default=1.0)
    p.add_argument("--supcon_temp", type=float, default=0.5)
    p.add_argument("--supcon_warmup", type=int, default=10)
    p.add_argument("--supcon_log", type=int, default=1)
    p.add_argument("--supcon_use_mem", type=int, default=1)
    p.add_argument("--supcon_queue", type=int, default=4096)
    p.add_argument("--supcon_loss_weight", type=float, default=None)
    p.add_argument("--supcon_schedule", type=str, default="0-")
    p.add_argument("--supcon_proj_dim", type=int, default=128)
    p.add_argument("--supcon_proj_hidden", type=int, default=512)
    p.add_argument("--supcon_proj_bn", type=int, default=2, choices=[0,1,2])

    # Pairing
    p.add_argument("--pairing", action="store_true", help="Bật paired sampler/negative background loss")
    p.add_argument("--bgpair_map", type=str, default="bgpair_map.json", help="Đường dẫn map abnormal->normal")

    # Demo (no CLI)
    if len(sys.argv) == 1:
        LOGGER.info("[INFO] No CLI arguments supplied — using demo paths (edit DEMO_* at top if needed)")
        args = p.parse_args([])
        args.yaml = DEMO_YAML
        args.model = DEMO_MODEL
        args.debug_images = True
        args.tsne_every = 0  # default OFF to keep slim
        args.tsne_per_class = 300
        args.pairing = True
        args.bgpair_map = DEMO_BGPAIR
        args.save = 1
        args.save_period = -1
        args.amp = 1
        return args

    # Normal CLI
    args = p.parse_args()
    if not Path(args.yaml).exists():
        LOGGER.warning(f"[WARN] {args.yaml} not found.")
    if not Path(args.model).exists():
        LOGGER.warning(f"[WARN] {args.model} not found.")
    return args

def fix_val_prediction_format(trainer, *a, **k):
    pred = getattr(trainer, "pred", None)
    if not isinstance(pred, (list, tuple)) or len(pred) < 2:
        return
    scores, boxes = pred[0], pred[1]  # [B, N, C], [B, N, 4]
    preds = []
    probs = scores.sigmoid()
    for i in range(scores.shape[0]):
        conf, cls = probs[i].max(dim=1)  # [N]
        mask = conf > 0.001
        if mask.sum() == 0:
            preds.append(torch.zeros((0, 6), device=scores.device))
            continue
        filtered_boxes = boxes[i][mask]
        filtered_conf = conf[mask].unsqueeze(1)
        filtered_cls = cls[mask].float().unsqueeze(1)
        preds.append(torch.cat([filtered_boxes, filtered_conf, filtered_cls], dim=1))  # [N, 6]
    trainer.pred = preds

def main():
    args = parse_args()
    for i in range(1, args.runs + 1):
        train_one(args, i)


if __name__ == "__main__":
    main()

# main_stn.py (slim v6.2, safe)
from __future__ import annotations
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import yaml
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from ultralytics.utils.stn_pairing import UsePairedLoader
from ultralytics.utils.stn_utils import (
    STNControl, DebugImages, DebugBgPairROIs,
    InjectSupConArgsMinimal, ReinforceSupConToLoss, SupConScheduler,
    LinkTrainerToLoss, SyncEpochToLoss, TapSTNFeat, SupConPercentLogger,
    AttachSupConProjToOptim, LossNaNGuard, BatchSanityFilter,
    supcon_register_projector, PublishThetaToStateV2,
    apply_logging_filter, set_seed, LRGuard, SafeTSNE,
    debug_pred_boxes_on_val, apply_val_debug_overrides, register_val_probe, register_results_csv_guard,
    ForceValArgs, ValPredStats, SaveLastBestOnly, FixFinalEvalModel
)

# ---------------------- (Paths & DEMO PATHS) ----------------------
RUNS_ROOT    = r"C:\OneDrive\Study\AI\STN_Final_Term\runs"
PAIRING_ROOT = r"C:\OneDrive\Study\AI\STN_Final_Term\pairing"

DEMO_YAML   = r"C:\OneDrive\Study\AI\STN_Final_Term\dataset.yaml"
DEMO_MODEL  = r"C:\OneDrive\Study\AI\STN_Final_Term\models\yolo11m_stn.pt"
DEMO_BGPAIR = rf"{PAIRING_ROOT}\bgpair_map.json"

# ---------------------- TupleUnwrapGuard ----------------------
class TupleUnwrapGuard:
    _STN_NAMES = {"SpatialTransformer", "STN", "SpatialTransformer2D", "SpatialTransformerBlock"}

    def __init__(self, only_after_stn: bool = True, verbose: bool = True):
        self.only_after_stn = bool(only_after_stn)
        self.verbose = bool(verbose)
        self._hooks = []

    @staticmethod
    def _is_stn(m) -> bool:
        return m.__class__.__name__ in TupleUnwrapGuard._STN_NAMES

    def _pre_hook(self, module, inputs):
        if not inputs:
            return None
        x = inputs[0]
        if isinstance(x, tuple):
            if len(x) >= 1 and torch.is_tensor(x[0]):
                return (x[0],) + tuple(inputs[1:])
        return None

    def _attach_to_model(self, model):
        if not hasattr(model, "model"):
            return 0
        hooks_added, stn_seen = 0, False
        for m in model.model:
            if self._is_stn(m):
                stn_seen = True
                continue
            if self.only_after_stn and not stn_seen:
                continue
            h = m.register_forward_pre_hook(self._pre_hook)
            self._hooks.append(h)
            hooks_added += 1
        if self.verbose:
            LOGGER.info(f"[TupleUnwrapGuard] attached pre_hooks={hooks_added} (after_stn={self.only_after_stn})")
        return hooks_added

    def _clear(self):
        for h in self._hooks:
            try: h.remove()
            except Exception: pass
        self._hooks.clear()

    def on_train_start(self, trainer, *a, **k):
        model = getattr(trainer, "model", None)
        if model is not None:
            self._attach_to_model(model)

    def on_val_start(self, validator, *a, **k):
        trainer = getattr(validator, "trainer", None)
        model = getattr(trainer, "model", None) if trainer is not None else None
        if model is not None:
            self._attach_to_model(model)

    def on_train_end(self, *a, **k):
        self._clear()

    def on_val_end(self, *a, **k):
        self._clear()

# ---------------------- InitialLRGuard ----------------------
class InitialLRGuard:
    def __init__(self, key: str = "initial_lr"):
        self.key = key

    def _ensure(self, trainer):
        opt = getattr(trainer, "optimizer", None)
        if opt is None:
            return
        base = float(getattr(getattr(trainer, "args", SimpleNamespace()), "lr0", 1e-3))
        for g in opt.param_groups:
            if self.key not in g:
                g[self.key] = g.get("lr", base)

    def on_train_start(self, trainer, *a, **k):
        self._ensure(trainer)

    def on_train_epoch_start(self, trainer, *a, **k):
        self._ensure(trainer)

    def on_train_batch_end(self, trainer, *a, **k):
        self._ensure(trainer)

# ---------------------- Training routine ----------------------
def _pick_dbg_epochs(total: int) -> set:
    if total <= 1: return {0}
    S = {0, 1, 2, min(5, total-1), total-1}
    return {e for e in S if 0 <= e < total}

def train_one(args: argparse.Namespace, run_id: int):
    set_seed(run_id)
    LOGGER.info(f"===== RUN {run_id}/{args.runs} =====")

    yolo = YOLO(args.model)

    # ensure output root exists
    Path(args.output).mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.output) / f"{Path(args.model).stem}_run{run_id}"

    # SupCon config (early -> model.args)
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
    inj = InjectSupConArgsMinimal(**supcon_cfg)
    yolo.add_callback("on_pretrain_routine_start", inj.on_pretrain_routine_start)
    yolo.add_callback("on_pretrain_routine_end",   inj.on_pretrain_routine_end)
    yolo.add_callback("on_train_start",            inj.on_train_start)

    yolo.add_callback("on_train_start", LinkTrainerToLoss().on_train_start)
    syncer = SyncEpochToLoss()
    yolo.add_callback("on_train_epoch_start", syncer.on_train_epoch_start)
    yolo.add_callback("on_train_batch_start", syncer.on_train_batch_start)

    yolo.add_callback("on_train_epoch_start",
                      SupConScheduler(args.supcon_schedule, default_on=args.supcon_on).on_train_epoch_start)
    yolo.add_callback("on_train_epoch_start",
                      ReinforceSupConToLoss(supcon_cfg.keys()).on_train_epoch_start)

    attacher = AttachSupConProjToOptim(in_dim=1024, out_dim=128, hidden=512, bn=1,
                                       lr=float(getattr(args, "lr0", 1e-3)))
    yolo.add_callback("on_train_batch_end", attacher.on_train_batch_end)

    _initlr_guard = InitialLRGuard()
    yolo.add_callback("on_train_start",       _initlr_guard.on_train_start)
    yolo.add_callback("on_train_epoch_start", _initlr_guard.on_train_epoch_start)
    yolo.add_callback("on_train_batch_end",   _initlr_guard.on_train_batch_end)

    nan_guard = LossNaNGuard(stop_on_nan=True, save_bad_batch=True)
    yolo.add_callback("on_train_batch_start", BatchSanityFilter(eps=1e-6).on_train_batch_start)
    yolo.add_callback("on_train_batch_end",   nan_guard.on_train_batch_end)

    _lr_guard = LRGuard()
    yolo.add_callback("on_train_start",       _lr_guard.on_train_start)
    yolo.add_callback("on_train_epoch_start", _lr_guard.on_train_epoch_start)

    yolo.add_callback("on_val_batch_end", debug_pred_boxes_on_val)
    yolo.add_callback("on_val_batch_end", ValPredStats(every=1).on_val_batch_end)

    class _DiscoverSupConProj:
        def __init__(self): self.done=False
        def _try_register(self, trainer):
            if self.done: return
            sources = [getattr(trainer,"criterion",None),
                       getattr(trainer,"loss",None),
                       getattr(getattr(trainer,"model",None),"criterion",None)]
            aliases = ("_supcon_proj","supcon_projector","projector","supcon_proj")
            for src in sources:
                if src is None: continue
                for name in aliases:
                    proj = getattr(src, name, None)
                    if proj is not None:
                        try:
                            supcon_register_projector(proj); self.done=True
                            LOGGER.info(f"[DiscoverSupConProj] projector registered from {type(src).__name__}.{name}")
                            return
                        except Exception: pass
        def on_pretrain_routine_end(self, t,*a,**k): self._try_register(t)
        def on_train_start(self, t,*a,**k):          self._try_register(t)
        def on_fit_epoch_start(self, t,*a,**k):      self._try_register(t)
        def on_train_batch_end(self, t,*a,**k):      self._try_register(t)
    _discover = _DiscoverSupConProj()
    yolo.add_callback("on_pretrain_routine_end", _discover.on_pretrain_routine_end)
    yolo.add_callback("on_train_start",          _discover.on_train_start)
    yolo.add_callback("on_fit_epoch_start",      _discover.on_fit_epoch_start)
    yolo.add_callback("on_train_batch_end",      _discover.on_train_batch_end)

    freeze_ep  = int(args.freeze_epochs)
    ctrl = STNControl(freeze_epochs=freeze_ep, stn_warmup=0, tmax=0.20, smin=0.90, smax=1.10, log=True)

    def _log_stn_alpha(trainer, *a, **k):
        e = int(getattr(trainer, "epoch", -1))
        a_ = getattr(ctrl, "_alpha", None)
        try:
            LOGGER.info(f"[STN/Alpha] epoch={e} alpha={0.0 if a_ is None else float(a_):.3f} (0=off,1=full)")
        except Exception:
            pass

    yolo.add_callback("on_train_epoch_start", ctrl.on_train_epoch_start)
    yolo.add_callback("on_train_epoch_start", _log_stn_alpha)
    yolo.add_callback("on_val_start",         ctrl.on_val_start)
    yolo.add_callback("on_val_end",           ctrl.on_val_end)

    theta_pub = PublishThetaToStateV2(verbose=True)
    yolo.add_callback("on_train_start", theta_pub.on_train_start)
    yolo.add_callback("on_train_end",   theta_pub.on_train_end)
    yolo.add_callback("on_val_start",   theta_pub.on_val_start)
    yolo.add_callback("on_val_end",     theta_pub.on_val_end)

    tap = TapSTNFeat()
    yolo.add_callback("on_train_start", tap.on_train_start)
    yolo.add_callback("on_train_end",   tap.on_train_end)

    tguard = TupleUnwrapGuard(only_after_stn=True, verbose=True)
    yolo.add_callback("on_train_start", tguard.on_train_start)
    yolo.add_callback("on_val_start",   tguard.on_val_start)
    yolo.add_callback("on_train_end",   tguard.on_train_end)
    yolo.add_callback("on_val_end",     tguard.on_val_end)

    # ---- EPOCHS để vẽ/lưu (chỉ một vài mốc)
    dbg_epochs = _pick_dbg_epochs(int(args.epochs))

    if args.debug_images:
        yolo.add_callback("on_train_epoch_end",
                          DebugImages(epochs=dbg_epochs, max_images=5).on_train_epoch_end)
        yolo.add_callback("on_train_epoch_end",
                          DebugBgPairROIs(epochs=dbg_epochs, max_pairs=6).on_train_epoch_end)

    # ---- VAL DEBUG OVERRIDES: chỉ bật ở epoch trong dbg_epochs
    apply_val_debug_overrides(yolo, conf=0.001, iou=0.50, epochs=dbg_epochs)
    yolo.add_callback("on_val_start",
                      ForceValArgs(conf=0.001, iou=0.50, max_det=2000, agnostic=False).on_val_start)
    yolo.add_callback("on_val_start", FixFinalEvalModel().on_val_start)
    # ---- GT labels dir (nếu có) cho probe
    gt_labels_dir = None
    try:
        if args.yaml and Path(args.yaml).exists():
            with open(args.yaml, "r", encoding="utf-8") as f:
                ycfg = yaml.safe_load(f) or {}
            valp = ycfg.get("val", None) or ycfg.get("validation", None)
            if isinstance(valp, str):
                p = Path(valp)
                if p.is_dir():
                    parts = list(p.parts)
                    for i, s in enumerate(parts):
                        if s.lower() == "images":
                            parts[i] = "labels"; break
                    cand = Path(*parts)
                    if cand.exists():
                        gt_labels_dir = cand
                    else:
                        alt = p.parent / "labels"
                        if alt.exists():
                            gt_labels_dir = alt
    except Exception:
        pass
    register_val_probe(yolo, gt_labels_dir=str(gt_labels_dir) if gt_labels_dir else None)
    register_results_csv_guard(yolo)

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

    pct_logger = SupConPercentLogger()
    yolo.add_callback("on_train_batch_end", pct_logger.on_train_batch_end)
    yolo.add_callback("on_train_epoch_end", pct_logger.on_train_epoch_end)

    # Save last/best only
    yolo.add_callback("on_train_start", SaveLastBestOnly().on_train_start)
    yolo.add_callback("on_fit_epoch_end", SaveLastBestOnly().on_fit_epoch_end)

    # ---- BG PAIRING LOADER
    if args.pairing:
        if args.batch % 2 != 0:
            LOGGER.warning(f"[Pairing] batch={args.batch} phải CHẴN; tự động giảm còn {args.batch - 1}.")
            args.batch -= 1
        if not Path(args.bgpair_map).exists():
            LOGGER.warning(f"[Pairing] bgpair_map not found at '{args.bgpair_map}'.")
        pl = UsePairedLoader(data_yaml=args.yaml, bgpair_map=args.bgpair_map,
                             batch_size=args.batch, seed=run_id)
        yolo.add_callback("on_train_start", pl.on_train_start)
        if hasattr(pl, "on_fit_epoch_start"):
            yolo.add_callback("on_fit_epoch_start", pl.on_fit_epoch_start)
            LOGGER.info("[Pairing] hook=on_fit_epoch_start attached")
        elif hasattr(pl, "on_train_epoch_start"):
            yolo.add_callback("on_train_epoch_start", pl.on_train_epoch_start)
            LOGGER.info("[Pairing] hook=on_train_epoch_start attached")
        else:
            LOGGER.warning("[Pairing] No per-epoch hook found on UsePairedLoader (sampler.set_epoch won’t run).")
        LOGGER.info(f"[Pairing] Enabled with bgpair_map='{args.bgpair_map}', batch={args.batch}")

    # ---------- Train ----------
    device_arg = "0" if (args.device == "auto" and torch.cuda.is_available()) else args.device
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
    )
    LOGGER.info("✔ Training finished\n")

# ---------------------- Arg parsing ----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO-STN + SupCon training (slim)")

    # training common
    p.add_argument("--yaml", help="Path to dataset.yaml for Ultralytics YOLO.")
    p.add_argument("--model", help="Path to model weights .pt or model YAML.")
    p.add_argument("--output", default=RUNS_ROOT, help="Output root directory for runs (logs, ckpts, images).")
    p.add_argument("--epochs", type=int, default=10, help="Total training epochs.")
    p.add_argument("--batch", type=int, default=8, help="Global batch size.")
    p.add_argument("--imgsz", type=int, default=640, help="Training/validation image size (square).")
    p.add_argument("--runs", type=int, default=1, help="Number of repeated runs (seeds).")
    p.add_argument("--resume", action="store_true", help="Resume from the last checkpoint in the run folder.")
    p.add_argument("--device", default="auto", help="Device selector: 'auto', 'cpu', '0', '0,1', …")
    p.add_argument("--patience", type=int, default=10, help="Early-stopping patience.")

    # STN
    p.add_argument("--freeze_epochs", type=int, default=5, help="Freeze/disable STN for first N epochs.")

    p.add_argument("--debug_images", action="store_true", help="Dump STN debug images.")
    p.add_argument("--tsne_every", type=int, default=0, help="Export t-SNE every N epochs (0 to disable).")
    p.add_argument("--tsne_per_class", type=int, default=300, help="Max samples per class for t-SNE.")
    p.add_argument("--amp", type=int, default=1, help="Automatic Mixed Precision: 1=enable, 0=disable.")
    p.add_argument("--save", type=int, default=1, help="1=save weights, 0=do not save")
    p.add_argument("--save_period", type=int, default=-1, help="-1=only best/last; >0=save every N epochs")

    # SupCon
    p.add_argument("--supcon_on", type=int, default=1)
    p.add_argument("--supcon_feat", type=str, default="stn")
    p.add_argument("--supcon_warp_gt", type=int, default=0)
    p.add_argument("--supcon_out", type=int, default=7)
    p.add_argument("--supcon_min_box", type=int, default=1)
    p.add_argument("--supcon_max_per_class", type=int, default=0)
    p.add_argument("--supcon_gain", type=float, default=0.25)
    p.add_argument("--supcon_temp", type=float, default=0.5)
    p.add_argument("--supcon_warmup", type=int, default=5)
    p.add_argument("--supcon_log", type=int, default=1)
    p.add_argument("--supcon_use_mem", type=int, default=1)
    p.add_argument("--supcon_queue", type=int, default=4096)
    p.add_argument("--supcon_loss_weight", type=float, default=None)
    p.add_argument("--supcon_schedule", type=str, default="2-")
    p.add_argument("--supcon_proj_dim", type=int, default=128)
    p.add_argument("--supcon_proj_hidden", type=int, default=512)
    p.add_argument("--supcon_proj_bn", type=int, default=2, choices=[0, 1, 2])

    # Pairing
    p.add_argument("--pairing", action="store_true", help="Bật paired sampler/negative background loss")
    p.add_argument("--bgpair_map", type=str, default=rf"{PAIRING_ROOT}\bgpair_map.json",
                   help="Đường dẫn map abnormal->normal (JSON).")

    # Demo (no CLI)
    if len(sys.argv) == 1:
        LOGGER.info("[INFO] No CLI arguments supplied — using demo paths")
        args = p.parse_args([])
        args.yaml = DEMO_YAML
        args.model = DEMO_MODEL
        args.output = RUNS_ROOT
        args.debug_images = True
        args.tsne_every = 0
        args.tsne_per_class = 300
        args.pairing = True
        args.bgpair_map = DEMO_BGPAIR
        args.save = 1
        args.save_period = -1
        args.amp = 1
        args.freeze_epochs = 5
        return args

    # Normal CLI
    args = p.parse_args()
    if not Path(args.yaml).exists():
        LOGGER.warning(f"[WARN] {args.yaml} not found.")
    if not Path(args.model).exists():
        LOGGER.warning(f"[WARN] {args.model} not found.")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    return args

def main():
    apply_logging_filter()
    args = parse_args()
    for i in range(1, args.runs + 1):
        train_one(args, i)

if __name__ == "__main__":
    main()

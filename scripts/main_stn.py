# main_stn.py (slim v6, fixed & instrumented) — Stabilize mAP, correct STN tap, val-identity window, rich debug logs
from __future__ import annotations
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import yaml
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# pairing + supcon + stn utilities
from ultralytics.utils.stn_pairing import UsePairedLoader
from ultralytics.utils.stn_utils import (
    STNControl, DebugImages, DebugBgPairROIs,
    InjectSupConArgsMinimal, ReinforceSupConToLoss, SupConScheduler,
    LinkTrainerToLoss, SyncEpochToLoss, TapSTNFeat, SupConPercentLogger,
    AttachSupConProjToOptim, LossNaNGuard, BatchSanityFilter,
    supcon_register_projector,
    PublishThetaToStateV2,
)

from stn_train_utils import (
    apply_logging_filter, set_seed,
    LRGuard, SafeTSNE,
    debug_pred_boxes_on_val,
    apply_val_debug_overrides, register_val_probe, register_results_csv_guard,
    ForceReturnTheta, CaptureThetaFromSTN,
    setup_stn_quiet_logs, QuietSTNLogs, ForceValArgs,
    SeedIdentityThetaOnValStart, ValPredStats,
    ThetaStats
)

# ---------------------- (Optional) DEMO PATHS ----------------------
DEMO_YAML   = r"C:/OneDrive/Study/AI/STN_Final_Term/dataset.yaml"
DEMO_MODEL  = r"C:/OneDrive/Study/AI/STN_Final_Term/models/yolo11m_stn.pt"
DEMO_BGPAIR = r"C:/OneDrive/Study/AI/STN_Final_Term/scripts/bgpair_map.json"


# ---------------------- Early VAL identity window ----------------------
class ValIdentityWindow:
    """Ép STN chạy identity trong validation cho các epoch < until_epoch, và LOG rõ ràng."""
    def __init__(self, ctrl: STNControl, until_epoch: int = 5):
        self.ctrl = ctrl
        self.until = int(until_epoch)

    def on_val_start(self, validator, *a, **k):
        trainer = getattr(validator, "trainer", validator)
        e = int(getattr(trainer, "epoch", 0))
        if e < self.until:
            LOGGER.debug("[STN] validation: identity (window)")
            self.ctrl.on_val_start(validator)
        else:
            LOGGER.debug("[STN] validation: NORMAL (no identity window)")

    def on_val_end(self, validator, *a, **k):
        trainer = getattr(validator, "trainer", validator)
        e = int(getattr(trainer, "epoch", 0))
        if e < self.until:
            self.ctrl.on_val_end(validator)


# ---------------------- Helpers ----------------------
def _resolve_stn_index(ultra_model) -> int | None:
    """
    Tự dò index của SpatialTransformer trong yolo.model.model để tap đúng sau STN.
    Trả về None nếu không tìm thấy.
    """
    try:
        seq = getattr(ultra_model, "model", None)
        if seq is None:
            return None
        for i, m in enumerate(seq):
            cls = type(m).__name__
            if cls.lower() == "spatialtransformer":
                return i
    except Exception:
        pass
    return None


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
        supcon_out=args.supcon_out,  # sẽ cập nhật bằng index STN nếu dò được
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

    # >>> Auto-fix supcon_out to STN index
    stn_idx = _resolve_stn_index(yolo.model)
    if stn_idx is not None:
        if supcon_cfg["supcon_out"] != stn_idx:
            LOGGER.warning(f"[TapSTNFeat] supcon_out={supcon_cfg['supcon_out']} -> {stn_idx} (auto from SpatialTransformer index)")
            supcon_cfg["supcon_out"] = stn_idx
        else:
            LOGGER.info(f"[TapSTNFeat] supcon_out already matches STN index = {stn_idx}")
    else:
        LOGGER.warning(f"[TapSTNFeat] Could not locate SpatialTransformer; keep supcon_out={supcon_cfg['supcon_out']}")

    # push SupCon args early to model.args
    if not isinstance(getattr(yolo.model, "args", None), SimpleNamespace):
        yolo.model.args = SimpleNamespace()
    for k, v in supcon_cfg.items():
        setattr(yolo.model.args, k, v)
    LOGGER.info(f"[SupCon/EARLY] supcon_feat={yolo.model.args.supcon_feat}, on={yolo.model.args.supcon_on}, out={yolo.model.args.supcon_out}")

    # ===================== CALLBACKS =====================
    # A) CORE WIRING/PLUMBING (trainer<->loss, SupCon args)
    inj = InjectSupConArgsMinimal(**supcon_cfg)
    yolo.add_callback("on_pretrain_routine_start", inj.on_pretrain_routine_start)
    yolo.add_callback("on_pretrain_routine_end",   inj.on_pretrain_routine_end)
    yolo.add_callback("on_train_start",            inj.on_train_start)

    yolo.add_callback("on_train_start", LinkTrainerToLoss().on_train_start)
    syncer = SyncEpochToLoss()
    yolo.add_callback("on_train_epoch_start", syncer.on_train_epoch_start)
    yolo.add_callback("on_train_batch_start", syncer.on_train_batch_start)

    # B) SUPCON: lịch bật/tắt + reinforce hyper theo epoch (LOG bổ sung)
    _sched = SupConScheduler(args.supcon_schedule, default_on=args.supcon_on)
    yolo.add_callback("on_train_epoch_start", _sched.on_train_epoch_start)
    yolo.add_callback("on_train_epoch_start", ReinforceSupConToLoss(supcon_cfg.keys()).on_train_epoch_start)

    # C) SAFETY & GUARDS
    attacher  = AttachSupConProjToOptim()
    nan_guard = LossNaNGuard(stop_on_nan=True, save_bad_batch=True)
    yolo.add_callback("on_pretrain_routine_end", attacher.on_train_start)
    yolo.add_callback("on_train_start",          attacher.on_train_start)
    yolo.add_callback("on_fit_epoch_start",      attacher.on_train_start)
    yolo.add_callback("on_train_batch_start",    attacher.on_train_batch_start)
    yolo.add_callback("on_train_batch_end",      attacher.on_train_batch_end)
    yolo.add_callback("on_train_batch_start",    BatchSanityFilter(eps=1e-6).on_train_batch_start)
    yolo.add_callback("on_train_batch_end",      nan_guard.on_train_batch_end)

    _lr_guard = LRGuard()
    yolo.add_callback("on_train_start",       _lr_guard.on_train_start)
    yolo.add_callback("on_train_epoch_start", _lr_guard.on_train_epoch_start)

    # D) PRED DEBUG (nhẹ, không đụng format)
    yolo.add_callback("on_val_batch_end", debug_pred_boxes_on_val)
    yolo.add_callback("on_val_batch_end", ValPredStats(every=1).on_val_batch_end)  # đếm pred/min/median/max

    # E) LAZY DISCOVER PROJECTOR (thêm log nguồn)
    class _DiscoverSupConProj:
        def __init__(self): self.done=False
        def _try_register(self, trainer):
            if self.done: return
            sources = [
                getattr(trainer, "criterion", None),
                getattr(trainer, "loss", None),
                getattr(getattr(trainer, "model", None), "criterion", None),
            ]
            aliases = ("_supcon_proj", "supcon_projector", "projector", "supcon_proj")
            for src in sources:
                if src is None: continue
                for name in aliases:
                    proj = getattr(src, name, None)
                    if proj is not None:
                        try:
                            supcon_register_projector(proj)
                            self.done = True
                            LOGGER.info(f"[DiscoverSupConProj] projector registered from {type(src).__name__}.{name}")
                            return
                        except Exception as ex:
                            LOGGER.warning(f"[DiscoverSupConProj] register failed from {type(src).__name__}.{name}: {ex}")
        def on_pretrain_routine_end(self, trainer, *a, **k): self._try_register(trainer)
        def on_train_start(self, trainer, *a, **k):          self._try_register(trainer)
        def on_fit_epoch_start(self, trainer, *a, **k):      self._try_register(trainer)
        def on_train_batch_end(self, trainer, *a, **k):      self._try_register(trainer)
    _discover = _DiscoverSupConProj()
    yolo.add_callback("on_pretrain_routine_end", _discover.on_pretrain_routine_end)
    yolo.add_callback("on_train_start",          _discover.on_train_start)
    yolo.add_callback("on_fit_epoch_start",      _discover.on_fit_epoch_start)
    yolo.add_callback("on_train_batch_end",      _discover.on_train_batch_end)

    # F) STN CONTROL — freeze ngắn, có warmup alpha, và biên độ vừa phải (đủ để học)
    freeze_ep  = int(args.freeze_epochs)  # mặc định 3
    ctrl = STNControl(
        freeze_epochs=freeze_ep,
        stn_warmup=8,           # mở dần alpha trong 8 epoch
        tmax=0.20,              # cho phép tịnh tiến rộng hơn (20%)
        smin=0.90, smax=1.10,   # scale 0.9–1.1
        log=True
    )

    # Quiet logs cho STN: ⚠️ KHÔNG gọi on_val_start ở đây để tránh ép identity toàn bộ VAL
    _stn_quiet = setup_stn_quiet_logs()
    yolo.add_callback("on_train_epoch_start", lambda tr, *a, **k: _stn_quiet.on_train_epoch_start(tr, ctrl=ctrl))
    # (cố tình bỏ) yolo.add_callback("on_val_start", lambda tr, *a, **k: _stn_quiet.on_val_start(tr, ctrl=ctrl))

    # === STN θ plumbing (force-return flags + capture via forward hook) ===
    _force_theta = ForceReturnTheta()
    _cap_theta   = CaptureThetaFromSTN()
    yolo.add_callback("on_train_start", _force_theta.on_train_start)
    yolo.add_callback("on_val_start",   _force_theta.on_val_start)
    yolo.add_callback("on_train_start", _cap_theta.on_train_start)
    yolo.add_callback("on_val_start",   _cap_theta.on_val_start)
    yolo.add_callback("on_train_end",   _cap_theta.on_train_end)
    yolo.add_callback("on_val_end",     _cap_theta.on_val_end)

    def _log_stn_alpha(trainer, *a, **k):
        e = int(getattr(trainer, "epoch", -1))
        a = getattr(ctrl, "_alpha", None)
        try:
            LOGGER.info(f"[STN/Alpha] epoch={e} alpha={0.0 if a is None else float(a):.3f} (0=off,1=full)")
        except Exception:
            pass
    yolo.add_callback("on_train_epoch_start", ctrl.on_train_epoch_start)
    yolo.add_callback("on_train_epoch_start", _log_stn_alpha)

    # G) THETA publish (KHÔNG wire θ ra head)
    theta_pub = PublishThetaToStateV2(verbose=True)
    yolo.add_callback("on_train_start", theta_pub.on_train_start)
    yolo.add_callback("on_train_end",   theta_pub.on_train_end)
    yolo.add_callback("on_val_start",   theta_pub.on_val_start)
    yolo.add_callback("on_val_end",     theta_pub.on_val_end)

    # ép args validator + seed θ=I cho batch đầu (để tránh batch trống sớm)
    yolo.add_callback("on_val_start", ForceValArgs(conf=0.001, iou=0.50, max_det=2000, agnostic=False).on_val_start)
    yolo.add_callback("on_val_start", SeedIdentityThetaOnValStart(B_hint=int(args.batch)).on_val_start)

    # VAL: chỉ ép identity trong N epoch đầu để không “vỡ” mAP sớm
    _val_id = ValIdentityWindow(ctrl, until_epoch=0)
    yolo.add_callback("on_val_start", _val_id.on_val_start)
    yolo.add_callback("on_val_end",   _val_id.on_val_end)

    # H) FEATURE TAP (tap đúng theo supcon_out đã auto-fix)
    tap = TapSTNFeat()
    yolo.add_callback("on_train_start", tap.on_train_start)
    yolo.add_callback("on_train_end",   tap.on_train_end)

    # I) DEBUG ẢNH + PREVIEW FG/BG PAIR (opt-in)
    if args.debug_images:
        yolo.add_callback("on_train_epoch_end", DebugImages(epochs=range(args.epochs + 1), max_images=5).on_train_epoch_end)
        yolo.add_callback("on_train_epoch_end", DebugBgPairROIs(epochs=range(args.epochs + 1), max_pairs=6).on_train_epoch_end)

    # θ stats: train mỗi 200 batch, val mỗi batch
    yolo.add_callback("on_train_batch_end", ThetaStats(every=500, tag="train").on_train_batch_end)
    yolo.add_callback("on_val_batch_end",   ThetaStats(tag="val").on_val_batch_end)

    # J) VAL DEBUG OVERRIDES + PROBE IoU + CSV GUARD
    apply_val_debug_overrides(yolo, conf=0.001, iou=0.50)

    # Suy ra thư mục GT labels từ dataset.yaml để probe (nếu có)
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
                            parts[i] = "labels"
                            break
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

    # K) TSNE (OFF nếu tsne_every<=0)
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

    # L) LOG % SUPCON
    pct_logger = SupConPercentLogger()
    yolo.add_callback("on_train_batch_end", pct_logger.on_train_batch_end)
    yolo.add_callback("on_train_epoch_end", pct_logger.on_train_epoch_end)

    # M) BG PAIRING LOADER
    if args.pairing:
        if args.batch % 2 != 0:
            LOGGER.warning(f"[Pairing] batch={args.batch} phải CHẴN; tự động giảm còn {args.batch - 1}.")
            args.batch -= 1
        pl = UsePairedLoader(data_yaml=args.yaml, bgpair_map=args.bgpair_map, batch_size=args.batch, seed=run_id)
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
    p.add_argument("--output", default="runs_stn", help="Output root directory for runs (logs, ckpts, images).")
    p.add_argument("--epochs", type=int, default=100, help="Total training epochs.")
    p.add_argument("--batch", type=int, default=8, help="Global batch size.")
    p.add_argument("--imgsz", type=int, default=640, help="Training/validation image size (square).")
    p.add_argument("--runs", type=int, default=1, help="Number of repeated runs (seeds).")
    p.add_argument("--resume", action="store_true", help="Resume from the last checkpoint in the run folder.")
    p.add_argument("--device", default="auto", help="Device selector: 'auto', 'cpu', '0', '0,1', …")
    p.add_argument("--patience", type=int, default=100, help="Early-stopping patience.")

    # STN
    p.add_argument("--freeze_epochs", type=int, default=0, help="Freeze/disable STN for first N epochs.")
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
    p.add_argument("--supcon_out", type=int, default=1)
    p.add_argument("--supcon_min_box", type=int, default=1)
    p.add_argument("--supcon_max_per_class", type=int, default=0)
    p.add_argument("--supcon_gain", type=float, default=1.0)
    p.add_argument("--supcon_temp", type=float, default=0.5)
    p.add_argument("--supcon_warmup", type=int, default=10)
    p.add_argument("--supcon_log", type=int, default=1)
    p.add_argument("--supcon_use_mem", type=int, default=1)
    p.add_argument("--supcon_queue", type=int, default=4096)
    p.add_argument("--supcon_loss_weight", type=float, default=None)
    p.add_argument("--supcon_schedule", type=str, default="0-")  # bật SupCon từ epoch 5
    p.add_argument("--supcon_proj_dim", type=int, default=128)
    p.add_argument("--supcon_proj_hidden", type=int, default=512)
    p.add_argument("--supcon_proj_bn", type=int, default=2, choices=[0, 1, 2])

    # Pairing
    p.add_argument("--pairing", action="store_true", help="Bật paired sampler/negative background loss")
    p.add_argument("--bgpair_map", type=str, default="bgpair_map.json", help="Đường dẫn map abnormal->normal")

    # Demo (no CLI)
    if len(sys.argv) == 1:
        LOGGER.info("[INFO] No CLI arguments supplied — using demo paths")
        args = p.parse_args([])
        args.yaml = DEMO_YAML
        args.model = DEMO_MODEL
        args.debug_images = True
        args.tsne_every = 0
        args.tsne_per_class = 300
        args.pairing = True
        args.bgpair_map = DEMO_BGPAIR
        args.save = 0
        args.save_period = 0
        args.amp = 1
        args.freeze_epochs = 0
        return args

    # Normal CLI
    args = p.parse_args()
    if args.yaml and not Path(args.yaml).exists():
        LOGGER.warning(f"[WARN] {args.yaml} not found.")
    if args.model and not Path(args.model).exists():
        LOGGER.warning(f"[WARN] {args.model} not found.")
    return args


def main():
    apply_logging_filter()  # ẩn bớt SupConDbg nếu có
    args = parse_args()
    for i in range(1, args.runs + 1):
        train_one(args, i)


if __name__ == "__main__":
    main()

# main_stn.py (Rút gọn)
from __future__ import annotations
import sys
from pathlib import Path

# Thêm thư mục gốc (STN_Final_Term) vào sys.path
# để Python ưu tiên import code đã sửa trong thư mục ./ultralytics
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Trỏ ngược 1 cấp (từ /scripts -> /STN_Final_Term)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ==================== KẾT THÚC THÊM CODE ====================

import argparse
from types import SimpleNamespace
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# --- Import các thành phần cốt lõi ---
from ultralytics.utils.stn_pairing import UsePairedLoader
from ultralytics.utils.stn_utils import (
    STNControl,
    SupConScheduler,
    LinkTrainerToLoss,
    SyncEpochToLoss,
    TapSTNFeat,
    AttachSupConProjToOptim,  # QUAN TRỌNG: Sửa lỗi AMP AssertionError
)
# (ntnhan) Import seed từ file helper của bạn
from stn_train_utils import set_seed

# ---------------------- (Optional) DEMO PATHS ----------------------
DEMO_YAML = r"C:/OneDrive/Study/AI/STN_Final_Term/dataset.yaml"
DEMO_MODEL = r"C:/OneDrive/Study/AI/STN_Final_Term/models/yolo11m_stn.pt"
DEMO_BGPAIR = r"C:/OneDrive/Study/AI/STN_Final_Term/scripts/bgpair_map.json"


# ---------------------- Helpers ----------------------
def _resolve_stn_index(ultra_model) -> int | None:
    """
    Tự dò index của SpatialTransformer trong yolo.model.model để tap đúng sau STN.
    """
    try:
        seq = getattr(ultra_model, "model", None)
        if seq is None: return None
        for i, m in enumerate(seq):
            if "SpatialTransformer" in type(m).__name__:
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

    # ---- SupCon config (đẩy sớm vào model.args) ----
    supcon_cfg = {
        "supcon_on": args.supcon_on,
        "supcon_feat": args.supcon_feat,
        "supcon_warp_gt": args.supcon_warp_gt,
        "supcon_out": args.supcon_out,  # Sẽ được auto-fix bên dưới
        "supcon_min_box": args.supcon_min_box,
        "supcon_max_per_class": args.supcon_max_per_class,
        "supcon_gain": args.supcon_gain,
        "supcon_loss_weight": (args.supcon_loss_weight if args.supcon_loss_weight is not None else args.supcon_gain),
        "supcon_temp": args.supcon_temp,
        "supcon_warmup": args.supcon_warmup,
        "supcon_log": args.supcon_log,
        "supcon_use_mem": args.supcon_use_mem,
        "supcon_queue": args.supcon_queue,
        "supcon_schedule": args.supcon_schedule,
        "supcon_proj_dim": args.supcon_proj_dim,
        "supcon_proj_hidden": args.supcon_proj_hidden,
        "supcon_proj_bn": args.supcon_proj_bn,
    }

    # >>> Auto-fix supcon_out to STN index
    stn_idx = _resolve_stn_index(yolo.model)
    if stn_idx is not None:
        if supcon_cfg["supcon_out"] != stn_idx:
            LOGGER.warning(
                f"[TapSTNFeat] supcon_out={supcon_cfg['supcon_out']} -> {stn_idx} (auto from SpatialTransformer index)")
            supcon_cfg["supcon_out"] = stn_idx
        else:
            LOGGER.info(f"[TapSTNFeat] supcon_out already matches STN index = {stn_idx}")
    else:
        LOGGER.warning(f"[TapSTNFeat] Could not locate SpatialTransformer; keep supcon_out={supcon_cfg['supcon_out']}")

    # Đẩy SupCon args sớm vào model.args để loss.py có thể đọc
    if not isinstance(getattr(yolo.model, "args", None), SimpleNamespace):
        yolo.model.args = SimpleNamespace()
    for k, v in supcon_cfg.items():
        setattr(yolo.model.args, k, v)
    LOGGER.info(
        f"[SupCon/EARLY] supcon_feat={yolo.model.args.supcon_feat}, on={yolo.model.args.supcon_on}, out={yolo.model.args.supcon_out}")

    # ===================== CALLBACKS CỐT LÕI =====================

    # --- 1. Kết nối Trainer <-> Loss (để loss đọc được epoch) ---
    yolo.add_callback("on_train_start", LinkTrainerToLoss().on_train_start)
    syncer = SyncEpochToLoss()
    yolo.add_callback("on_train_epoch_start", syncer.on_train_epoch_start)
    yolo.add_callback("on_train_batch_start", syncer.on_train_batch_start)

    # --- 2. Lịch bật/tắt SupCon ---
    _sched = SupConScheduler(args.supcon_schedule, default_on=args.supcon_on)
    yolo.add_callback("on_train_epoch_start", _sched.on_train_epoch_start)

    # --- 4. STN Control (Warmup/Freeze) ---
    ctrl = STNControl(
        freeze_epochs=int(args.freeze_epochs),
        stn_warmup=8,  # mở dần alpha trong 8 epoch
        tmax=0.20,  # cho phép tịnh tiến rộng hơn (20%)
        smin=0.90, smax=1.10,  # scale 0.9–1.1
        log=True
    )
    yolo.add_callback("on_train_epoch_start", ctrl.on_train_epoch_start)
    # Ép STN chạy identity khi validation (quan trọng cho mAP)
    yolo.add_callback("on_val_start", ctrl.on_val_start)
    yolo.add_callback("on_val_end", ctrl.on_val_end)

    # --- 5. TapSTNFeat (Lấy feature map cho SupCon) ---
    tap = TapSTNFeat()
    yolo.add_callback("on_train_start", tap.on_train_start)
    yolo.add_callback("on_train_end", tap.on_train_end)

    # --- 6. BG Pairing Loader ---
    if args.pairing:
        if args.batch % 2 != 0:
            LOGGER.warning(f"[Pairing] batch={args.batch} phải CHẴN; tự động giảm còn {args.batch - 1}.")
            args.batch -= 1
        pl = UsePairedLoader(data_yaml=args.yaml, bgpair_map=args.bgpair_map, batch_size=args.batch, seed=run_id)
        yolo.add_callback("on_train_start", pl.on_train_start)

        # Đảm bảo sampler được cập nhật mỗi epoch (cho DDP)
        if hasattr(pl, "on_fit_epoch_start"):
            yolo.add_callback("on_fit_epoch_start", pl.on_fit_epoch_start)
            LOGGER.info("[Pairing] hook=on_fit_epoch_start attached")
        elif hasattr(pl, "on_train_epoch_start"):
            yolo.add_callback("on_train_epoch_start", pl.on_train_epoch_start)
            LOGGER.info("[Pairing] hook=on_train_epoch_start attached")

        LOGGER.info(f"[Pairing] Enabled with bgpair_map='{args.bgpair_map}', batch={args.batch}")

    # --- 7. Tùy chỉnh các tham số Val (để lưu file .txt) ---
    # (Giữ lại từ code gốc của bạn vì nó cần thiết cho mục tiêu luận văn)
    val_args = {'save_txt': True, 'save_conf': True, 'save': True, 'plots': True, 'conf': 0.001, 'iou': 0.5}
    LOGGER.info(f"[VAL/Debug] overrides applied: {val_args}")

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
        save_period=int(args.save_period),
        amp=bool(args.amp),

        # Truyền các tham số val tùy chỉnh vào
        **val_args
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
    p.add_argument("--supcon_schedule", type=str, default="0-")  # bật SupCon từ epoch 0
    p.add_argument("--supcon_proj_dim", type=int, default=128)
    p.add_argument("--supcon_proj_hidden", type=int, default=512)
    p.add_argument("--supcon_proj_bn", type=int, default=2, choices=[0, 1, 2])

    # Pairing
    p.add_argument("--pairing", action="store_true", help="Bật paired sampler/negative background loss")
    p.add_argument("--bgpair_map", type=str, default="bgpair_map.json", help="Đường dẫn map abnormal->normal")

    # Bỏ các cờ debug (debug_images, tsne_every,...)

    # Demo (no CLI)
    if len(sys.argv) == 1:
        LOGGER.info("[INFO] No CLI arguments supplied — using demo paths")
        args = p.parse_args([])
        args.yaml = DEMO_YAML
        args.model = DEMO_MODEL
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
    # apply_logging_filter() # Đã xóa (để xem log đầy đủ)
    args = parse_args()
    for i in range(1, args.runs + 1):
        train_one(args, i)


if __name__ == "__main__":
    main()
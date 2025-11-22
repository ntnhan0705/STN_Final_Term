# main_stn.py — attach_callbacks “đầy đủ tham số” + register_pairing (safe, no CleanupSTNWrappers, simplify=False)
from __future__ import annotations
import argparse
import logging
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import LOGGER

# tất cả đều lấy từ stn_utils.py
from ultralytics.utils.stn_utils import attach_callbacks, register_pairing, register_val_trap_and_safety

# ===================== Final-eval SAFE PATCH =====================
# Dùng in-memory model cho final_eval (tránh AutoBackend nạp lại rồi fuse làm rụng module custom)
from ultralytics.engine import trainer as _T
from ultralytics.utils.torch_utils import de_parallel

def _final_eval_safe(self):
    m = de_parallel(self.model).eval()
    try:
        # Ưu tiên validate bằng model đang ở RAM
        self.metrics = self.validator(model=m)
        LOGGER.info("[FinalEval] Used in-memory model (no AutoBackend).")
    except Exception as e:
        LOGGER.warning(f"[FinalEval] In-memory failed: {e} -> fallback to file")
        f = self.best if getattr(self, "best", None) and getattr(self.best, "exists", lambda: False)() else self.last
        self.metrics = self.validator(model=str(f))

# Tự phát hiện lớp trainer cần patch (BaseTrainer cho v8.3.124; một số version dùng Trainer)
_BaseT = getattr(_T, "BaseTrainer", None) or getattr(_T, "Trainer", None)
if _BaseT is None:
    raise RuntimeError("Ultralytics API changed: neither BaseTrainer nor Trainer found for patching.")
_BaseT.final_eval = _final_eval_safe
# ================================================================
# ---------------- CLI ----------------

def _add_file_handler(logger: logging.Logger, path: Path, level=logging.INFO):
    """Thêm file handler (tránh nhân đôi)."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == path:
            return
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    fh.setLevel(level)
    logger.addHandler(fh)


def setup_logging(run_dir: Path):
    """Ghi ra stn_train.log + stn_val.log trong thư mục run."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    train_log = run_dir / "stn_train.log"
    val_log = run_dir / "stn_val.log"
    _add_file_handler(root_logger, train_log, level=logging.INFO)
    _add_file_handler(logging.getLogger("ultralytics"), train_log, level=logging.INFO)
    _add_file_handler(logging.getLogger("ultralytics.val"), val_log, level=logging.INFO)
    LOGGER.info(f"[LogFiles] train={train_log} val={val_log}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO-STN + SupCon (đầy đủ callback từ stn_utils)")

    # Basic
    p.add_argument("--yaml",   required=True, help="dataset.yaml")
    p.add_argument("--model",  required=True, help=".pt hoặc model yaml")
    # mặc định đúng project của bạn
    p.add_argument("--output", default=r"C:/OneDrive/Study/AI/STN_Final_Term/runs", help="thư mục runs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch",  type=int, default=8)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--runs",   type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--save", type=int, default=1)
    p.add_argument("--save_period", type=int, default=-1)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--name", type=str, default=None)

    # STN schedule (chỉ phần lịch & log)
    p.add_argument("--freeze_epochs", type=int, default=10,
                   help="số epoch đầu giữ STN ở chế độ identity (không học)")
    p.add_argument("--stn_warmup", type=int, default=3,
                   help="số epoch chuyển dần từ identity -> full STN")
    p.add_argument("--stn_tmax", type=float, default=0.20,
                   help="biên độ jitter affine tối đa (tỉ lệ theo chiều ảnh)")
    p.add_argument("--stn_smin", type=float, default=0.90,
                   help="scale tối thiểu cho STN")
    p.add_argument("--stn_smax", type=float, default=1.10,
                   help="scale tối đa cho STN")
    p.add_argument("--stn_val_identity", type=int, default=0,
                   help="1 => ép STN identity khi chạy validation")
    p.add_argument("--stn_log", type=int, default=1,
                   help="1 => log mode STN mỗi epoch")

    # STN regularizer + STN grad scale
    p.add_argument("--stn_reg", type=float, default=0.0,
                   help="weight cho STN regularization thêm vào box_loss")
    p.add_argument("--stn_grad_mult", type=float, default=0.2,
                   help="scale riêng gradient cho STN (0.2 => update nhẹ)")

    # Pairing
    p.add_argument("--pairing", action="store_true")
    p.add_argument("--bgpair_map", type=str,
                   default=r"C:\OneDrive\Study\AI\STN_Final_Term\pairing\bgpair_map.json")

    # SupCon (tham số được “inject” vào loss + lịch bật/tắt)
    p.add_argument("--supcon_start_epoch", type=int, default=5)  # dùng SupConScheduler kiểu "3-"
    p.add_argument("--supcon_feat", type=str, default="stn")
    p.add_argument("--supcon_warp_gt", type=int, default=0)
    p.add_argument("--supcon_out", type=int, default=7)
    p.add_argument("--supcon_min_box", type=int, default=1)
    p.add_argument("--supcon_max_per_class", type=int, default=0)
    p.add_argument("--supcon_gain", type=float, default=0.5)
    p.add_argument("--supcon_temp", type=float, default=0.5)
    p.add_argument("--supcon_warmup", type=int, default=5)
    p.add_argument("--supcon_log", type=int, default=1)
    p.add_argument("--supcon_use_mem", type=int, default=1)
    p.add_argument("--supcon_queue", type=int, default=4096)
    p.add_argument("--supcon_loss_weight", type=float, default=None)
    p.add_argument("--supcon_neg_iou_ignore", type=float, default=0.20)
    p.add_argument("--supcon_neg_sameimg_only", type=int, default=1)
    p.add_argument("--supcon_neg_cap", type=int, default=2048)
    p.add_argument("--supcon_neg_per_pos", type=float, default=2.0)
    p.add_argument("--supcon_min_neg_w", type=float, default=1e-3)
    p.add_argument("--supcon_log_n", type=int, default=6)
    p.add_argument("--supcon_proj_dim", type=int, default=0)
    p.add_argument("--supcon_proj_hidden", type=int, default=512)
    p.add_argument("--supcon_proj_bn", type=int, default=1)
    p.add_argument("--supcon_proj_lr", type=float, default=1e-3,
                   help="learning rate riêng cho SupCon projector param group")

    # Debug
    p.add_argument("--debug_images_every", type=int, default=5)
    p.add_argument("--debug", type=int, default=0, help="1 => gắn callback debug thuần log")

    # Dev fallback
    import sys, torch
    if len(sys.argv) == 1:
        LOGGER.info("[Info] No CLI args provided; using dev fallback paths")
        dev_cli = [
            "--yaml",   r"C:/OneDrive/Study/AI/STN_Final_Term/dataset.yaml",
            "--model",  r"C:/OneDrive/Study/AI/STN_Final_Term/models/yolo11m_stn.pt",
            "--output", r"C:/OneDrive/Study/AI/STN_Final_Term/runs",
            "--pairing",
        ]
        args = p.parse_args(dev_cli)
    else:
        args = p.parse_args()

    # Chuẩn hoá & in ra đường dẫn thực tế
    args.yaml = str(Path(args.yaml).resolve())
    args.model = str(Path(args.model).resolve())
    args.output = str(Path(args.output).resolve())
    args.bgpair_map = str(Path(args.bgpair_map).resolve())

    LOGGER.info(
        f"[Paths] data={args.yaml}\n"
        f"        model={args.model}\n"
        f"        output={args.output}\n"
        f"        bgpair_map={args.bgpair_map}"
    )

    # Thiết bị
    dev = str(args.device).lower()
    if dev in ("auto", "cuda", "gpu"):
        if torch.cuda.is_available():
            args.device = "0"
            LOGGER.info("[Device] Using CUDA device 0")
        else:
            args.device = "cpu"
            LOGGER.warning("[Device] CUDA not available -> falling back to CPU")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    return args

# ---------------- Gắn callback & train ----------------
def train_one(args: argparse.Namespace, run_idx: int) -> None:
    run_name = args.name or f"{Path(args.model).stem}_run{run_idx:03d}"
    run_dir = Path(args.output) / run_name
    LOGGER.info(f"[Run] name={run_name} | save_dir={run_dir}")
    # Always write logs at the root runs directory (not inside each run folder)
    setup_logging(Path(args.output))
    yolo = YOLO(args.model)

    # 1) Lịch SupCon dạng "start-"
    supcon_schedule_str = f"{max(1, int(args.supcon_start_epoch))}-"

    # 2) attach toàn bộ callback — KHÔNG gắn CleanupSTNWrappers để tránh set None vào ModuleList
    supcon_proj_cfg = None
    if int(args.supcon_proj_dim) > 0:
        supcon_proj_cfg = dict(
            in_dim=1,
            out_dim=int(args.supcon_proj_dim),
            hidden=int(args.supcon_proj_hidden),
            bn=int(args.supcon_proj_bn),
            lr=float(args.supcon_proj_lr),
        )
    attach_callbacks(
        yolo,
        stn_cfg=dict(
            freeze_epochs=int(args.freeze_epochs),
            stn_warmup=int(args.stn_warmup),
            tmax=float(args.stn_tmax),
            smin=float(args.stn_smin),
            smax=float(args.stn_smax),
            val_identity=int(args.stn_val_identity),
            log=int(args.stn_log),
        ),
        publish_theta=True,

        supcon_inject=dict(
            supcon_feat=str(args.supcon_feat),
            supcon_warp_gt=int(args.supcon_warp_gt),
            supcon_out=int(args.supcon_out),
            supcon_min_box=int(args.supcon_min_box),
            supcon_max_per_class=int(args.supcon_max_per_class),
            supcon_gain=float(args.supcon_gain),
            supcon_temp=float(args.supcon_temp),
            supcon_warmup=int(args.supcon_warmup),
            supcon_log=int(args.supcon_log),
            supcon_use_mem=int(args.supcon_use_mem),
            supcon_queue=int(args.supcon_queue),
            supcon_loss_weight=(None if args.supcon_loss_weight is None else float(args.supcon_loss_weight)),
            supcon_neg_iou_ignore=float(args.supcon_neg_iou_ignore),
            supcon_neg_sameimg_only=int(args.supcon_neg_sameimg_only),
            supcon_neg_cap=int(args.supcon_neg_cap),
            supcon_neg_per_pos=float(args.supcon_neg_per_pos),
            supcon_min_neg_w=float(args.supcon_min_neg_w),
            supcon_log_n=int(args.supcon_log_n),
            supcon_proj_dim=int(args.supcon_proj_dim),
            supcon_proj_hidden=int(args.supcon_proj_hidden),
            supcon_proj_bn=int(args.supcon_proj_bn),
            stn_reg=float(args.stn_reg),

        ),
        supcon_schedule=supcon_schedule_str,         # ví dụ: "3-"
        supcon_reinforce_keys=[
            "supcon_feat", "supcon_warp_gt", "supcon_out", "supcon_min_box", "supcon_max_per_class",
            "supcon_gain", "supcon_temp", "supcon_warmup", "supcon_use_mem", "supcon_queue",
            "supcon_loss_weight", "supcon_neg_iou_ignore", "supcon_neg_sameimg_only", "supcon_neg_cap",
            "supcon_neg_per_pos", "supcon_min_neg_w", "supcon_log_n", "supcon_proj_dim", "supcon_proj_hidden",
            "supcon_proj_bn", "supcon_on",
        ],
        supcon_tap=dict(out_idx=int(args.supcon_out)),
        supcon_proj_attach=supcon_proj_cfg,
        supcon_percent_logger=bool(int(args.supcon_log)),

        link_trainer_to_loss=True,
        sync_epoch_to_loss=True,
        nan_guard=dict(stop_on_nan=True, save_bad_batch=True),
        batch_sanity=dict(eps=1e-6),

        enable_val_loss=False,
        val_force_args=None,
        val_debug_overrides=None,
        val_trap=False,

        debug_images=dict(epochs={0, 1, 2, 5, 10}, max_images=5),
        debug_bgpair=dict(epochs={0, 1, 2, 5, 10}, max_pairs=4),

        results_csv_guard=True,
        final_eval_fix=True,
        save_last_best_only=True,
    )
    # --- Bật gói ValTrap + Safety: predict path + NMS + log chi tiết ---
    register_val_trap_and_safety(
        yolo,
        conf=0.01,     # hoặc 0.10 tuỳ bạn muốn mAP nghiêm hay thoáng
        iou=0.10,      # cho CXR bạn đang dùng 0.10 nên giữ nguyên để so sánh
        max_det=300,
        half_if_cuda=True,
        nms=True,
        post_loss_k=0,  # 0 = không cần tính val-loss post, tập trung mAP trước
        log_file=Path(args.output) / "stn_val.log",
    )

    # Pairing
    if args.pairing:
        register_pairing(yolo, bgpair_map=args.bgpair_map, batch_size=args.batch)

    # 3) Train — ép simplify=False để tránh rủi ro “rụng module” khi final_eval
    yolo.train(
        data=args.yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.output,
        name=run_name,
        resume=args.resume,
        patience=args.patience,
        save=bool(args.save),
        save_period=args.save_period,
        amp=bool(args.amp),
        val=True,
        split="val",
        exist_ok=False,
        verbose=True,

        # để validator có biểu đồ (PR/F1/confusion)
        plots=True,

        # RẤT QUAN TRỌNG: tránh simplify trong pipeline cuối để không mất module custom
        simplify=False,
    )

def main():
    args = parse_args()

    # Cấy một số tham số “nghiên cứu” vào ENV để Trainer đọc được
    import os
    os.environ["STN_GRAD_MULT"] = str(args.stn_grad_mult)

    # When running from PyCharm, pre-attach file handlers so log files are created early
    if os.environ.get("PYCHARM_HOSTED"):
        setup_logging(Path(args.output))
        LOGGER.info(f"[PyCharm] Pre-attached file handlers at {Path(args.output)}")

    for i in range(1, int(args.runs) + 1):
        LOGGER.info(f"===== RUN {i}/{args.runs} =====")
        train_one(args, i)


if __name__ == "__main__":
    main()

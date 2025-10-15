from __future__ import annotations
"""
main_stn.py — YOLO + STN + SupCon (inject từ ngoài, không phá Ultralytics)

- KHÔNG truyền supcon_* vào model.train(...)
- Dùng model.add_callback(...) để inject supcon_* vào model.args và criterion.hyp
- Đóng băng STN vài epoch đầu, TSNE, debug ảnh STN bằng callback
"""

import sys, random
from types import SimpleNamespace
from pathlib import Path
import argparse
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.stn_utils import (
    STNControl, DebugImages, TSNEEmbeddings as EmbeddingTSNE, TapSTNFeat
)

# --- DEMO paths (đổi theo máy bạn nếu cần) ---
DEMO_YAML  = r"C:/OneDrive/Study/AI/STN_Final_Term/dataset.yaml"
DEMO_MODEL = r"C:/OneDrive/Study/AI/STN_Final_Term/models/yolo11m_stn.pt"
# ---------------------------------------------

# --------------------- utils ---------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collect_supcon_cfg(args: argparse.Namespace) -> dict:
    return {
        "supcon_on":            args.supcon_on,
        "supcon_feat":          args.supcon_feat,
        "supcon_warp_gt":       args.supcon_warp_gt,
        "supcon_out":           args.supcon_out,
        "supcon_min_box":       args.supcon_min_box,
        "supcon_max_per_class": args.supcon_max_per_class,
        "supcon_gain":          args.supcon_gain,
        "supcon_loss_weight":   args.supcon_loss_weight if args.supcon_loss_weight is not None else args.supcon_gain,
        "supcon_temp":          args.supcon_temp,
        "supcon_warmup":        args.supcon_warmup,
        "supcon_log":           args.supcon_log,
        "supcon_use_mem":       args.supcon_use_mem,
        "supcon_queue":         args.supcon_queue,
        "supcon_schedule":      args.supcon_schedule,
        "supcon_proj_dim":      args.supcon_proj_dim,
        "supcon_proj_hidden":   args.supcon_proj_hidden,
        "supcon_proj_bn":       args.supcon_proj_bn,
    }

def make_supcon_injector(supcon_cfg: dict) -> dict:
    """
    Trả về dict {hook_name: fn} để đăng ký bằng model.add_callback().
    - on_pretrain_routine_start: inject sớm vào model.args & criterion.hyp
    - on_train_start: reinforce (phòng khi Ultralytics recreate criterion)
    """
    def _inject_to_model_args(model):
        args_ns = getattr(model, "args", None)
        if args_ns is None or not isinstance(args_ns, SimpleNamespace):
            args_ns = SimpleNamespace()
        for k, v in supcon_cfg.items():
            setattr(args_ns, k, v)
        model.args = args_ns
        setattr(model, "_supcon_cfg", supcon_cfg.copy())

    def _inject_to_criterion(trainer):
        model = trainer.model
        crit = getattr(model, "criterion", None)
        if crit is None:
            return
        hyp = getattr(crit, "hyp", None)
        if hyp is None or not isinstance(hyp, dict):
            hyp = {}
        hyp.update(supcon_cfg)
        crit.hyp = hyp
        # nếu loss có thuộc tính riêng
        try:
            setattr(crit, "supcon_cfg", supcon_cfg.copy())
        except Exception:
            pass

    def on_pretrain_routine_start(trainer):
        _inject_to_model_args(trainer.model)
        _inject_to_criterion(trainer)
        LOGGER.info("[INJECT OK] SupCon -> model.args & criterion.hyp")

    def on_train_start(trainer):
        _inject_to_model_args(trainer.model)
        _inject_to_criterion(trainer)

    return {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_start": on_train_start,
    }

# --------------------- train one run ---------------------
def train_one(args: argparse.Namespace, run_id: int):
    set_seed(run_id)
    LOGGER.info(f"===== RUN {run_id}/{args.runs} =====")

    model = YOLO(args.model)
    save_dir = Path(args.output) / f"{Path(args.model).stem}_run{run_id}"

    # SupCon: chỉ lưu cấu hình ở đây, inject bằng callback (không truyền vào .train)
    supcon_cfg = collect_supcon_cfg(args)
    LOGGER.info(f"[SupCon] injected (from main_stn.py): {supcon_cfg}")

    # Đăng ký injector callbacks
    for name, fn in make_supcon_injector(supcon_cfg).items():
        model.add_callback(name, fn)

    # Thiết bị
    device_arg = "0" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    # === Đóng băng STN vài epoch đầu ===
    model.add_callback("on_train_epoch_start",
                       STNControl(args.freeze_epochs).on_train_epoch_start)

    # === Hook feature sau STN (để TSNE / debug dùng) ===
    tap = TapSTNFeat()
    model.add_callback("on_train_start", tap.on_train_start)
    model.add_callback("on_train_end",   tap.on_train_end)

    # === Debug ảnh STN mỗi epoch ===
    if args.debug_images:
        model.add_callback(
            "on_train_epoch_end",
            DebugImages(epochs=range(args.epochs + 1), max_images=5).on_train_epoch_end
        )

    # === TSNE ROI embeddings theo chu kỳ ===
    if args.tsne_every > 0:
        model.add_callback(
            "on_train_epoch_end",
            EmbeddingTSNE(
                every=args.tsne_every,
                loader="train",      # đổi thành "val" nếu muốn chạy trên tập val
                per_class=args.tsne_per_class,
                total_max=0,
                max_batches=0,
                use_roialign=True,
                roialign_out=1,
                min_feat_wh=1,
                pca_dim=128,
                min_channels=1024,
                verbose=True
            ).on_train_epoch_end
        )

    # TUYỆT ĐỐI KHÔNG truyền supcon_* hay callbacks= vào đây
    model.train(
        data=args.yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device_arg,
        resume=args.resume,
        project=str(save_dir.parent),
        name=save_dir.name,
        patience=args.patience,
        val=True,
        verbose=True,
        seed=run_id,
    )
    LOGGER.info("✔ Training finished")

# --------------------- cli ---------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO-STN + SupCon (clean)")

    # Train chung
    p.add_argument("--yaml", help="Dataset YAML")
    p.add_argument("--model", help="Model .pt")
    p.add_argument("--output", default="runs_stn")
    p.add_argument("--epochs", type=int, default=121)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--patience", type=int, default=121)

    # --- STN & debug ---
    p.add_argument("--freeze_epochs", type=int, default=61,
                   help="Số epoch đầu tắt STN (đóng băng), sau mốc này STN mới bật")
    p.add_argument("--debug_images", action="store_true",
                   help="Lưu ảnh so sánh STN (gốc vs warp) mỗi epoch")

    # --- TSNE ---
    p.add_argument("--tsne_every", type=int, default=20,
                   help=">0: vẽ TSNE ROI embeddings mỗi N epoch (0=tắt)")
    p.add_argument("--tsne_per_class", type=int, default=2000,
                   help="Tối đa số ROI/lớp dùng cho TSNE")

    # --- SupCon (chỉ để collect, không truyền vào trainer) ---
    p.add_argument("--supcon_on", type=int, default=1)
    p.add_argument("--supcon_feat", type=str, default="stn")
    p.add_argument("--supcon_warp_gt", type=int, default=0)
    p.add_argument("--supcon_out", type=int, default=7)
    p.add_argument("--supcon_min_box", type=int, default=1)
    p.add_argument("--supcon_max_per_class", type=int, default=0)
    p.add_argument("--supcon_gain", type=float, default=2.5)
    p.add_argument("--supcon_temp", type=float, default=0.2)
    p.add_argument("--supcon_warmup", type=int, default=0)
    p.add_argument("--supcon_log", type=int, default=0)
    p.add_argument("--supcon_use_mem", type=int, default=1)
    p.add_argument("--supcon_queue", type=int, default=4096)
    p.add_argument("--supcon_loss_weight", type=float, default=None)
    p.add_argument("--supcon_schedule", type=str, default="0-")
    p.add_argument("--supcon_proj_dim", type=int, default=128)
    p.add_argument("--supcon_proj_hidden", type=int, default=512)
    p.add_argument("--supcon_proj_bn", type=int, default=1)

    # Không truyền gì -> dùng demo + bật sẵn debug ảnh
    if len(sys.argv) == 1:
        LOGGER.info("[INFO] No CLI arguments supplied  using defaults + demo paths")
        args = p.parse_args([])
        args.yaml = DEMO_YAML
        args.model = DEMO_MODEL
        args.debug_images = True          # bật ảnh debug STN khi chạy demo
        args.tsne_every = 20              # giữ TSNE mặc định
        args.tsne_per_class = 2000
        return args

    args = p.parse_args()
    for k in ("yaml", "model"):
        if getattr(args, k, None) is None:
            p.error(f"Missing --{k}")
    return args

# --------------------- entry ---------------------
def main():
    args = parse_args()
    for i in range(1, args.runs + 1):
        train_one(args, i)

if __name__ == "__main__":
    main()

# main_stn.py
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from ultralytics.utils.stn_utils import (
    STNControl, DebugImages,
    ReinforceSupConToLoss, SupConScheduler,
    LinkTrainerToLoss, SyncEpochToLoss, TapSTNFeat,
    TSNEEmbeddings as EmbeddingTSNE,
    SupConPercentLogger, AttachSupConProjToOptim,
    InjectSupConArgsMinimal,   # <<< THÊM DÒNG NÀY
)


DEMO_YAML  = r"C:/OneDrive/Study/AI/STN_Final_Term/dataset.yaml"
DEMO_MODEL = r"C:/OneDrive/Study/AI/STN_Final_Term/models/yolo11m_stn.pt"


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one(args: argparse.Namespace, run_id: int):
    set_seed(run_id)
    LOGGER.info(f"===== RUN {run_id}/{args.runs} =====")

    yolo = YOLO(args.model)
    save_dir = Path(args.output) / f"{Path(args.model).stem}_run{run_id}"

    # --- SupCon config (DO NOT pass to yolo.train) ---
    supcon_cfg = dict(
        supcon_on=args.supcon_on,
        supcon_feat=args.supcon_feat,              # 'stn' to use feature AFTER STN
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

    # Early best-effort: set on the CURRENT yolo.model (trainer may clone later)
    if not isinstance(getattr(yolo.model, "args", None), SimpleNamespace):
        yolo.model.args = SimpleNamespace()
    for k, v in supcon_cfg.items():
        setattr(yolo.model.args, k, v)
    LOGGER.info(f"[SupCon/EARLY] model.args set: supcon_feat={yolo.model.args.supcon_feat}, "
                f"on={yolo.model.args.supcon_on}, queue={yolo.model.args.supcon_queue}")

    # --- Register callbacks (order matters) ---
    # 0) Two-phase injection to FINAL trainer.model + criterion
    inj = InjectSupConArgsMinimal(**supcon_cfg)
    yolo.add_callback("on_pretrain_routine_start", inj.on_pretrain_routine_start)  # cache early (trainer._supcon_cfg + ENV)
    yolo.add_callback("on_pretrain_routine_end",   inj.on_pretrain_routine_end)    # mirror to model.args & criterion.hyp
    yolo.add_callback("on_train_start",            inj.on_train_start)             # mirror again if criterion recreated

    # 1) SupCon schedule and reinforce into loss every epoch
    yolo.add_callback("on_train_epoch_start",
                      SupConScheduler(args.supcon_schedule, default_on=args.supcon_on).on_train_epoch_start)
    yolo.add_callback("on_train_epoch_start",
                      ReinforceSupConToLoss(supcon_cfg.keys()).on_train_epoch_start)

    # 2) Link trainer into loss & sync epoch/batch indices; also attach proj params
    yolo.add_callback("on_train_start", LinkTrainerToLoss().on_train_start)
    syncer = SyncEpochToLoss()
    attacher = AttachSupConProjToOptim()
    yolo.add_callback("on_train_start", attacher.on_train_start)
    yolo.add_callback("on_train_batch_start", attacher.on_train_start)
    yolo.add_callback("on_train_epoch_start", syncer.on_train_epoch_start)
    yolo.add_callback("on_train_batch_start", syncer.on_train_batch_start)

    # 3) Freeze/disable STN for first N epochs, then enable
    yolo.add_callback("on_train_epoch_start", STNControl(args.freeze_epochs).on_train_epoch_start)

    # 4) Hook STN feature & theta (for SupCon/TSNE and debug)
    tap = TapSTNFeat()
    yolo.add_callback("on_train_start", tap.on_train_start)
    yolo.add_callback("on_train_end",   tap.on_train_end)

    # 5) STN debug images
    if args.debug_images:
        yolo.add_callback("on_train_epoch_end",
                          DebugImages(epochs=range(args.epochs + 1), max_images=5).on_train_epoch_end)

    # 6) TSNE embeddings (ROI from feature AFTER STN)
    if args.tsne_every > 0:
        yolo.add_callback(
            "on_train_epoch_end",
            EmbeddingTSNE(
                every=args.tsne_every, loader="train",
                per_class=args.tsne_per_class, total_max=0, max_batches=0,
                use_roialign=True, roialign_out=1,
                min_feat_wh=1, pca_dim=128, min_channels=128, verbose=True
            ).on_train_epoch_end
        )

    # 7) % contribution of supcon_loss
    pct_logger = SupConPercentLogger()
    yolo.add_callback("on_train_batch_end", pct_logger.on_train_batch_end)
    yolo.add_callback("on_train_epoch_end", pct_logger.on_train_epoch_end)

    LOGGER.info("[SupCon] injected: { " + ", ".join(f"{k}={v}" for k, v in supcon_cfg.items()) + " }")
    LOGGER.info(f"[SanityCheck] Before train: model.args.supcon_feat={getattr(yolo.model.args, 'supcon_feat', None)}")

    # --- Train
    device_arg = "0" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    yolo.train(
        data=args.yaml, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=device_arg, resume=args.resume, project=str(save_dir.parent),
        name=save_dir.name, patience=args.patience, val=True, verbose=True,
        seed=run_id, amp=bool(args.amp),
    )
    LOGGER.info("✔ Training finished\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO-STN + SupCon training")

    # training common
    p.add_argument("--yaml")
    p.add_argument("--model")
    p.add_argument("--output", default="runs_stn")
    p.add_argument("--epochs", type=int, default=61)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--patience", type=int, default=61)
    p.add_argument("--freeze_epochs", type=int, default=21,
                   help="Freeze/disable STN for first N epochs")
    p.add_argument("--debug_images", action="store_true")
    p.add_argument("--tsne_every", type=int, default=10)
    p.add_argument("--tsne_per_class", type=int, default=256)
    p.add_argument("--amp", type=int, default=1)

    # SupCon (internal, DO NOT pass to yolo.train)
    p.add_argument("--supcon_on", type=int, default=1)
    p.add_argument("--supcon_feat", type=str, default="stn")
    p.add_argument("--supcon_warp_gt", type=int, default=0)
    p.add_argument("--supcon_out", type=int, default=7)
    p.add_argument("--supcon_min_box", type=int, default=1)
    p.add_argument("--supcon_max_per_class", type=int, default=0)
    p.add_argument("--supcon_gain", type=float, default=2.5)
    p.add_argument("--supcon_temp", type=float, default=0.2)
    p.add_argument("--supcon_warmup", type=int, default=0)
    p.add_argument("--supcon_log", type=int, default=1)
    p.add_argument("--supcon_use_mem", type=int, default=1)
    p.add_argument("--supcon_queue", type=int, default=4096)
    p.add_argument("--supcon_loss_weight", type=float, default=None)
    p.add_argument("--supcon_schedule", type=str, default="0-")
    p.add_argument("--supcon_proj_dim", type=int, default=128)
    p.add_argument("--supcon_proj_hidden", type=int, default=512)
    p.add_argument("--supcon_proj_bn", type=int, default=1)

    if len(sys.argv) == 1:
        LOGGER.info("[INFO] No CLI arguments supplied  using defaults + demo paths")
        args = p.parse_args([])
        args.yaml = DEMO_YAML
        args.model = DEMO_MODEL
        args.debug_images = True
        args.tsne_every = 20
        args.tsne_per_class = 2000
        return args

    args = p.parse_args()
    if not Path(args.yaml).exists():
        LOGGER.warning(f"[WARN] {args.yaml} not found.")
    if not Path(args.model).exists():
        LOGGER.warning(f"[WARN] {args.model} not found.")
    return args


def main():
    args = parse_args()
    for i in range(1, args.runs + 1):
        train_one(args, i)


if __name__ == "__main__":
    main()

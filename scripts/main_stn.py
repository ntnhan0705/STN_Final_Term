# main_stn.py
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.stn_pairing import UsePairedLoader

from ultralytics.utils.stn_utils import (
    STNControl, DebugImages, DebugBgPairROIs,  # <-- có DebugBgPairROIs
    ReinforceSupConToLoss, SupConScheduler,
    LinkTrainerToLoss, SyncEpochToLoss, TapSTNFeat,
    TSNEEmbeddings as EmbeddingTSNE,
    SupConPercentLogger, AttachSupConProjToOptim,
    InjectSupConArgsMinimal, PeekBatch,ForceSTNIdentityOnVal,
)

DEMO_YAML  = r"C:/OneDrive/Study/AI/STN_Final_Term/dataset.yaml"
DEMO_MODEL = r"C:/OneDrive/Study/AI/STN_Final_Term/models/yolo11m_stn.pt"
DEMO_BGPAIR = r"C:/OneDrive/Study/AI/STN_Final_Term/scripts/bgpair_map.json"  # sửa nếu khác


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

    # --- SupCon config (KHÔNG pass trực tiếp vào yolo.train) ---
    supcon_cfg = dict(
        supcon_on=args.supcon_on,
        supcon_feat=args.supcon_feat,              # 'stn' → feature sau STN
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

    # Set sớm vào yolo.model.args (trainer có thể clone sau)
    if not isinstance(getattr(yolo.model, "args", None), SimpleNamespace):
        yolo.model.args = SimpleNamespace()
    for k, v in supcon_cfg.items():
        setattr(yolo.model.args, k, v)
    LOGGER.info(f"[SupCon/EARLY] model.args set: supcon_feat={yolo.model.args.supcon_feat}, "
                f"on={yolo.model.args.supcon_on}, queue={yolo.model.args.supcon_queue}")

    # --- Đăng ký callbacks (thứ tự quan trọng) ---
    # 0) Two-phase inject → trainer.model + criterion
    inj = InjectSupConArgsMinimal(**supcon_cfg)
    yolo.add_callback("on_pretrain_routine_start", inj.on_pretrain_routine_start)
    yolo.add_callback("on_pretrain_routine_end",   inj.on_pretrain_routine_end)
    yolo.add_callback("on_train_start",            inj.on_train_start)

    # 1) Lịch bật/tắt SupCon + reinforce hyper mỗi epoch
    yolo.add_callback("on_train_epoch_start",
                      SupConScheduler(args.supcon_schedule, default_on=args.supcon_on).on_train_epoch_start)
    yolo.add_callback("on_train_epoch_start",
                      ReinforceSupConToLoss(supcon_cfg.keys()).on_train_epoch_start)

    # 2) Link trainer vào loss & sync epoch/batch; gắn proj params vào optimizer
    yolo.add_callback("on_train_start", LinkTrainerToLoss().on_train_start)
    syncer = SyncEpochToLoss()
    attacher = AttachSupConProjToOptim()

    # --- FIX: ensure SupCon projector param-group has a real LR ---
    class FixSupConProjLR:
        def on_train_start(self, trainer):
            opt = getattr(trainer, "optimizer", None)
            if opt is None or not getattr(opt, "param_groups", None):
                return
            base_lr = opt.param_groups[0].get("lr", 0.001)
            base_wd = opt.param_groups[0].get("weight_decay", 0.0)

            fixed = 0
            for pg in opt.param_groups:
                # vá những nhóm có lr=0.0 nhưng có params cần học
                if pg.get("lr", 0.0) == 0.0 and any(p.requires_grad for p in pg.get("params", [])):
                    pg["lr"] = base_lr
                    if "weight_decay" not in pg:
                        pg["weight_decay"] = base_wd
                    fixed += 1
            if fixed:
                LOGGER.info(f"[SupCon/FIX] Restored lr for {fixed} param-group(s) (projector?) to {base_lr}")

    # đăng ký ngay sau attacher
    yolo.add_callback("on_train_start", FixSupConProjLR().on_train_start)

    yolo.add_callback("on_train_start", attacher.on_train_start)
    yolo.add_callback("on_train_batch_start", attacher.on_train_start)
    yolo.add_callback("on_train_epoch_start", syncer.on_train_epoch_start)
    yolo.add_callback("on_train_batch_start", syncer.on_train_batch_start)

    # ==== STN trust-region + warmup (giữ 1 LR chung) ====

    # 3.1) Warmup: ramp alpha đủ dài (ít nhất 12 epoch)
    freeze_ep = int(args.freeze_epochs)
    stn_warmup = int(args.supcon_warmup) if int(args.supcon_warmup) > 0 else 0
    stn_warmup = max(12, stn_warmup, freeze_ep // 2)  # đảm bảo >=12

    # 3.2) Trust-region ban đầu: rất chặt
    tmax0, smin0, smax0 = 0.20, 0.90, 1.10  # dịch ≤20%, scale trong ±10%

    ctrl = STNControl(
        freeze_epochs=freeze_ep,
        stn_warmup=stn_warmup,
        tmax=tmax0, smin=smin0, smax=smax0,
        log=True,
    )

    # 3.3) Lịch nới lỏng biên (sau khi model bớt "rung")
    def _relax_stn_bounds(trainer):
        e = int(getattr(trainer, "epoch", 0))
        # nới nhẹ sau 20 epoch
        if e == 20:
            ctrl.tmax, ctrl.smin, ctrl.smax = 0.25, 0.85, 1.15
        # nới tiếp (nếu cần) sau 40 epoch
        if e == 40:
            ctrl.tmax, ctrl.smin, ctrl.smax = 0.30, 0.80, 1.20

    # 3.4) Đăng ký callback (đặt _relax_stn_bounds trước để giá trị cập nhật có hiệu lực)
    yolo.add_callback("on_train_epoch_start", _relax_stn_bounds)
    yolo.add_callback("on_train_epoch_start", ctrl.on_train_epoch_start)
    yolo.add_callback("on_val_start", ctrl.on_val_start)
    yolo.add_callback("on_val_end", ctrl.on_val_end)
    _stn_bypass = ForceSTNIdentityOnVal()
    yolo.add_callback("on_val_start", _stn_bypass.on_val_start)
    yolo.add_callback("on_val_end", _stn_bypass.on_val_end)
    # (tuỳ chọn) scale gradient STN nhưng vẫn 1 LR chung
    setattr(args, "stn_grad_mult", float(getattr(args, "stn_grad_mult", 0.1)))  # 0.1–0.2

    # 4) Hook lấy STN feature/theta (phục vụ SupCon/TSNE & debug)
    tap = TapSTNFeat()
    yolo.add_callback("on_train_start", tap.on_train_start)
    yolo.add_callback("on_train_end",   tap.on_train_end)

    # 5) Ảnh debug STN + preview cặp FG|BG (bbox nền)
    if args.debug_images:
        yolo.add_callback(
            "on_train_epoch_end",
            DebugImages(epochs=range(args.epochs + 1), max_images=5).on_train_epoch_end
        )
        # Lưu ảnh: runs_stn/<run>/bgpair_preview/epoch_XXX.jpg
        yolo.add_callback(
            "on_train_epoch_end",
            DebugBgPairROIs(epochs=range(args.epochs + 1), max_pairs=6).on_train_epoch_end
        )

    # 6) TSNE embeddings từ ROI sau STN
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

    # 7) Log % đóng góp supcon_loss
    pct_logger = SupConPercentLogger()
    yolo.add_callback("on_train_batch_end", pct_logger.on_train_batch_end)
    yolo.add_callback("on_train_epoch_end", pct_logger.on_train_epoch_end)

    # 8) (MỚI) Thay DataLoader bằng paired loader khi bật --pairing
    if args.pairing:
        if args.batch % 2 != 0:
            LOGGER.warning(f"[Pairing] batch={args.batch} phải CHẴN; tự động giảm còn {args.batch - 1}.")
            args.batch -= 1

        # TẠO 1 INSTANCE DUY NHẤT và đăng ký HAI callback
        pl = UsePairedLoader(
            data_yaml=args.yaml,
            bgpair_map=args.bgpair_map,
            batch_size=args.batch,
            seed=run_id,
        )
        yolo.add_callback("on_train_start", pl.on_train_start)
        yolo.add_callback("on_fit_epoch_start", pl.on_fit_epoch_start)  # set epoch cho sampler + có .reset() no-op

        LOGGER.info(f"[Pairing] Enabled with bgpair_map='{args.bgpair_map}', batch={args.batch}")

    # 5b) Peek batch: 2 tấm/panel, mỗi tấm tối đa 4 ảnh => 2x2
    peek = PeekBatch(save_first_n=999, max_images=4, panels=2)  # 999 = dump mọi epoch (đổi tùy ý)
    yolo.add_callback("on_fit_epoch_start", peek.on_fit_epoch_start)

    # --- Train ---
    device_arg = "0" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    yolo.train(
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
        amp=bool(args.amp),
        # ⬇⬇⬇ kiểm soát checkpoint: chỉ lưu best.pt và last.pt
        save=bool(args.save),
        save_period=int(args.save_period),   # -1 = KHÔNG lưu theo epoch
    )
    LOGGER.info("✔ Training finished\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO-STN + SupCon training")

    # training common
    p.add_argument("--yaml", help="Path to model/config YAML used by Ultralytics YOLO.")
    p.add_argument("--model", help="Path to model weights .pt or model YAML.")
    p.add_argument("--output", default="runs_stn", help="Output root directory for runs (logs, ckpts, images).")
    p.add_argument("--epochs", type=int, default=160, help="Total training epochs.")
    p.add_argument("--batch", type=int, default=8, help="Global batch size.")
    p.add_argument("--imgsz", type=int, default=640, help="Training/validation image size (square).")
    p.add_argument("--runs", type=int, default=1, help="Number of repeated runs (seeds).")
    p.add_argument("--resume", action="store_true", help="Resume from the last checkpoint in the run folder.")
    p.add_argument("--device", default="auto",
                   help="Device selector: 'auto', 'cpu', '0', '0,1', ... (CUDA device indices).")
    p.add_argument("--patience", type=int, default=160,
                   help="Early-stopping patience (epochs with no val improvement).")
    p.add_argument("--freeze_epochs", type=int, default=20,
                   help="Freeze/disable STN for first N epochs (bypass/no-grad per your pipeline).")
    p.add_argument("--debug_images", action="store_true",
                   help="Dump STN input/output and GT/PRED overlays via DebugImages callback.")
    p.add_argument("--tsne_every", type=int, default=20,
                   help="Export t-SNE every N epochs (0 to disable).")
    p.add_argument("--tsne_per_class", type=int, default=256,
                   help="Max samples per class for t-SNE (higher -> slower & larger file).")
    p.add_argument("--amp", type=int, default=1,
                   help="Automatic Mixed Precision: 1=enable, 0=disable.")
    # ⬇⬇⬇ THÊM: điều khiển lưu checkpoint
    p.add_argument("--save", type=int, default=1, help="1=save weights, 0=do not save")
    p.add_argument("--save_period", type=int, default=0,
                   help="-1=only best/last; >0=save every N epochs")

    # SupCon (internal, KHÔNG pass trực tiếp vào yolo.train)
    p.add_argument("--supcon_on", type=int, default=1)
    p.add_argument("--supcon_feat", type=str, default="stn")
    p.add_argument("--supcon_warp_gt", type=int, default=0)
    p.add_argument("--supcon_out", type=int, default=7)
    p.add_argument("--supcon_min_box", type=int, default=1)
    p.add_argument("--supcon_max_per_class", type=int, default=0)
    p.add_argument("--supcon_gain", type=float, default=2.5)
    p.add_argument("--supcon_temp", type=float, default=0.5)
    p.add_argument("--supcon_warmup", type=int, default=0)
    p.add_argument("--supcon_log", type=int, default=1)
    p.add_argument("--supcon_use_mem", type=int, default=1)
    p.add_argument("--supcon_queue", type=int, default=4096)
    p.add_argument("--supcon_loss_weight", type=float, default=None)
    p.add_argument("--supcon_schedule", type=str, default="0-")
    p.add_argument("--supcon_proj_dim", type=int, default=128)
    p.add_argument("--supcon_proj_hidden", type=int, default=512)
    p.add_argument("--supcon_proj_bn", type=int, default=2, choices=[0, 1, 2])

    # Pairing flags
    p.add_argument("--pairing", action="store_true",
                   help="Bật paired sampler/negative background loss")
    p.add_argument("--bgpair_map", type=str, default="bgpair_map.json",
                   help="Đường dẫn map abnormal->normal")

    # --- Demo mode (không có CLI args) ---
    if len(sys.argv) == 1:
        LOGGER.info("[INFO] No CLI arguments supplied  using defaults + demo paths")
        args = p.parse_args([])
        args.yaml = DEMO_YAML
        args.model = DEMO_MODEL
        args.debug_images = True
        args.tsne_every = 20
        args.tsne_per_class = 3000
        # bật pairing trong demo
        args.pairing = True
        args.bgpair_map = DEMO_BGPAIR
        # rõ ràng set chế độ lưu chỉ best/last
        args.save = 1
        args.save_period = -1
        return args

    # --- Normal CLI ---
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

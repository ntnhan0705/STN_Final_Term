from __future__ import annotations

"""train_stn.py — Two‑stage fine‑tuning for YOLO models with Spatial Transformer (STN).

* Stage 1: STN **disabled** for the first `--freeze_epochs` epochs.
* Stage 2: STN **enabled** for joint training.

If you run the script **without CLI arguments**, hard‑coded demo paths are
used so you can quickly test the pipeline. Otherwise supply the three
required paths: `--yaml  --model  --weights`.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.block import SpatialTransformer as STN
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.stn_utils import DebugImages

# ---------------------------------------------------------------------------
# Fallback demo values (edit once)
# ---------------------------------------------------------------------------
DEMO_YAML = r"C:/OneDrive/Study/AI/STN_Final_Term/dataset.yaml"
DEMO_MODEL = r"C:\OneDrive\Study\AI\STN_Final_Term\models\yolo11m_stn.pt"
DEMO_WEIGHTS = r"C:/OneDrive/Study/AI/STN_Final_Term/models/weights_stn.pt"
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class STNControl:
    """Enable STN after *freeze_epochs*."""

    def __init__(self, freeze_epochs: int):
        self.freeze_epochs = freeze_epochs

    def on_train_epoch_start(self, trainer):
        enable = trainer.epoch >= self.freeze_epochs
        for m in trainer.model.modules():
            if isinstance(m, STN):
                m.enabled = enable
        LOGGER.info(f"[STN] {'ENABLED' if enable else 'disabled'} at epoch {trainer.epoch}")


def load_weights(model: DetectionModel, ckpt_path: str | Path) -> None:
    """Load tensors from various checkpoint formats (Ultralytics, plain state_dict)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:  # Ultralytics ckpt
        model_obj = ckpt["model"]
        src_sd = model_obj.state_dict() if hasattr(model_obj, "state_dict") else model_obj  # could be OrderedDict
    elif isinstance(ckpt, dict):  # plain state_dict (including OrderedDict)
        src_sd = ckpt
    else:  # nn.Module
        src_sd = ckpt.state_dict()  # type: ignore

    dst_sd = model.state_dict()
    copied = 0
    for k, v in dst_sd.items():
        if k in src_sd and src_sd[k].shape == v.shape:
            dst_sd[k] = src_sd[k]
            copied += 1
    model.load_state_dict(dst_sd)
    LOGGER.info(f"[Weights] copied {copied}, missed {len(dst_sd) - copied}")


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------

def train_one(args: argparse.Namespace, run_id: int) -> None:
    set_seed(run_id)
    LOGGER.info(f"===== RUN {run_id}/{args.runs} =====")

    yolo = YOLO(args.model)
    load_weights(yolo.model, args.weights)

    save_dir = Path(args.output) / f"{Path(args.model).stem}_run{run_id}"

    yolo.add_callback("on_train_epoch_start", STNControl(args.freeze_epochs).on_train_epoch_start)
    yolo.add_callback("on_train_epoch_end", DebugImages(epochs=range(args.epochs + 1), max_images=5).on_train_epoch_end)

    device_arg = "0" if args.device == "auto" and torch.cuda.is_available() else args.device

    yolo.train(
        data=args.yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device_arg,
        pretrained=False,
        resume=args.resume,
        project=str(save_dir.parent),
        name=save_dir.name,
        patience=args.patience,
        val=True,
        verbose=True,
    )
    LOGGER.info("✔ Training finished\n")

# ---------------------------------------------------------------------------
# CLI parser (kept exactly as requested)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI; when no args, use parser defaults + demo paths without duplicating numbers."""
    p = argparse.ArgumentParser(description="YOLO‑STN staged fine‑tuning")
    # REQUIRED paths (no defaults here)
    p.add_argument("--yaml", help="Dataset YAML path")
    p.add_argument("--model", help="YOLO model YAML with STN")
    p.add_argument("--weights", help="Pretrained checkpoint path (.pt)")

    # Hyper‑parameters (single source‑of‑truth defaults)
    p.add_argument("--freeze_epochs", type=int, default=80, help="Epochs to freeze STN")
    p.add_argument("--output", default="runs_stn", help="Output dir for runs/")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--runs", type=int, default=1, help="Number of repeated runs")
    p.add_argument("--resume", action="store_true", help="Resume last run in output dir")
    p.add_argument("--device", default="auto", help="CUDA device id or 'cpu'/'auto'")
    p.add_argument("--patience", type=int, default=50, help="Early‑stopping patience")

    if len(sys.argv) == 1:
        LOGGER.info("[INFO] No CLI arguments supplied – using parser defaults + demo paths")
        args = p.parse_args([])  # use defaults above
        # inject demo paths
        args.yaml = DEMO_YAML
        args.model = DEMO_MODEL
        args.weights = DEMO_WEIGHTS
        return args

    # CLI provided → require the three paths
    p.set_defaults(yaml=None, model=None, weights=None)
    args = p.parse_args()
    missing = [k for k in ("yaml", "model", "weights") if getattr(args, k) is None]
    if missing:
        p.error("The following arguments are required: " + ", ".join("--" + m for m in missing))
    return args

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    for path_key in ("yaml", "model", "weights"):
        pa = Path(getattr(args, path_key))
        if not pa.exists():
            LOGGER.warning(f"[WARN] {pa} not found. Please provide a valid --{path_key}.")

    for i in range(1, args.runs + 1):
        train_one(args, i)


if __name__ == "__main__":
    main()

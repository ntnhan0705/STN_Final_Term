"""weight_mapper.py — Compare two YOLO checkpoints and (optionally) transfer weights.

Run **with arguments** or just **double-click** to test default paths.

Default demo (no CLI args) will use the 3 constants below. Adjust them to your paths.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

# ====== EDIT DEMO PATHS HERE =================================================
DEMO_OLD  = r"C:/OneDrive/Study/AI/STN_Final_Term/models/yolo11m.pt"
DEMO_NEW  = r"C:/OneDrive/Study/AI/STN_Final_Term/models/yolo11m_stn.pt"
DEMO_INSERT_AT = 0
DEMO_OFFSET    = 2
# ============================================================================

# ---------------------------------------------------------------------------
# Key remapper
# ---------------------------------------------------------------------------

def remap_key(key: str, insert_start: Optional[int], offset: int) -> str:
    if insert_start is None or offset == 0:
        return key
    m = re.match(r"model\.(\d+)(\..*)", key)
    if not m:
        return key
    idx, rest = int(m.group(1)), m.group(2)
    if idx >= insert_start:
        idx += offset
    return f"model.{idx}{rest}"

# ---------------------------------------------------------------------------
# Safe loader
# ---------------------------------------------------------------------------

def _load_state_dict(path: str | Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    from torch.serialization import add_safe_globals
    try:
        from ultralytics.nn.tasks import DetectionModel  # type: ignore
        add_safe_globals([DetectionModel])
    except Exception:
        pass
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)

    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
    if isinstance(ckpt, dict) and "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
        return ckpt["model"].state_dict()
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

# ---------------------------------------------------------------------------
# Channel copy helpers
# ---------------------------------------------------------------------------

def _copy_slice(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    slices = tuple(slice(0, min(s, d)) for s, d in zip(src.shape, dst.shape))
    out = dst.clone()
    out[slices] = src[slices]
    return out


def _copy_repeat(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    reps = [math.ceil(d / s) for s, d in zip(src.shape, dst.shape)]
    tiled = src.repeat(tuple(reps))
    cropped = tiled[tuple(slice(0, d) for d in dst.shape)]
    out = dst.clone()
    out.copy_(cropped)
    return out

# ---------------------------------------------------------------------------
# Weight transfer
# ---------------------------------------------------------------------------

def transfer_weights(
    old_sd: Dict[str, torch.Tensor],
    new_sd: Dict[str, torch.Tensor],
    *,
    insert_start: Optional[int],
    offset: int,
    mode: str = "slice",
) -> Tuple[Dict[str, torch.Tensor], int, int]:
    merged = new_sd.copy()
    exact = flexible = 0
    for k_new, v_new in new_sd.items():
        k_old = remap_key(k_new, insert_start, -offset)
        if k_old not in old_sd:
            continue
        v_old = old_sd[k_old]
        if v_old.shape == v_new.shape:
            merged[k_new] = v_old
            exact += 1
        elif v_old.dim() == v_new.dim() and v_old.shape[-2:] == v_new.shape[-2:]:
            merged[k_new] = _copy_slice(v_old, v_new) if mode == "slice" else _copy_repeat(v_old, v_new)
            flexible += 1
    return merged, exact, flexible

# ---------------------------------------------------------------------------
# Comparison & print
# ---------------------------------------------------------------------------

def compare_models(
    old_sd: Dict[str, torch.Tensor],
    new_sd: Dict[str, torch.Tensor],
    *,
    insert_start: Optional[int],
    offset: int,
    max_print: int | None = 20,
) -> None:
    matched, mismatched = 0, []
    for k_old, v_old in old_sd.items():
        k_new = remap_key(k_old, insert_start, offset)
        if k_new not in new_sd:
            continue
        v_new = new_sd[k_new]
        if v_old.shape == v_new.shape:
            matched += 1
        else:
            mismatched.append((k_old, v_old.shape, k_new, v_new.shape))

    print(f"Exact match {matched}/{len(new_sd)} | mismatched {len(mismatched)}")
    if max_print != 0:
        print("-- Mismatched --")
        show = mismatched if max_print is None else mismatched[:max_print]
        for ko, so, kn, sn in show:
            print(f"{ko} {so} -> {kn} {sn}")
        if max_print is not None and len(mismatched) > max_print:
            print("...")

# ---------------------------------------------------------------------------
# CLI + fallback demo
# ---------------------------------------------------------------------------

def _parse_args() -> Optional[argparse.Namespace]:
    if len(sys.argv) == 1:
        return None  # demo mode
    p = argparse.ArgumentParser(description="Compare two YOLO checkpoints and optionally transfer weights.")
    p.add_argument("--old", required=True, help="Path to trained checkpoint (.pt)")
    p.add_argument("--new", required=True, help="Path to new architecture checkpoint (.pt)")
    p.add_argument("--insert_at", type=int, default=None, help="First index that shifted (+offset)")
    p.add_argument("--offset", type=int, default=0, help="Number of blocks inserted (+) or removed (-)")
    p.add_argument("--mode", choices=["slice", "repeat"], default="slice", help="Channel copy strategy")
    p.add_argument("--all", action="store_true", help="Print all mismatches")
    p.add_argument("--out", help="Output .pt file with merged weights")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args is None:
        # ---------- DEMO RUN ----------
        print("[Demo] No CLI args provided → using DEMO_* constants\n")
        args = argparse.Namespace(
            old=DEMO_OLD,
            new=DEMO_NEW,
            insert_at=DEMO_INSERT_AT,
            offset=DEMO_OFFSET,
            mode="repeat",
            all=True,
            out="weights_stn.pt",
        )

    old_sd = _load_state_dict(args.old)
    new_sd = _load_state_dict(args.new)

    compare_models(
        old_sd,
        new_sd,
        insert_start=args.insert_at,
        offset=args.offset,
        max_print=None if args.all else 20,
    )

    merged, ex, flex = transfer_weights(
        old_sd,
        new_sd,
        insert_start=args.insert_at,
        offset=args.offset,
        mode=args.mode,
    )
    print(f"Copied exact {ex}, flexible {flex}")

    if args.out:
        torch.save({"model": merged}, args.out)
        print(f"[✓] Saved {args.out}")


if __name__ == "__main__":
    main()
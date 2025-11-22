# ultralytics/utils/stn_utils.py — clean, grouped, and debug-friendly
from __future__ import annotations
import os, math, random, json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFont, ImageDraw, Image
from torchvision.utils import make_grid

# Ultralytics
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import plot_images
# Debug-only classes tách riêng
try:
    from ultralytics.utils.stn_utils_debug import DebugImages, DebugBgPairROIs
except Exception:
    DebugImages = DebugBgPairROIs = None

# =============================================================================
# 0) Public API
# =============================================================================
__all__ = [
    # Attach entrypoints
    "attach_callbacks",
    "attach_minimal_callbacks",
    "register_pairing",
    # STN controls + theta plumbing
    "STNControl",
    "PublishThetaToStateV2",
    "SeedIdentityThetaOnValStart",
    "ThetaStats",
    # SupCon helpers
    "InjectSupConArgsMinimal",
    "SupConScheduler",
    "ReinforceSupConToLoss",
    "LinkTrainerToLoss",
    "SyncEpochToLoss",
    "TapSTNFeat",
    "SupConPercentLogger",
    "AttachSupConProjToOptim",
    # Safety + hygiene
    "LossNaNGuard",
    "BatchSanityFilter",
    "LRGuard",
    # Debug / Preview
    "DebugImages",
    "DebugBgPairROIs",
    # Validation helpers
    "EnableValLoss",
    "ForceValArgs",
    "apply_val_debug_overrides",
    "register_val_probe",
    "register_results_csv_guard",
    # Misc
    "set_seed",
    "FixFinalEvalModel",
    "SaveLastBestOnly",
    # SupCon feature utils (để loss.py dùng)
    "supcon_to_2d",
    "supcon_safe_cat_0",
    "supcon_safe_cat_meta_0",
]

# =============================================================================
# 1) Constants & small drawing utils
# =============================================================================
CLASS_LABELS = {
    0: "Phình ĐMC", 1: "Xẹp phổi", 2: "Vôi hóa", 3: "Tim to",
    4: "Đông đặc", 5: "ILD", 6: "Thấm nhiễm", 7: "Mờ phổi",
    8: "Nốt/Khối", 9: "Khác", 10: "Tràn dịch",
    11: "Dày MP", 12: "Tràn khí", 13: "Xơ hóa"
}
CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
    (0, 128, 128), (128, 128, 0), (75, 0, 130), (255, 105, 180),
    (0, 191, 255), (34, 139, 34)
]
_FONT_PATH = "arial.ttf"

def _put_text(img: np.ndarray, txt: str, pos: Tuple[int, int], col=(255, 255, 0), size=22) -> np.ndarray:
    pil = Image.fromarray(img); d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(_FONT_PATH, size)
    except Exception:
        font = ImageFont.load_default()
    d.text(pos, txt, font=font, fill=col)
    return np.asarray(pil).copy()

# =============================================================================
# 2) SupCon-safe feature helpers (để ghép an toàn trong loss.py)
# =============================================================================
def compute_supcon_loss(
    feats,                  # Tensor [N,D] hoặc List[Tensor] (gom dim=0)
    labels,                 # Tensor [N] int64
    temperature: float = 0.07,
    queue_feats: torch.Tensor | None = None,   # optional [M,D]
    queue_labels: torch.Tensor | None = None,  # optional [M]
):
    """
    Trả về: scalar tensor (same device/dtype as feats), hoặc 0 nếu không đủ positive.
    - Tự GAP về [*,D] nếu đầu vào là [B,C,H,W] hay [B,C,L].
    - Ghép queue_* vào cuối theo dim=0 nếu có.
    """
    # chuẩn hoá feats -> Tensor [N,D]
    if feats is None:
        return torch.zeros([], device='cpu')
    if isinstance(feats, (list, tuple)):
        chunks = []
        for x in feats:
            if x is None or (not torch.is_tensor(x)) or x.numel() == 0:
                continue
            if x.dim() == 4:
                x = x.mean(dim=(2, 3))
            elif x.dim() == 3:
                x = x.mean(dim=2)
            elif x.dim() == 2:
                pass
            else:
                continue
            chunks.append(x)
        if not chunks:
            return torch.zeros([], device='cpu')
        feats = torch.cat(chunks, dim=0)
    else:
        x = feats
        if x.dim() == 4:   x = x.mean(dim=(2, 3))
        elif x.dim() == 3: x = x.mean(dim=2)
        elif x.dim() == 2: pass
        else:              return torch.zeros([], device=feats.device)
        feats = x

    device = feats.device
    dtype  = feats.dtype

    if labels is None:
        return torch.zeros([], device=device, dtype=dtype)

    labels = labels.to(device=device, dtype=torch.long)

    # ghép queue nếu có
    if (queue_feats is not None and queue_feats.numel() > 0 and
        queue_labels is not None and queue_labels.numel() > 0):
        qf = queue_feats
        if qf.dim() == 4:   qf = qf.mean(dim=(2, 3))
        elif qf.dim() == 3: qf = qf.mean(dim=2)
        elif qf.dim() != 2: qf = None
        if qf is not None and qf.numel() > 0:
            feats  = torch.cat([feats,  qf.to(device=device, dtype=dtype)],  dim=0)
            labels = torch.cat([labels, queue_labels.to(device=device, dtype=torch.long)], dim=0)

    N = feats.shape[0]
    if N < 2 or labels.numel() != N:
        return torch.zeros([], device=device, dtype=dtype)

    # L2-norm
    z = F.normalize(feats, dim=1)

    # similarity
    sim = torch.matmul(z, z.t())               # [N,N]
    self_mask = torch.eye(N, device=device, dtype=torch.bool)
    pos_mask  = (labels.view(-1,1) == labels.view(1,-1)) & (~self_mask)
    if pos_mask.sum() == 0:
        return torch.zeros([], device=device, dtype=dtype)

    sim = sim / max(temperature, 1e-6)
    sim_no_self = sim.masked_fill(self_mask, float('-inf'))
    log_prob = sim_no_self.log_softmax(dim=1)  # [N,N]

    pos_count = pos_mask.sum(dim=1).clamp_min(1)
    valid_i   = (pos_mask.sum(dim=1) > 0)
    if valid_i.sum() == 0:
        return torch.zeros([], device=device, dtype=dtype)

    numerator = (log_prob * pos_mask).sum(dim=1)
    loss_vec  = - numerator[valid_i] / pos_count[valid_i]
    return loss_vec.mean().to(dtype=dtype)

def supcon_to_2d(x: torch.Tensor | None) -> torch.Tensor | None:
    """[B,C,H,W]/[N,C]/[C] -> [N,C] (GAP nếu có H,W)."""
    if x is None:
        return None
    if not torch.is_tensor(x):
        raise TypeError(f"supcon_to_2d expects Tensor or None, got {type(x)}")
    if x.ndim == 4:
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    elif x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(x.shape[0], -1)
    return x

def supcon_safe_cat_0(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
    """Ghép theo dim=0 (xếp chồng mẫu), ép mọi tensor về [N,C], bỏ None/empty."""
    buf: list[torch.Tensor] = []
    for t in tensors:
        t2 = supcon_to_2d(t) if t is not None else None
        if t2 is not None and t2.numel() > 0:
            buf.append(t2)
    if not buf:
        return None
    cset = {t.shape[1] for t in buf}
    if len(cset) != 1:
        shapes = [tuple(t.shape) for t in buf]
        raise RuntimeError(f"[SupCon] Feature dim mismatch: {shapes}")
    return torch.cat(buf, dim=0)

def supcon_safe_cat_meta_0(items: list[torch.Tensor | None]) -> torch.Tensor | None:
    """Ghép metadata (nhãn/img_ids/mask) theo dim=0, bỏ None."""
    mm = [m for m in items if (m is not None and torch.is_tensor(m) and m.numel() > 0)]
    if not mm:
        return None
    return torch.cat(mm, dim=0)

# =============================================================================
# 3) STN wrappers (identity / blend) — single, non-duplicated implementation
# =============================================================================
class _Ctx:
    @staticmethod
    def root(obj):
        m = getattr(obj, "model", None) or getattr(getattr(obj, "trainer", None), "model", None)
        return getattr(m, "model", m) if m is not None else None
    @staticmethod
    def is_stn(m) -> bool:
        return m.__class__.__name__ in {"SpatialTransformer", "STN", "SpatialTransformer2D", "SpatialTransformerBlock"}
    @staticmethod
    def stn_modules(model):
        if model is None:
            return []
        for m in model.modules():
            if _Ctx.is_stn(m):
                yield m
    @staticmethod
    def state(owner):
        if not hasattr(owner, "state"):
            owner.state = {}
        return owner.state

def _stn_identity_forward(self, x, *a, **k):
    """Trả ảnh gốc; publish theta/out nếu có hook."""
    try:
        b = x.shape[0] if hasattr(x, "shape") else 1
        theta_I = x.new_tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0).repeat(b, 1, 1)
        if hasattr(self, "record_theta"):
            try:
                self.record_theta(theta_I)
            except Exception:
                pass
        if hasattr(self, "record_out"):
            try:
                self.record_out(x)
            except Exception:
                pass
    except Exception:
        pass
    return x

def _stn_blend_forward(self, x, *a, **k):
    """Blend (original → stabilized) với clamp θ an toàn + publish out."""
    orig = getattr(self, "_stn_forward_orig", None)
    if not callable(orig):
        return x

    out = orig(x, *a, **k)  # chạy forward gốc

    # unpack về (x_t, theta)
    if isinstance(out, tuple):
        x_t, theta = out[0], out[1]
    else:
        x_t, theta = out, getattr(self, "theta", None)

    if isinstance(x_t, (list, tuple)):  # lấy phần tử đầu nếu list/tuple
        x_t = x_t[0]
    if not torch.is_tensor(x_t):
        x_t = x

    try:
        if torch.is_tensor(theta) and hasattr(x, "shape"):
            B, C, H, W = x.shape
            th = theta
            if th.dim() == 2:
                th = th.unsqueeze(0)
            if th.size(0) == 1 and B > 1:
                th = th.expand(B, -1, -1).contiguous()
            elif th.size(0) != B:
                th = th[:B] if th.size(0) > B else torch.cat([th, th.new_zeros(B - th.size(0), 2, 3)], 0)

            tmax = float(getattr(self, "_stn_tmax", 0.20))
            smin = float(getattr(self, "_stn_smin", 0.90))
            smax = float(getattr(self, "_stn_smax", 1.10))

            t = th[..., :, 2]      # (B,2)
            M = th[..., :, :2]     # (B,2,2)

            t = torch.clamp(t, min=-tmax, max=tmax)
            det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
            scale = torch.clamp(det.abs().sqrt(), min=smin, max=smax).view(-1, 1, 1)
            M = M / scale

            if bool(getattr(self, "_stn_stabilize_rot", False)):
                I = torch.eye(2, device=x.device, dtype=x.dtype).unsqueeze(0)
                M = 0.75 * (M - I) + I

            theta_safe = torch.cat([M, t.unsqueeze(-1)], -1)
            grid = F.affine_grid(theta_safe, size=(B, C, H, W), align_corners=False)
            x_stab = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

            try:
                self.theta = theta_safe.detach()
            except Exception:
                pass

            x_t = x_stab
    except Exception:
        pass

    a_ = float(getattr(self, "_stn_blend_alpha", 1.0))
    out_f = x_t if a_ >= 1.0 else (x if a_ <= 0.0 else (x + a_ * (x_t - x)))

    if hasattr(self, "record_out"):
        try:
            self.record_out(out_f)
        except Exception:
            pass

    if not hasattr(self, "_stn_blend_warned") and isinstance(out, (list, tuple)):
        try:
            LOGGER.warning("[STN/Blend] forward() trả về %s; đã chuẩn hoá để blend an toàn.", type(out).__name__)
        except Exception:
            pass
        self._stn_blend_warned = True

    self._stn_mode = "blend"
    return out_f

class STNControl(_Ctx):
    """
    Freeze STN sớm (identity), rồi warmup blend theo alpha; tuỳ chọn ép identity ở vòng VAL.
    """
    def __init__(self, freeze_epochs=0, stn_warmup=20, tmax=0.20, smin=0.90, smax=1.10,
                 val_identity=True, log=False):
        self.freeze_epochs = max(0, int(freeze_epochs))
        self.stn_warmup = max(0, int(stn_warmup))
        self.tmax, self.smin, self.smax = float(tmax), float(smin), float(smax)
        self.val_identity = bool(val_identity)
        self.log, self._mode, self._alpha, self._epoch = bool(log), "identity", 0.0, -1

    @staticmethod
    def _ensure_orig_forward(m):
        if not hasattr(m, "_stn_forward_orig"):
            m._stn_forward_orig = m.forward

    def _patch_identity(self, m):
        import types
        self._ensure_orig_forward(m)
        m.forward = types.MethodType(_stn_identity_forward, m)
        m._stn_mode = "identity"
        for k in ("_stn_blend_alpha", "_stn_tmax", "_stn_smin", "_stn_smax"):
            if hasattr(m, k):
                try:
                    delattr(m, k)
                except Exception:
                    pass

    def _patch_blend(self, m, alpha: float):
        import types
        self._ensure_orig_forward(m)
        m._stn_blend_alpha = float(max(0.0, min(1.0, alpha)))
        m._stn_tmax = float(self.tmax)
        m._stn_smin = float(self.smin)
        m._stn_smax = float(self.smax)
        m.forward = types.MethodType(_stn_blend_forward, m)
        m._stn_mode = "blend"

    def _alpha_for(self, e: int) -> float:
        if e < self.freeze_epochs:
            return 0.0
        if self.stn_warmup <= 0:
            return 1.0
        k = (e - self.freeze_epochs) / float(self.stn_warmup)
        return float(max(0.0, min(1.0, k)))

    def _apply_identity(self, model, on: bool):
        for m in self.stn_modules(model):
            if on:
                self._patch_identity(m)
            elif getattr(m, "_stn_forward_orig", None):
                m.forward = m._stn_forward_orig

    def _apply_blend(self, model, alpha: float):
        for m in self.stn_modules(model):
            self._patch_blend(m, alpha)

    def on_train_epoch_start(self, t):
        model = self.root(t)
        if model is None:
            return
        e = int(getattr(t, "epoch", 0))
        self._epoch = e
        if e < self.freeze_epochs:
            self._apply_identity(model, True)
            self._mode, self._alpha = "identity", 0.0
            if self.log:
                LOGGER.info(f"[STN] identity @ epoch {e}")
        else:
            a = self._alpha_for(e)
            self._apply_identity(model, False)
            self._apply_blend(model, a)
            self._mode, self._alpha = "blend", a
            if self.log:
                LOGGER.info(f"[STN] blend α={a:.3f} (t≤{self.tmax:.2f}, s∈[{self.smin:.2f},{self.smax:.2f}])")

    def on_val_start(self, v):
        if not self.val_identity:
            if self.log:
                LOGGER.info("[STN] validation: keep current mode (no identity override)")
            return
        m = self.root(v)
        if m is not None:
            self._apply_identity(m, True)
        if self.log:
            LOGGER.info("[STN] validation: identity")

    def on_val_end(self, v):
        if not self.val_identity:
            return
        m = self.root(v)
        if m is None:
            return
        if self._mode == "blend":
            self._apply_identity(m, False)
            self._apply_blend(m, self._alpha)
            if self.log:
                LOGGER.info(f"[STN] restore blend α={self._alpha:.3f}")

# =============================================================================
# 4) Theta plumbing (publish/seed/stats)
# =============================================================================
class PublishThetaToStateV2(_Ctx):
    """Attach vào STN để thu theta/out và lưu vào owner.state['stn_theta'/'stn_out']."""
    def __init__(self, verbose=True):
        self.verbose = verbose
        self._prev = {"train": {}, "val": {}}

    def _expected_B(self, owner):
        try:
            tr = owner if getattr(owner, "batch", None) is not None else getattr(owner, "trainer", None)
            if tr and isinstance(getattr(tr, "batch", None), dict):
                imgs = tr.batch.get("img", None)
                if torch.is_tensor(imgs):
                    return int(imgs.shape[0])
        except Exception:
            pass
        return None

    def _remember_B_and_clear(self, owner):
        S = self.state(owner)
        B = self._expected_B(owner)
        if B:
            S["_B"] = B
        S["stn_theta"] = None  # clear trước batch

    def _attach(self, owner, slot: str):
        root = getattr(owner, "model", None) or owner
        S = _Ctx.state(owner)
        attached, prev_map = 0, {}

        def _norm_theta(th):
            if torch.is_tensor(th) and th.dim() == 2:
                th = th.unsqueeze(0)
            return th

        for m in self.stn_modules(root):
            if getattr(m, "_theta_pub_attached", False):
                continue

            prv_theta = getattr(m, "record_theta", None)
            prv_out = getattr(m, "record_out", None)
            prev_map[m] = (prv_theta, prv_out)

            def rec_theta(theta, _m=m, _prev=prv_theta):
                try:
                    _prev and _prev(theta)
                except Exception:
                    pass
                th = _norm_theta(theta)
                if th is None:
                    return
                try:
                    dev = next(_m.parameters()).device
                    th = th.to(device=dev, non_blocking=True)
                except Exception:
                    pass
                S["stn_theta"] = th.detach() if hasattr(th, "detach") else th
                try:
                    _m.theta = S["stn_theta"]
                except Exception:
                    pass

            def rec_out(out_img, _prev=prv_out):
                try:
                    _prev and _prev(out_img)
                except Exception:
                    pass
                if torch.is_tensor(out_img):
                    S["stn_out"] = out_img.detach()
                else:
                    S["stn_out"] = out_img

            setattr(m, "record_theta", rec_theta)
            setattr(m, "record_out", rec_out)
            setattr(m, "_theta_pub_attached", True)
            attached += 1

        self._prev[slot] = prev_map
        if self.verbose:
            LOGGER.info(f"[ThetaPub] attach[{slot}] -> {attached}")
        return attached

    def _detach(self, slot: str):
        prev_map = self._prev.get(slot, {})
        r = 0
        for m, (prv_theta, prv_out) in list(prev_map.items()):
            try:
                if prv_theta is not None:
                    setattr(m, "record_theta", prv_theta)
                elif hasattr(m, "record_theta"):
                    delattr(m, "record_theta")
            except Exception:
                pass
            try:
                if prv_out is not None:
                    setattr(m, "record_out", prv_out)
                elif hasattr(m, "record_out"):
                    delattr(m, "record_out")
            except Exception:
                pass
            r += 1
        prev_map.clear()
        self._prev[slot] = {}
        if self.verbose:
            LOGGER.info(f"[ThetaPub] detach[{slot}] -> {r}")

    # hooks
    def on_train_start(self, t): self._attach(t, "train")
    def on_train_end(self, t):   self._detach("train")
    def on_val_start(self, v):   self._attach(v, "val")
    def on_val_end(self, v):     self._detach("val")
    def on_train_batch_start(self, t, *a, **k): self._remember_B_and_clear(t)
    def on_val_batch_start(self, v, *a, **k):   self._remember_B_and_clear(v)

class SeedIdentityThetaOnValStart(_Ctx):
    def __init__(self, B_hint=8): self.B_hint = int(B_hint)
    def on_val_start(self, v):
        S = self.state(getattr(v, "trainer", v))
        B = self.B_hint
        th = torch.zeros((B, 2, 3), dtype=torch.float32); th[:, 0, 0] = 1.0; th[:, 1, 1] = 1.0
        S["stn_theta"] = th

class ThetaStats(_Ctx):
    def __init__(self, every=1, tag="train"): self.every, self.tag, self._i = max(1, int(every)), tag, 0
    def on_train_batch_end(self, t):
        if self.tag != "train": return
        self._tick(t)
    def on_val_batch_end(self, v):
        if self.tag != "val": return
        self._tick(getattr(v, "trainer", v))
    def _tick(self, owner):
        self._i += 1
        if (self._i % self.every) != 0: return
        S = self.state(owner); th = S.get("stn_theta", None)
        if isinstance(th, torch.Tensor):
            tmin, tmax = float(th.min()), float(th.max())
            LOGGER.info(f"[θ/{self.tag}] shape={tuple(th.shape)} min={tmin:.4f} max={tmax:.4f}")

# =============================================================================
# 5) SupCon: inject/schedule/reinforce/link/tap/percent/proj
# =============================================================================
_SUPCON_PROJ_GLOBAL = None
def supcon_register_projector(proj):
    global _SUPCON_PROJ_GLOBAL
    _SUPCON_PROJ_GLOBAL = proj
    LOGGER.info("[SupConProj] registered")
def _supcon_get_global_projector(): return _SUPCON_PROJ_GLOBAL

class InjectSupConArgsMinimal:
    def __init__(self, **cfg): self.cfg = dict(cfg)
    def _ensure_ns(self, obj, attr):
        val = getattr(obj, attr, None)
        if val is None or isinstance(val, dict):
            val = SimpleNamespace(**(val or {})); setattr(obj, attr, val)
        return val
    def on_pretrain_routine_start(self, t): t._supcon_cfg = SimpleNamespace(**self.cfg)
    def on_pretrain_routine_end(self, t):
        ma = self._ensure_ns(t.model, "args")
        [setattr(ma, k, v) for k, v in self.cfg.items()]
        crit = getattr(t.model, "criterion", None)
        if crit is not None:
            hyp = self._ensure_ns(crit, "hyp"); [setattr(hyp, k, v) for k, v in self.cfg.items()]
        t._supcon_cfg = SimpleNamespace(**self.cfg)
    def on_train_start(self, t):
        cache = getattr(t, "_supcon_cfg", None)
        cfg = vars(cache) if isinstance(cache, SimpleNamespace) else self.cfg
        ma = self._ensure_ns(t.model, "args"); [setattr(ma, k, v) for k, v in cfg.items()]
        crit = getattr(t.model, "criterion", None)
        if crit is not None:
            hyp = self._ensure_ns(crit, "hyp"); [setattr(hyp, k, v) for k, v in cfg.items()]

class SupConScheduler:
    def __init__(self, schedule_str: str, default_on: int = 0):
        self.default_on = 1 if default_on else 0
        self.ranges = self._parse(schedule_str or "")
    @staticmethod
    def _parse(spec: str):
        out = []
        for tok in spec.replace(" ", "").split(","):
            if not tok: continue
            if "-" in tok:
                a, b = tok.split("-", 1); a = a.strip(); b = b.strip()
                lo = 1 if a == "" else int(a); hi = 10**9 if b == "" else int(b)
            else:
                lo = hi = int(tok)
            out.append((lo, hi))
        return out
    def _on(self, e_disp: int) -> int:
        if not self.ranges: return self.default_on
        for lo, hi in self.ranges:
            if lo <= e_disp <= hi: return 1
        return 0
    def on_train_epoch_start(self, t):
        e = int(getattr(t, "epoch", 0)) + 1
        want = self._on(e)
        ma = getattr(t.model, "args", None)
        if ma is None or isinstance(ma, dict):
            ma = SimpleNamespace(**(ma or {})); t.model.args = ma
        setattr(ma, "supcon_on", int(want))
        if want == 1 and hasattr(t, "loss_names"):
            try:
                if not t.loss_names or t.loss_names[-1] != "supcon_loss":
                    t.loss_names = ("box_loss", "cls_loss", "dfl_loss", "supcon_loss")
            except Exception:
                pass
        LOGGER.info(f"[SupCon/schedule] e={e} -> on={want}")

# --- PATCH: ReinforceSupConToLoss (thay thế class/hook cũ) ---
class ReinforceSupConToLoss:
    """
    Patches trainer.criterion.__call__ để thêm SupCon khi TRAIN.
    Cứng cáp hơn: nếu chưa có criterion, sẽ retry mỗi epoch cho tới khi thành công.
    """
    def __init__(self):
        self._done = False
        self._warned = False

    def _try_patch(self, trainer):
        if self._done:
            return
        crit = getattr(trainer, "criterion", None)
        if not callable(crit):
            if not self._warned:
                LOGGER.warning("[ReinforceSupConToLoss] no criterion to patch (will retry next epoch).")
                self._warned = True
            return
        # tránh double-patch
        if getattr(crit, "_supcon_patched", False):
            self._done = True
            LOGGER.info("[ReinforceSupConToLoss] already patched (skip).")
            return

        orig_call = crit.__call__

        def wrapped_call(*args, **kwargs):
            # args: (feats, batch) theo loss v8
            out = orig_call(*args, **kwargs)
            # Không cộng SupCon khi supcon_on==0 hoặc không có ctx
            try:
                trainer_ctx = getattr(trainer, "_supcon_ctx", None)
                if not trainer_ctx or not getattr(trainer_ctx, "on", 0):
                    return out
                # out = (total, items) hoặc dict/Namespace -> normalize
                total, items = out if isinstance(out, (list, tuple)) and len(out) == 2 else (None, out)
                sup = getattr(trainer, "_supcon_term", 0.0)
                if total is not None and isinstance(total, torch.Tensor):
                    total = total + (sup if torch.is_tensor(sup) else torch.tensor(float(sup), device=total.device))
                    # chèn supcon_loss vào items nếu có dict
                    if isinstance(items, dict):
                        items["supcon_loss"] = sup if torch.is_tensor(sup) else float(sup)
                    return total, items
                return out
            except Exception as e:
                LOGGER.warning(f"[ReinforceSupConToLoss] runtime skip (keep original loss): {e}")
                return out

        try:
            crit.__call__ = wrapped_call
            setattr(crit, "_supcon_patched", True)
            self._done = True
            LOGGER.info(f"[ReinforceSupConToLoss] patched -> crit={crit.__class__.__name__}")
        except Exception as e:
            LOGGER.warning(f"[ReinforceSupConToLoss] patch failed (will retry): {e}")

    # gắn vào các hook “sớm nhưng nhiều lần”
    def on_train_start(self, trainer):        self._try_patch(trainer)
    def on_train_epoch_start(self, trainer):  self._try_patch(trainer)
    def on_fit_epoch_start(self, trainer):    self._try_patch(trainer)


class LinkTrainerToLoss:
    def _link(self, t):
        """Attach trainer to loss and sync epoch if available."""
        # Prefer loss/criterion on model, fallback to trainer fields
        loss_obj = (
            getattr(getattr(t, "model", None), "loss", None)
            or getattr(getattr(t, "model", None), "criterion", None)
            or getattr(t, "loss", None)
            or getattr(t, "criterion", None)
        )
        if loss_obj is None:
            return
        try:
            loss_obj._trainer = t
            if hasattr(t, "epoch"):
                loss_obj.epoch = int(t.epoch)
        except Exception:
            pass

    def on_train_start(self, t):        self._link(t)
    def on_train_epoch_start(self, t):  self._link(t)
    def on_train_batch_start(self, t):  self._link(t)

class SyncEpochToLoss:
    def _sync(self, t):
        loss_obj = (
            getattr(getattr(t, "model", None), "loss", None)
            or getattr(getattr(t, "model", None), "criterion", None)
            or getattr(t, "loss", None)
            or getattr(t, "criterion", None)
        )
        if loss_obj is not None and hasattr(t, "epoch"):
            try:
                loss_obj.epoch = int(t.epoch)
            except Exception:
                pass

    def on_train_epoch_start(self, t):  self._sync(t)
    def on_train_batch_start(self, t):  self._sync(t)

class TapSTNFeat:
    def __init__(self, out_idx: int | None = None, out_name: str | None = None):
        self.out_idx  = out_idx
        self.out_name = (out_name or None)
        self.h        = None
        self.where    = None
        self.trainer  = None  # <- giữ reference trainer

    def _cfg(self, t, key, default=None):
        cfg = getattr(t, "_supcon_cfg", None)
        if cfg is None: return default
        return (cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default))

    def _hook(self, module, inputs, output):
        # GAP về [B,C] và đẩy vào trainer._supcon_buf['feats']
        if not torch.is_tensor(output) or output.numel() == 0 or self.trainer is None:
            return
        x = output
        if x.dim() == 4:   x = x.mean(dim=(2,3))
        elif x.dim() == 3: x = x.mean(dim=2)
        elif x.dim() == 2: pass
        else:              return
        buf = getattr(self.trainer, "_supcon_buf", None)
        if not isinstance(buf, dict):
            buf = {'feats': []}
        buf.setdefault('feats', []).append(x)  # giữ gradient
        setattr(self.trainer, "_supcon_buf", buf)

    def _resolve(self, model, idx, name):
        modlist = getattr(model, "model", None)
        if modlist is None: return None, None
        if isinstance(idx, int) and 0 <= idx < len(modlist):
            return modlist[idx], f"model.model[{idx}]"
        if name:
            low = str(name).lower()
            for i, m in enumerate(modlist):
                if low in m.__class__.__name__.lower():
                    return m, f"model.model[{i}]<{m.__class__.__name__}>"
        # fallback: layer đầu có forward
        for i, m in enumerate(modlist):
            if hasattr(m, "forward"):
                return m, f"model.model[{i}]<{m.__class__.__name__}>"
        return None, None

    def on_train_start(self, t, *_, **__):
        self.trainer = t  # <- giữ trainer để ghi buffer
        idx  = self._cfg(t, "supcon_out", self.out_idx)
        name = self._cfg(t, "supcon_name", self.out_name)
        m, where = self._resolve(t.model, idx, (name.strip() or None) if isinstance(name, str) else None)
        if m is None:
            LOGGER.info("[TapSTNFeat] no layer")
            return
        if self.h is not None:
            try: self.h.remove()
            except Exception: pass
        self.h     = m.register_forward_hook(self._hook)
        self.where = where
        LOGGER.info(f"[TapSTNFeat] attach at {where}")

    def on_train_end(self, *a, **k):
        if self.h is not None:
            try: self.h.remove()
            except Exception: pass
        self.h, self.where, self.trainer = None, None, None

    def on_train_batch_end(self, *a, **k):
        # dọn buffer sau batch để không lẫn sang batch sau
        tr = self.trainer
        if tr is not None and isinstance(getattr(tr, "_supcon_buf", None), dict):
            tr._supcon_buf['feats'] = []

class SupConPercentLogger:
    def __init__(self): self.num = 0.0; self.den = 0.0
    def _read(self, t):
        n, v = getattr(t, "loss_names", None), getattr(t, "tloss", None)
        if n and v is not None:
            vals = v.detach().cpu().tolist() if torch.is_tensor(v) else list(v)
            m = {k: float(x) for k, x in zip(n, vals)}; return m.get("supcon_loss", 0.0), sum(m.values())
        return None, None
    def on_train_batch_end(self, t):
        s, d = self._read(t)
        if s is not None: self.num += s; self.den += max(d, 1e-9)
    def on_train_epoch_end(self, t):
        if self.den > 0: LOGGER.info(f"[SupCon%] epoch {t.epoch + 1}:  {100 * self.num / self.den:.1f}%")
        self.num = self.den = 0.0

class AttachSupConProjToOptim:
    def __init__(self, in_dim: int, out_dim: int = 128, hidden: int = 512, bn: int = 2, lr: float = 1e-3):
        self.in_dim, self.out_dim, self.hidden, self.bn, self.lr = int(in_dim), int(out_dim), int(hidden), int(bn), float(lr)
        self.attached = False
    def _resolve(self, t):
        cand = []
        crit = getattr(t, "criterion", None); loss = getattr(t, "loss", None)
        if crit is not None: cand += [getattr(crit, k, None) for k in ("_supcon_proj", "supcon_projector", "projector", "supcon_proj")]
        if loss is not None: cand += [getattr(loss, k, None) for k in ("_supcon_proj", "supcon_projector", "projector", "supcon_proj")]
        cand += [getattr(t, k, None) for k in ("_supcon_proj", "supcon_projector", "supcon_proj")]
        model = getattr(t, "model", None)
        if model is not None: cand += [getattr(model, k, None) for k in ("_supcon_proj", "supcon_projector", "supcon_proj")]
        cand.append(_supcon_get_global_projector())
        for c in cand:
            if c is not None: return c
        return None
    def _attach(self, t, proj, when: str):
        if proj is None: return False
        opt = getattr(t, "optimizer", None)
        if opt is None: LOGGER.info(f"[SupConProj] no optimizer ({when})"); return False
        params = [p for p in proj.parameters() if p.requires_grad]
        if not params: LOGGER.info("[SupConProj] no trainable params"); return False
        proj_ids = {id(p) for p in params}
        target_pg = None
        for g in opt.param_groups:
            if proj_ids & {id(p) for p in g.get("params", [])}:
                target_pg = g; break
        if target_pg is None:
            opt.add_param_group({"params": list(params), "lr": float(self.lr), "weight_decay": 0.0, "name": "supcon_proj"})
            LOGGER.info(f"[SupConProj] ADDED (when={when}) | n_params={sum(p.numel() for p in params)} | lr={self.lr}")
        else:
            existed = {id(p) for p in target_pg["params"]}
            target_pg["params"].extend([p for p in params if id(p) not in existed])
            if float(target_pg.get("lr", 0.0)) == 0.0: target_pg["lr"] = float(self.lr)
            LOGGER.info(f"[SupConProj] REUSED (when={when}) | lr={target_pg['lr']}")
        if getattr(t, 'amp', False):
            try:
                from torch.cuda.amp.grad_scaler import GradScaler
                t.scaler = GradScaler(enabled=True)
                LOGGER.warning("[SupCon/Optim] Optimizer changed. Re-initializing GradScaler...")
            except Exception: pass
        try: supcon_register_projector(proj)
        except Exception: pass
        return True
    def _try(self, t, when):
        if self.attached: return
        proj = self._resolve(t)
        if proj is None: return
        if self._attach(t, proj, when):
            self.attached = True
    def on_train_start(self, t, *a, **k):       self._try(t, "on_train_start")
    def on_train_batch_start(self, t, *a, **k): self._try(t, "on_train_batch_start")
    def on_train_batch_end(self, t, *a, **k):   self._try(t, "on_train_batch_end")

# =============================================================================
# 6) Safety: NaN guard + batch sanitation + LRGuard
# =============================================================================
class LossNaNGuard:
    def __init__(self, stop_on_nan=True, save_bad_batch=True): self.stop_on_nan, self.save_bad_batch = stop_on_nan, save_bad_batch
    def on_train_batch_end(self, t, *_, **__):
        items = getattr(t, "loss_items", None)
        if items is None: return
        def f(v):
            if isinstance(v, float): return v
            if torch.is_tensor(v): return float(v.detach().item())
            try: return float(v)
            except Exception: return float("nan")
        names = ["box_loss", "cls_loss", "dfl_loss", "supcon_loss"]
        vals = {k: f(v) for k, v in (items.items() if isinstance(items, dict) else zip(names, items))}
        bad = {k: v for k, v in vals.items() if (math.isnan(v) or math.isinf(v))}
        if not bad: return
        e = getattr(t, "epoch", -1); step = getattr(t, "batch_i", getattr(t, "ni", -1))
        LOGGER.error(f"[NaNGuard] epoch={e} step={step} NaN/Inf in {list(bad.keys())}")
        batch = getattr(t, "batch", None)
        if self.save_bad_batch and isinstance(batch, dict) and ("img" in batch):
            try:
                save_dir = Path(getattr(t, "save_dir", Path(".")))
                fname = save_dir / f"nan_batch_e{e:03d}_i{int(step):06d}.jpg"; plot_images(images=batch["img"], batch=batch, fname=fname)
                LOGGER.error(f"[NaNGuard] saved bad batch -> {fname}")
            except Exception as ex:
                LOGGER.error(f"[NaNGuard] save-batch failed: {ex}")
        # context
        try:
            imgs = (t.batch.get("img") if isinstance(t.batch, dict) else None)
            if torch.is_tensor(imgs):
                img_min, img_max = float(imgs.min().item()), float(imgs.max().item())
            else:
                img_min, img_max = None, None
            lr = None
            try:
                for g in t.optimizer.param_groups: lr = g.get("lr", None); break
            except Exception: pass
            li = getattr(t, "loss_items", None)
            li = [float(v) for v in li] if li is not None else None
            LOGGER.error(f"[NaNGuard/ctx] img_minmax=({img_min},{img_max}) lr={lr} loss_items={li}")
        except Exception:
            pass
        if self.stop_on_nan: raise RuntimeError("[NaNGuard] Stop due to NaN")

class BatchSanityFilter:
    def __init__(self, eps: float = 1e-6): self.eps = float(eps)
    def on_train_batch_start(self, t, *args, **kwargs):
        batch = getattr(t, "batch", None)
        if not isinstance(batch, dict): return
        for k in ("img", "bboxes", "cls"):
            v = batch.get(k, None)
            if torch.is_tensor(v): batch[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.is_tensor(batch.get("bboxes", None)):
            bb = batch["bboxes"]
            if torch.all(bb.abs() <= 2.0):  # YOLO-normed
                batch["bboxes"] = torch.clamp(bb, 0.0 - self.eps, 1.0 + self.eps)

class LRGuard:
    def on_train_start(self, t):
        args = getattr(t, "args", getattr(t.model, "args", None))
        lr0 = float(getattr(args, "lr0", 1e-3)); lrf = float(getattr(args, "lrf", lr0))
        if lr0 <= 0: LOGGER.warning("[LRGuard] lr0<=0?")
        if lr0 > 1e-2: LOGGER.warning(f"[LRGuard] lr0={lr0} có thể hơi cao (>=1e-2)")

# =============================================================================
# 7) Debug images (STN) + BG-pair preview
# =============================================================================
def _to_uint8_grid(t: torch.Tensor, nrow: int = 4) -> Optional["torch.Tensor"]:
    try:
        if not torch.is_tensor(t) or t.dim() != 4:
            return None
        t = t.detach().float()
        if t.size(1) == 1:
            t = t.repeat(1, 3, 1, 1)
        g = make_grid(t, nrow=nrow, normalize=True, scale_each=True).clamp(0, 1)
        g = (g.cpu() * 255.0).byte()
        return g
    except Exception:
        return None

class _DebugImagesLegacy(_Ctx):
    """Render ORIGINAL vs STN (+θ) ở các epoch chỉ định."""
    def __init__(self, epochs=(0, 5, 10, 15, 20), max_images=5, subdir: str = "stn_dbg"):
        self.epochs = set(int(e) for e in epochs)
        self.max_images = int(max_images)
        self.subdir = str(subdir)
        self.samples = []
        self.dbg_dir = None

    @staticmethod
    def _grab_loader(t):
        vdl = getattr(getattr(t, "validator", None), "dataloader", None)
        return vdl or getattr(t, "train_loader", None)

    def _cache_samples(self, t):
        ld = self._grab_loader(t)
        if ld is None:
            LOGGER.warning("[DebugImages] no dataloader available")
            return
        try:
            for batch in ld:
                imgs = batch["img"] if isinstance(batch, dict) else batch[0]
                tgts = batch if isinstance(batch, dict) else batch[1]
                # tên file
                imf = None
                if isinstance(tgts, dict):
                    if "im_file" in tgts and isinstance(tgts["im_file"], (list, tuple)):
                        imf = tgts["im_file"][0] if tgts["im_file"] else None
                    elif "im_files" in tgts and isinstance(tgts["im_files"], (list, tuple)):
                        imf = tgts["im_files"][0] if tgts["im_files"] else None

                # batch_idx==0
                if "batch_idx" in tgts:
                    idx0 = (tgts["batch_idx"] == 0)
                else:
                    idx0 = torch.ones((tgts["cls"].shape[0],), dtype=torch.bool)

                self.samples.append((
                    imgs[0].detach().cpu(),
                    tgts["bboxes"][idx0].detach().cpu(),
                    tgts["cls"][idx0].detach().cpu(),
                    imf
                ))
                if len(self.samples) >= self.max_images:
                    break
        except Exception as ex:
            LOGGER.warning(f"[DebugImages] cache-samples failed: {ex}")

    @staticmethod
    def _scale_xyxy(x: np.ndarray, rx: float, ry: float) -> np.ndarray:
        x = x.astype(np.float32).copy()
        x[:, [0, 2]] *= float(rx)
        x[:, [1, 3]] *= float(ry)
        return x

    @staticmethod
    def _warp_boxes_xywh_with_theta(bxywh: torch.Tensor, W: int, H: int, th: torch.Tensor | None):
        if th is None:
            x, y, w, h = bxywh.unbind(-1)
            return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)

        device = bxywh.device
        x, y, w, h = bxywh.unbind(-1)
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        Xs = torch.stack([x1, x2, x2, x1], 1)
        Ys = torch.stack([y1, y1, y2, y2], 1)

        pix2norm = lambda p, L: (p / max(L - 1, 1) * 2.0) - 1.0
        Xn, Yn = pix2norm(Xs, W), pix2norm(Ys, H)

        th = th.detach().to(torch.float32, device=device).view(2, 3)
        A = torch.tensor([[th[0, 0], th[0, 1], th[0, 2]],
                          [th[1, 0], th[1, 1], th[1, 2]],
                          [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        Ainv = torch.linalg.inv(A)

        ones = torch.ones_like(Xn)
        P_in = torch.stack([Xn, Yn, ones], 1)
        P_out = torch.einsum("ij,njk->nik", Ainv, P_in)
        Xo, Yo = P_out[:, 0, :], P_out[:, 1, :]

        norm2pix = lambda p, L: (p + 1.0) * 0.5 * (L - 1)
        Xp, Yp = norm2pix(Xo, W), norm2pix(Yo, H)

        x1p, x2p = Xp.min(1).values, Xp.max(1).values
        y1p, y2p = Yp.min(1).values, Yp.max(1).values
        x1p = x1p.clamp(0, W - 1); y1p = y1p.clamp(0, H - 1)
        x2p = x2p.clamp(1, W);     y2p = y2p.clamp(1, H)
        return torch.stack([x1p, y1p, x2p, y2p], 1)

    def on_train_epoch_end(self, t):
        ep = int(getattr(t, "epoch", 0))
        if ep not in self.epochs:
            return

        if self.dbg_dir is None:
            self.dbg_dir = os.path.join(str(t.save_dir), self.subdir)
            os.makedirs(self.dbg_dir, exist_ok=True)
        if not self.samples:
            self._cache_samples(t)
            if not self.samples:
                LOGGER.warning("[DebugImages] no sample cached")
                return

        model = t.model
        device = next(model.parameters()).device
        was_train = model.training
        model.eval()

        # giữ cùng mode STN như khi train (hoặc identity nếu trước đó identity)
        try:
            for cb in t.callbacks.get("on_train_epoch_start", []):
                if cb.__class__.__name__ == "STNControl":
                    if getattr(cb, "_mode", "identity") == "blend":
                        cb._apply_identity(model, False); cb._apply_blend(model, float(getattr(cb, "_alpha", 1.0)))
                        LOGGER.info(f"[DebugImages] STN blend α={getattr(cb,'_alpha',1.0):.3f} for viz.")
                    else:
                        cb._apply_identity(model, True); LOGGER.info("[DebugImages] STN identity for viz.")
                    break
        except Exception as e:
            LOGGER.warning(f"[DebugImages] STN patch failed: {e}")

        H_panel = 672
        try:
            for i, (img0, bxywh, bcls, path0) in enumerate(self.samples):
                x = img0.clone().to(device).unsqueeze(0).float()
                if x.max() > 1.0:
                    x = x / 255.0
                if hasattr(t, "state"):
                    t.state.pop("stn_theta", None)
                    t.state.pop("stn_out", None)

                _ = model(x)
                theta = getattr(t, "state", {}).get("stn_theta", None)
                stn_img = getattr(t, "state", {}).get("stn_out", None)

                ori = img0.detach().cpu().numpy()
                if ori.ndim == 3 and ori.shape[0] in (1, 3):
                    ori = np.transpose(ori, (1, 2, 0))
                if ori.max() <= 1.5:
                    ori = (ori * 255.0).round()
                ori = np.clip(ori, 0, 255).astype(np.uint8)
                if ori.ndim == 2:
                    ori = np.repeat(ori[..., None], 3, 2)
                if ori.shape[2] == 1:
                    ori = np.repeat(ori, 3, 2)

                Hs, Ws = ori.shape[:2]
                if torch.is_tensor(stn_img):
                    vis = stn_img[0].detach().cpu().numpy()
                    if vis.ndim == 3 and vis.shape[0] in (1, 3):
                        vis = np.transpose(vis, (1, 2, 0))
                    vis = (vis * 255.0).clip(0, 255).astype(np.uint8)
                    if vis.ndim == 2:
                        vis = np.repeat(vis[..., None], 3, 2)
                    if vis.shape[2] == 1:
                        vis = np.repeat(vis, 3, 2)
                else:
                    vis = ori.copy()

                def rz(im, H=H_panel):
                    h, w = im.shape[:2]
                    if h == H:
                        return im
                    Wn = int(round(w * H / float(max(h, 1))))
                    return cv2.resize(im, (Wn, H), interpolation=cv2.INTER_LINEAR)

                L, R = rz(ori), rz(vis)
                WL, WR = L.shape[1], R.shape[1]

                bxywh_src = (bxywh.detach().cpu().numpy() * np.array([Ws, Hs, Ws, Hs], np.float32))
                xyxy_src = xywh2xyxy(torch.from_numpy(bxywh_src)).numpy().astype(np.float32)
                xyxy_left = self._scale_xyxy(xyxy_src, float(WL) / Ws, float(H_panel) / Hs).round().clip(0, 1e9).astype(int)
                xyxy_right = xyxy_left.copy()

                if isinstance(theta, torch.Tensor):
                    th = theta[0] if (theta.dim() == 3 and theta.shape[0] >= 1) else (theta if theta.dim() == 2 else None)
                    if th is not None:
                        with torch.no_grad():
                            tb = torch.from_numpy(bxywh_src).to(torch.float32)
                            tbw = self._warp_boxes_xywh_with_theta(tb, Ws, Hs, th).cpu().numpy().astype(np.float32)
                            xyxy_right = self._scale_xyxy(tbw, float(WR) / Ws, float(H_panel) / Hs).round().clip(0, 1e9).astype(int)

                def draw(img, boxes):
                    out = img.copy()
                    labels = bcls.view(-1).tolist()
                    for (x1, y1, x2, y2), c in zip(boxes, labels):
                        x1 = int(max(0, min(img.shape[1] - 1, x1)))
                        y1 = int(max(0, min(img.shape[0] - 1, y1)))
                        x2 = int(max(1, min(img.shape[1], x2)))
                        y2 = int(max(1, min(img.shape[0], y2)))
                        col = CLASS_COLORS[int(c) % len(CLASS_COLORS)]
                        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
                        out = _put_text(out, CLASS_LABELS.get(int(c), str(int(c))), (x1 + 4, max(0, y1 - 22)), col, 28)
                    return out

                Ld, Rd = draw(L, xyxy_left), draw(R, xyxy_right)
                Ld = _put_text(Ld, "ORIGINAL", (10, 10), (255, 255, 0), 42)
                Rd = _put_text(Rd, "STN",      (10, 10), (255, 255, 0), 42)

                ttheta = (theta[0] if (isinstance(theta, torch.Tensor) and theta.dim() == 3) else theta)
                if not isinstance(ttheta, torch.Tensor) or ttheta.numel() == 0:
                    ttheta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
                T = ttheta.detach().cpu().view(2, 3).numpy()
                Rd = _put_text(Rd, f"θ0: {T[0,0]:+0.5f} {T[0,1]:+0.5f} {T[0,2]:+0.5f}", (10, 60), (120, 255, 120), 28)
                Rd = _put_text(Rd, f"θ1: {T[1,0]:+0.5f} {T[1,1]:+0.5f} {T[1,2]:+0.5f}", (10, 94), (120, 255, 120), 28)
                dx, dy = float(T[0, 2]) * (Ws / 2.0), float(T[1, 2]) * (Hs / 2.0)
                Rd = _put_text(Rd, f"Δt ≈ ({dx:+.2f}px, {dy:+.2f}px)", (10, 128), (180, 220, 255), 26)

                both = np.concatenate([Ld, Rd], 1)
                pad = 16
                Hh, Ww = both.shape[:2]
                canvas = np.full((Hh + 2 * pad, Ww + 2 * pad, 3), (40, 40, 40), np.uint8)
                canvas[pad:pad + Hh, pad:pad + Ww] = both

                base = os.path.splitext(os.path.basename(str(path0) or f"sample{i:02d}"))[0]
                out = os.path.join(self.dbg_dir, f"{base}_epoch_{ep:03d}_{i:02d}.png")
                cv2.imwrite(out, canvas)
        finally:
            model.train(was_train)

        LOGGER.info(f"[DebugImages] saved for epoch {ep}")

class _DebugBgPairROIsLegacy:
    def __init__(self, epochs=(0,1,2,5,10), max_pairs=4):
        self.epochs = set(epochs or [])
        self.max_pairs = int(max_pairs or 4)

    def __call__(self, trainer):
        e = getattr(trainer, "epoch", -1)
        if e not in self.epochs:
            return
        batch = getattr(trainer, "_last_batch", None)
        if not isinstance(batch, dict):
            return
        need = ("pair_idx", "abn_mask", "im_files")
        if not all(k in batch for k in need):
            LOGGER.warning("[DebugBgPairROIs] missing pair_idx/abn_mask/im_files -> skip this epoch")
            return

    @staticmethod
    def _label_path(p: str) -> str | None:
        q = Path(p)
        lbl = Path(str(q).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)).with_suffix(".txt")
        return str(lbl) if lbl.exists() else None

    @staticmethod
    def _read_yolo_xyxy(lbl: str, W: int, H: int):
        out = []
        if not lbl or not os.path.exists(lbl):
            return out
        for line in Path(lbl).read_text(encoding="utf-8").splitlines():
            s = line.strip().split()
            if len(s) >= 5:
                c = int(float(s[0])); cx, cy, bw, bh = map(float, s[1:5])
                x1, y1 = int(max(0, (cx - bw / 2) * W)), int(max(0, (cy - bh / 2) * H))
                x2, y2 = int(min(W - 1, (cx + bw / 2) * W)), int(min(H - 1, (cy + bh / 2) * H))
                out.append((x1, y1, x2, y2, c))
        return out

    @staticmethod
    def _draw(img, boxes, per_cls=False, tag=None):
        out = img.copy()
        for (x1, y1, x2, y2, ci) in boxes:
            col = (CLASS_COLORS[int(ci) % len(CLASS_COLORS)] if per_cls else (0, 0, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            name = CLASS_LABELS.get(int(ci), str(int(ci)))
            cv2.putText(out, name, (x1 + 4, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        if tag:
            cv2.putText(out, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
        return out

    def on_train_epoch_end(self, t):
        e = int(getattr(t, "epoch", 0))
        if e not in self.epochs:
            return
        try:
            batch = next(iter(t.train_loader))
        except Exception as ex:
            LOGGER.warning(f"[DebugBgPairROIs] cannot fetch batch: {ex}"); return

        pair_idx, abn_mask, im_files = batch.get("pair_idx"), batch.get("abn_mask"), batch.get("im_files")
        if pair_idx is None or abn_mask is None or im_files is None:
            LOGGER.warning("[DebugBgPairROIs] missing pair_idx/abn_mask/im_files"); return

        save_dir = Path(str(t.save_dir)) / "bgpair_preview"; save_dir.mkdir(parents=True, exist_ok=True)
        rows, take = [], 0

        # Chuẩn hoá pair list: 1D (len=B) hoặc 2D (N,2)
        pairs = []
        if torch.is_tensor(pair_idx):
            arr = pair_idx.detach().cpu().numpy()
            if arr.ndim == 1:
                pairs = [(i, int(j)) for i, j in enumerate(arr.tolist()) if int(j) >= 0]
            elif arr.ndim >= 2 and arr.shape[1] >= 2:
                pairs = [(int(i), int(j)) for i, j in arr.tolist()]
        elif isinstance(pair_idx, (list, tuple)):
            if len(pair_idx) > 0 and isinstance(pair_idx[0], (list, tuple)):
                pairs = [(int(i), int(j)) for (i, j) in pair_idx]
            else:
                pairs = [(i, int(j)) for i, j in enumerate(pair_idx) if int(j) >= 0]

        for (i, j) in pairs:
            if take >= self.max_pairs:
                break
            # chọn đúng FG vs BG
            fg_i, bg_j = (i, j) if bool(abn_mask[i]) and not bool(abn_mask[j]) else ((j, i) if bool(abn_mask[j]) and not bool(abn_mask[i]) else (None, None))
            if fg_i is None:
                continue

            Lp, Rp = im_files[fg_i], im_files[bg_j]
            if not (os.path.exists(Lp) and os.path.exists(Rp)):
                continue
            Limg, Rimg = cv2.imread(Lp), cv2.imread(Rp)
            if Limg is None or Rimg is None:
                continue

            # overlay bbox từ label file
            def _ov(img, p, tag):
                H, W = img.shape[:2]
                lbl = self._label_path(p)
                boxes = self._read_yolo_xyxy(lbl, W, H)
                return self._draw(img, boxes, per_cls=True, tag=tag)

            Ld, Rd = _ov(Limg, Lp, "FG (abn)"), _ov(Rimg, Rp, "BG (pair)")
            h = 480
            def rz(im):
                r = h / max(1, im.shape[0]); w = int(round(im.shape[1] * r))
                return cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            Ld, Rd = rz(Ld), rz(Rd)
            pad = 12
            row = np.full((h, Ld.shape[1] + Rd.shape[1] + pad, 3), 30, np.uint8)
            row[:, :Ld.shape[1]] = Ld
            row[:, Ld.shape[1] + pad:] = Rd
            rows.append(row); take += 1

        if not rows:
            LOGGER.warning("[DebugBgPairROIs] no valid pairs in batch")
            return
        w = max(r.shape[1] for r in rows); out_h = sum(r.shape[0] for r in rows)
        out = np.zeros((out_h, w, 3), np.uint8); y = 0
        for r in rows:
            out[y:y + r.shape[0], :r.shape[1]] = r; y += r.shape[0]
        fp = save_dir / f"epoch_{e:03d}.jpg"
        cv2.imwrite(str(fp), out)
        LOGGER.info(f"[DebugBgPairROIs] saved {fp} ({len(rows)} pairs)")

# Fallback: nếu không import được bản debug tách riêng, dùng legacy tại chỗ
if DebugImages is None:
    DebugImages = _DebugImagesLegacy
if DebugBgPairROIs is None:
    DebugBgPairROIs = _DebugBgPairROIsLegacy

# =============================================================================
# 8) Validation helpers
# =============================================================================
class EnableValLoss:
    """Tính trung bình box/cls/dfl loss ở vòng validator nếu còn raw maps."""
    def __init__(self):
        self.enabled = True
        self.sum_box = 0.0
        self.sum_cls = 0.0
        self.sum_dfl = 0.0
        self.n = 0
        self._warned = False

    @staticmethod
    def _extract_three(items) -> Optional[Tuple[float, float, float]]:
        if items is None:
            return None
        if isinstance(items, (tuple, list)) and len(items) >= 3:
            try:
                return float(items[0]), float(items[1]), float(items[2])
            except Exception:
                return None
        if isinstance(items, dict):
            def _find(kname):
                for k, v in items.items():
                    if kname in k.lower():
                        try: return float(v)
                        except Exception: return None
                return None
            b, c, d = _find("box"), _find("cls"), _find("dfl")
            if b is not None and c is not None and d is not None:
                return b, c, d
        return None

    def on_val_start(self, trainer):
        self.sum_box = self.sum_cls = self.sum_dfl = 0.0
        self.n = 0
        self._warned = False

    def on_val_batch_end(self, trainer, batch, preds, batch_idx):
        if not self.enabled:
            return

        if isinstance(preds, tuple) and len(preds) == 2:
            raw_maps = preds[1]
        else:
            if not self._warned:
                LOGGER.warning("[Validator] skip loss this batch: no raw maps (postprocessed only)")
                self._warned = True
            return

        criterion = getattr(trainer, "criterion", None) or getattr(trainer, "loss", None)
        if criterion is None:
            if not self._warned:
                LOGGER.warning("[Validator] skip loss: trainer has no criterion")
                self._warned = True
            self.enabled = False
            return

        try:
            total, items = criterion(raw_maps, batch)
            trio = self._extract_three(items)
            if trio is None:
                if not self._warned:
                    LOGGER.warning("[Validator] skip loss: cannot parse loss items")
                    self._warned = True
                self.enabled = False
                return
            b, c, d = trio
            self.sum_box += b; self.sum_cls += c; self.sum_dfl += d; self.n += 1
        except Exception as ex:
            if not self._warned:
                LOGGER.warning(f"[Validator] skip loss: {ex}")
                self._warned = True
            self.enabled = False

    def on_val_end(self, trainer):
        if self.enabled and self.n > 0:
            mb = self.sum_box / self.n; mc = self.sum_cls / self.n; md = self.sum_dfl / self.n
            if getattr(trainer, "metrics", None) is None:
                trainer.metrics = {}
            trainer.metrics["val/box_loss"] = float(mb)
            trainer.metrics["val/cls_loss"] = float(mc)
            trainer.metrics["val/dfl_loss"] = float(md)
            LOGGER.info("VAL loss avg | box=%.4f cls=%.4f dfl=%.4f over %d batches", mb, mc, md, self.n)

# --- PATCH: ForceValArgs (replace class cũ nếu có) ---
class ForceValArgs:
    """
    Gắn lại các tham số infer/val cho validator.args mỗi epoch.
    Hỗ trợ conf, iou, agnostic, max_det (tự lọc kwargs an toàn).
    """
    _allowed = {"conf", "iou", "agnostic", "max_det"}

    def __init__(self, **kwargs):
        self.kw = {k: v for k, v in (kwargs or {}).items() if k in self._allowed}
        if not self.kw:
            LOGGER.warning("[Attach] ForceValArgs: no valid keys found -> nothing to enforce")

    def __call__(self, trainer):
        v = getattr(trainer, "validator", None)
        if v is None:
            LOGGER.warning("[ForceValArgs] validator not ready -> skip this epoch")
            return
        # ensure args exists
        if not hasattr(v, "args") or v.args is None:
            LOGGER.warning("[ForceValArgs] validator.args missing -> skip this epoch")
            return
        # set values
        for k, val in self.kw.items():
            try:
                setattr(v.args, k, val)
            except Exception as e:
                LOGGER.warning(f"[ForceValArgs] set {k}={val} failed: {e}")
        try:
            dd = {k: getattr(v.args, k, None) for k in self._allowed}
            LOGGER.info(f"[ForceValArgs] effective: {dd}")
        except Exception:
            pass


def apply_val_debug_overrides(
    yolo, conf: float = 0.001, iou: float = 0.5, *,
    epochs: set | list | tuple | None = None, every_n: int | None = None
):
    """Chỉ bật lưu ảnh/plots ở epoch thỏa điều kiện (epochs hoặc every_n)."""
    epochs = (set(epochs) if epochs is not None else None)
    every_n = int(every_n) if (every_n is not None and every_n > 0) else None

    def _on_val_start(v):
        e = int(getattr(getattr(v, "trainer", v), "epoch", 0))
        active = True
        if epochs is not None:
            active = (e in epochs)
        if active and every_n is not None:
            active = (e % every_n == 0)

        v.args.conf = float(conf); v.args.iou = float(iou)
        if active:
            v.args.save_txt = True; v.args.save_conf = True; v.args.save = True; v.args.plots = True
            LOGGER.info(f"[VAL/Debug] epoch {e}: SAVE=ON (conf={conf}, iou={iou})")
        else:
            v.args.save_txt = False; v.args.save_conf = False; v.args.save = False; v.args.plots = False
            LOGGER.info(f"[VAL/Debug] epoch {e}: SAVE=OFF (conf={conf}, iou={iou})")

    yolo.add_callback("on_val_start", _on_val_start)

def register_val_probe(yolo, gt_labels_dir: str | None = None):
    LOGGER.info(f"[VAL/Probe] GT labels dir: {gt_labels_dir}" if gt_labels_dir else "[VAL/Probe] no GT dir provided (skipping)")

def register_results_csv_guard(yolo):
    def _patch(tr):
        if getattr(tr, "_results_csv_guard", False): return
        tr._results_csv_guard = True
        _orig = getattr(tr, "read_results_csv", None)
        def safe_reader(self):
            try: return _orig(self)
            except Exception:
                LOGGER.warning("[CSVFix] Patched trainer.read_results_csv with guard")
                return {}
        if _orig is not None: setattr(tr.__class__, "read_results_csv", safe_reader)
    yolo.add_callback("on_train_start", _patch)

# =============================================================================
# 9) Misc helpers
# =============================================================================
def set_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class SaveLastBestOnly:
    def on_train_start(self, trainer):
        try:
            trainer.save_period = 0
        except Exception:
            pass
    def on_fit_epoch_end(self, trainer):
        wdir = Path(trainer.save_dir) / "weights"
        if wdir.exists():
            for p in wdir.glob("epoch*.pt"):
                try: p.unlink()
                except Exception: pass

class FixFinalEvalModel:
    def on_val_start(self, v):
        m = getattr(v, "model", None)
        if isinstance(m, (str, bytes, Path)):
            tr = getattr(v, "trainer", None)
            mdl = getattr(tr, "model", None) if tr is not None else None
            if mdl is not None:
                v.model = mdl
                LOGGER.warning("[FinalEvalFix] Validator.model is a Path -> using in-memory trainer.model instead.")

# =============================================================================
# 10) Attach helpers (không giấu tham số)
# =============================================================================
def ensure_detection_criterion(trainer) -> None:
    """
    Ưu tiên v8DetectionLoss (ultralytics.utils.loss.v8DetectionLoss).
    Nếu lỗi, fallback -> model.loss().
    """
    try:
        from ultralytics.utils.loss import v8DetectionLoss
        trainer.criterion = v8DetectionLoss(trainer.model)
        LOGGER.info("[VAL/Loss] set criterion = ultralytics.utils.loss.v8DetectionLoss (loss.py)")
    except Exception as e:
        LOGGER.warning(f"[VAL/Loss] cannot init v8DetectionLoss: {e}; fallback to model.loss")
        try:
            def _wrap(preds, batch):
                return trainer.model.loss(preds, batch)
            trainer.criterion = _wrap
            LOGGER.info("[VAL/Loss] using trainer.model.loss wrapper")
        except Exception as e2:
            LOGGER.warning(f"[VAL/Loss] model.loss wrapper failed too: {e2}")

def val_tune(trainer, *, plots=True, conf=0.25, iou=0.50, max_det=300, half_if_cuda=True) -> None:
    try:
        v = getattr(trainer, "validator", None)
        if v is None:
            return
        v.args.plots = bool(plots)
        v.args.conf = float(conf)
        v.args.iou = float(iou)
        v.args.max_det = int(max_det)
        if half_if_cuda and getattr(trainer.device, "type", "cpu") != "cpu":
            v.args.half = True
        LOGGER.info(f"[VAL/Setup] plots={v.args.plots} conf={v.args.conf} iou={v.args.iou} "
                    f"max_det={v.args.max_det} half={getattr(v.args,'half',None)}")
    except Exception as e:
        LOGGER.warning(f"[VAL/Setup] tune failed: {e}")

def attach_callbacks(
    yolo,
    *,
    # STN
    stn_cfg: dict | None = None,
    publish_theta: bool = False,
    # SupCon
    supcon_inject: dict | None = None,
    supcon_schedule: str | int | None = None,
    supcon_reinforce_keys: list[str] | None = None,
    supcon_tap: dict | None = None,
    supcon_proj_attach: dict | None = None,
    supcon_percent_logger: bool = False,
    link_trainer_to_loss: bool = False,
    sync_epoch_to_loss: bool = False,
    # Safety
    nan_guard: dict | bool | None = None,
    batch_sanity: dict | bool | None = None,
    # Validation
    enable_val_loss: bool = False,
    val_force_args: dict | None = None,
    val_debug_overrides: dict | None = None,
    ensure_val_criterion: bool = True,
    val_trap: bool | dict | None = None,
    # Debug
    debug_images: dict | None = None,
    debug_bgpair: dict | None = None,
    # Misc
    results_csv_guard: bool = False,
    final_eval_fix: bool = False,
    save_last_best_only: bool = False,
    val_probe_gt_dir: str | None = None,
):
    # ----- STN -----
    if stn_cfg:
        try:
            stn = STNControl(
                freeze_epochs=int(stn_cfg.get("freeze_epochs", 0)),
                stn_warmup=int(stn_cfg.get("stn_warmup", 0)),
                tmax=float(stn_cfg.get("tmax", 0.20)),
                smin=float(stn_cfg.get("smin", 0.90)),
                smax=float(stn_cfg.get("smax", 1.10)),
                val_identity=bool(int(stn_cfg.get("val_identity", 1))),
                log=bool(int(stn_cfg.get("log", 0))),
            )
            yolo.add_callback("on_pretrain_routine_start", stn.on_train_epoch_start)  # epoch 0
            yolo.add_callback("on_train_epoch_start",       stn.on_train_epoch_start)
            yolo.add_callback("on_val_start",               stn.on_val_start)
            yolo.add_callback("on_val_end",                 stn.on_val_end)
            LOGGER.info("[Attach] STNControl attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] STNControl failed: {e}")

    if publish_theta:
        try:
            pub = PublishThetaToStateV2(verbose=True)
            for hook in ("on_train_start", "on_train_end", "on_val_start", "on_val_end"):
                yolo.add_callback(hook, getattr(pub, hook))
            LOGGER.info("[Attach] PublishThetaToStateV2 attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] PublishThetaToStateV2 failed: {e}")

    # ----- SupCon -----
    if supcon_inject:
        try:
            inj = InjectSupConArgsMinimal(**supcon_inject)
            yolo.add_callback("on_pretrain_routine_start", inj.on_pretrain_routine_start)
            yolo.add_callback("on_pretrain_routine_end",   inj.on_pretrain_routine_end)
            yolo.add_callback("on_train_start",            inj.on_train_start)
            LOGGER.info("[Attach] InjectSupConArgsMinimal attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] InjectSupConArgsMinimal failed: {e}")

    if supcon_schedule is not None:
        try:
            sched_str = f"{int(supcon_schedule)}-" if isinstance(supcon_schedule, int) else str(supcon_schedule)
            sch = SupConScheduler(sched_str, default_on=0)
            yolo.add_callback("on_train_epoch_start", sch.on_train_epoch_start)
            LOGGER.info(f"[Attach] SupConScheduler('{sched_str}') attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] SupConScheduler failed: {e}")

    # --- SupCon reinforce (API mới, không còn 'keys') ---
    if supcon_reinforce_keys is not None:  # giữ điều kiện để “bật/tắt” tính năng
        try:
            rf = ReinforceSupConToLoss()
            yolo.add_callback("on_train_start", rf.on_train_start)
            yolo.add_callback("on_fit_epoch_start", rf.on_fit_epoch_start)
            yolo.add_callback("on_train_epoch_start", rf.on_train_epoch_start)
            LOGGER.info("[Attach] ReinforceSupConToLoss attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] ReinforceSupConToLoss failed: {e}")

    if supcon_tap:
        try:
            tap = TapSTNFeat(**supcon_tap)
            yolo.add_callback("on_train_start",      tap.on_train_start)
            yolo.add_callback("on_train_end",        tap.on_train_end)
            yolo.add_callback("on_train_batch_end",  tap.on_train_batch_end)
            LOGGER.info("[Attach] TapSTNFeat attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] TapSTNFeat failed: {e}")

    if supcon_proj_attach:
        try:
            pa = AttachSupConProjToOptim(**supcon_proj_attach)
            for h in ("on_train_start", "on_train_batch_start", "on_train_batch_end"):
                yolo.add_callback(h, getattr(pa, h))
            LOGGER.info("[Attach] AttachSupConProjToOptim attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] AttachSupConProjToOptim failed: {e}")

    if supcon_percent_logger:
        try:
            plog = SupConPercentLogger()
            yolo.add_callback("on_train_batch_end", plog.on_train_batch_end)
            yolo.add_callback("on_train_epoch_end", plog.on_train_epoch_end)
            LOGGER.info("[Attach] SupConPercentLogger attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] SupConPercentLogger failed: {e}")

    if link_trainer_to_loss:
        try:
            lnk = LinkTrainerToLoss()
            yolo.add_callback("on_train_start", lnk.on_train_start)
            yolo.add_callback("on_train_epoch_start", lnk.on_train_epoch_start)
            yolo.add_callback("on_train_batch_start", lnk.on_train_batch_start)
        except Exception as e:
            LOGGER.warning(f"[Attach] LinkTrainerToLoss failed: {e}")

    if sync_epoch_to_loss:
        try:
            syn = SyncEpochToLoss()
            yolo.add_callback("on_train_epoch_start", syn.on_train_epoch_start)
            yolo.add_callback("on_train_batch_start", syn.on_train_batch_start)
        except Exception as e:
            LOGGER.warning(f"[Attach] SyncEpochToLoss failed: {e}")

    # ----- Safety -----
    if nan_guard:
        try:
            cfg = {} if nan_guard is True else dict(nan_guard)
            ng = LossNaNGuard(**cfg)
            yolo.add_callback("on_train_batch_end", ng.on_train_batch_end)
            LOGGER.info("[Attach] LossNaNGuard attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] LossNaNGuard failed: {e}")

    if batch_sanity:
        try:
            cfg = {} if batch_sanity is True else dict(batch_sanity)
            bs = BatchSanityFilter(**cfg)
            yolo.add_callback("on_train_batch_start", bs.on_train_batch_start)
            LOGGER.info("[Attach] BatchSanityFilter attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] BatchSanityFilter failed: {e}")

    # ----- Validation -----
    if enable_val_loss:
        try:
            ev = EnableValLoss()
            yolo.add_callback("on_val_start", ev.on_val_start)
            yolo.add_callback("on_val_batch_end", ev.on_val_batch_end)
            yolo.add_callback("on_val_end", ev.on_val_end)
            LOGGER.info("[Attach] EnableValLoss attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] EnableValLoss failed: {e}")

    if ensure_val_criterion:
        try:
            yolo.add_callback("on_fit_start", ensure_detection_criterion)
            yolo.add_callback("on_train_start", ensure_detection_criterion)
            yolo.add_callback("on_val_start", ensure_detection_criterion)
        except Exception as e:
            LOGGER.warning(f"[Attach] ensure_detection_criterion failed: {e}")

    if val_force_args:
        try:
            fa = ForceValArgs(**val_force_args)
            yolo.add_callback("on_val_start", fa)  # class là callable
            LOGGER.info("[Attach] ForceValArgs attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] ForceValArgs failed: {e}")

    if val_debug_overrides:
        try:
            apply_val_debug_overrides(
                yolo,
                conf=float(val_debug_overrides.get("conf", 0.001)),
                iou=float(val_debug_overrides.get("iou", 0.5)),
                epochs=val_debug_overrides.get("epochs", None),
                every_n=val_debug_overrides.get("every_n", None),
            )
            LOGGER.info("[Attach] val_debug_overrides attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] val_debug_overrides failed: {e}")

    # >>> NEW: ValTrap optional
    if val_trap:
        try:
            vt = ValTrap() if val_trap is True else ValTrap()
            yolo.add_callback("on_val_start", vt.on_val_start)
            yolo.add_callback("on_val_batch_end", vt.on_val_batch_end)
            yolo.add_callback("on_val_end", vt.on_val_end)
            LOGGER.info("[Attach] ValTrap attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] ValTrap failed: {e}")

    # ----- Debug -----
    if debug_images:
        try:
            cfg = dict(debug_images)
            epochs = cfg.get("epochs", {0})
            max_images = int(cfg.get("max_images", 5))
            dbg = DebugImages(epochs=set(map(int, epochs)), max_images=max_images)
            yolo.add_callback("on_train_epoch_end", dbg.on_train_epoch_end)
            LOGGER.info(f"[Attach] DebugImages attached (epochs={sorted(list(epochs))}, max_images={max_images})")
        except Exception as e:
            LOGGER.warning(f"[Attach] DebugImages failed: {e}")

    if debug_bgpair:
        try:
            cfg = dict(debug_bgpair)
            dbp = DebugBgPairROIs(epochs=set(map(int, cfg.get("epochs", {0}))), max_pairs=int(cfg.get("max_pairs", 4)))
            yolo.add_callback("on_train_epoch_end", dbp.on_train_epoch_end)
            LOGGER.info("[Attach] DebugBgPairROIs attached")
        except Exception as e:
            LOGGER.warning(f"[Attach] DebugBgPairROIs failed: {e}")

    # ----- Misc -----
    if val_probe_gt_dir:
        try:
            register_val_probe(yolo, gt_labels_dir=str(val_probe_gt_dir))
        except Exception:
            pass

    if results_csv_guard:
        try:
            register_results_csv_guard(yolo)
        except Exception:
            pass

    if final_eval_fix:
        try:
            fix = FixFinalEvalModel()
            yolo.add_callback("on_val_start", fix.on_val_start)
        except Exception:
            pass

    if save_last_best_only:
        try:
            saver = SaveLastBestOnly()
            yolo.add_callback("on_train_start",  saver.on_train_start)
            yolo.add_callback("on_fit_epoch_end", saver.on_fit_epoch_end)
        except Exception:
            pass

# =============================================================================
# 11) Pairing registry: wrap collate để luôn có pair_idx/abn_mask/im_files
# =============================================================================
def normalize_bgpair_map(map_path: str):
    with open(map_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    def _norm(p): return os.path.normpath(p).lower()
    normed = { _norm(k): [_norm(v) for v in vs] for k, vs in raw.items() }

    by_base = {}
    for k, vs in normed.items():
        kb = os.path.basename(k)
        by_base.setdefault(kb, set()).update(os.path.basename(v) for v in vs)
    by_base = {k: sorted(list(v)) for k, v in by_base.items()}
    return normed, by_base


def _build_abn_mask_from_labels(batch):
    B = int(batch['img'].shape[0])
    abn = [0] * B
    if 'batch_idx' in batch and 'cls' in batch and len(batch['batch_idx']) > 0:
        for b in batch['batch_idx'].tolist():
            if 0 <= b < B:
                abn[b] = 1
    return abn


def _make_pairs_in_batch(im_files, bgmap_abs, bgmap_base):
    im_files_norm = [os.path.normpath(p).lower() for p in im_files]
    by_name = {os.path.basename(p): i for i, p in enumerate(im_files_norm)}
    in_batch = set(im_files_norm)
    pair_idx = [-1] * len(im_files_norm)

    # exact path
    for i, p in enumerate(im_files_norm):
        cand = bgmap_abs.get(p, [])
        j = next(
            (by_name[os.path.basename(c)] for c in cand
             if os.path.normpath(c).lower() in in_batch),
            -1,
        )
        pair_idx[i] = j

    # fallback basename
    for i, p in enumerate(im_files_norm):
        if pair_idx[i] != -1:
            continue
        base = os.path.basename(p)
        cand_base = bgmap_base.get(base, [])
        j = next((by_name[b] for b in cand_base if b in by_name), -1)
        pair_idx[i] = j

    return pair_idx


class PairingCollate:
    """
    Callable top-level (pickle được) để bọc collate_fn gốc và bơm:
      - im_files (chuẩn hóa path)
      - pair_idx
      - abn_mask
    """
    def __init__(self, default_collate_fn, bgmap_abs, bgmap_base):
        self.default_collate_fn = default_collate_fn
        self.bgmap_abs = bgmap_abs
        self.bgmap_base = bgmap_base
        # flag để register_pairing biết là đã wrap rồi
        self._pair_collate_wrapped = True

    def __call__(self, batch):
        b = self.default_collate_fn(batch)
        im_files = b.get("im_files", None)
        if im_files is None:
            return b

        b["im_files"] = [os.path.normpath(p) for p in im_files]
        b["pair_idx"] = _make_pairs_in_batch(
            b["im_files"], self.bgmap_abs, self.bgmap_base
        )
        b["abn_mask"] = _build_abn_mask_from_labels(b)
        return b


def register_pairing(yolo, *, bgpair_map: str, batch_size: int | None = None):
    """
    1) Load & chuẩn hoá pairing map.
    2) on_fit_epoch_start: bọc train_loader.collate_fn để bơm im_files/pair_idx/abn_mask.
    3) Nếu có UsePairedLoader của bạn, vẫn attach (không phụ thuộc).
    """
    try:
        bg_abs, bg_base = normalize_bgpair_map(bgpair_map)
        LOGGER.info(f"[Pairing] map loaded & normalized: {bgpair_map}")
    except Exception as e:
        LOGGER.warning(f"[Pairing] cannot read map: {e}")
        bg_abs, bg_base = {}, {}

    # optional: call your UsePairedLoader if available
    try:
        import inspect
        from ultralytics.utils.stn_pairing import UsePairedLoader

        sig = inspect.signature(UsePairedLoader)
        kwargs = {}
        if "bgpair_map" in sig.parameters:
            kwargs["bgpair_map"] = bgpair_map
        if "batch" in sig.parameters and batch_size is not None:
            kwargs["batch"] = int(batch_size)
        elif "batch_size" in sig.parameters and batch_size is not None:
            kwargs["batch_size"] = int(batch_size)

        pl = UsePairedLoader(**kwargs)
        for hook_name in ("on_fit_epoch_start", "on_train_start", "on_pretrain_routine_end"):
            if hasattr(pl, hook_name):
                yolo.add_callback(hook_name, getattr(pl, hook_name))
                LOGGER.info(f"[Pairing] UsePairedLoader.{hook_name} attached (optional)")
                break
    except Exception:
        pass

    def _patch_loader(trainer):
        dl = getattr(trainer, "train_loader", None)
        if dl is None or not hasattr(dl, "collate_fn") or dl.collate_fn is None:
            LOGGER.warning("[Pairing] train_loader.collate_fn not available; skip collate wrap")
            return

        # tránh wrap nhiều lần
        if getattr(dl.collate_fn, "_pair_collate_wrapped", False):
            return

        dl.collate_fn = PairingCollate(dl.collate_fn, bg_abs, bg_base)
        LOGGER.info("[Pairing] collate_fn wrapped with pairing metadata")

    yolo.add_callback("on_pretrain_routine_end", _patch_loader)
    yolo.add_callback("on_train_start",          _patch_loader)
    yolo.add_callback("on_fit_epoch_start",      _patch_loader)


# =============================================================================
# 12) ValTrap + Val safety: không mất mAP, ép NMS, tính val-loss nhẹ sau cùng
# =============================================================================
class ValTrap:
    """
    Ghi log chi tiết vòng validation để truy nguồn lỗi mAP/P/R/CSV.
    - on_val_start: chụp trạng thái validator (training graph? criterion? BN? args? dataset?),
                    snapshot kích thước results.csv trước khi chạy.
    - on_val_batch_end (batch đầu): log shapes của feats/preds + min/max/NaN/Inf check.
    - on_val_end: log metrics/stats/speed + tail(2) của results.csv + thời lượng validate.
    Kết quả in console và (nếu có save_dir) ghi file: <save_dir>/__val_trap.log
    """
    def __init__(self, log_path=None):
        self._wrote_first_batch = False
        from pathlib import Path as _P
        self.log_path = (_P(log_path) if log_path is not None else None)
        self._logfile = None
        self._t0 = None
        self._csv_size_before = None

    # ---------------- internals ----------------
    def _init_logfile(self, trainer):
        try:
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._logfile = self.log_path
                return
            sd = getattr(trainer, "save_dir", None)
            if sd:
                from pathlib import Path
                self._logfile = Path(sd) / "__val_trap.log"
        except Exception:
            self._logfile = None

    def _write(self, obj, tag, payload: dict):
        from ultralytics.utils import LOGGER
        # thêm epoch nếu có
        try:
            ep = getattr(obj, "epoch", None)
            if ep is not None:
                payload = {"epoch": ep, **payload}
        except Exception:
            pass

        # patch: if metrics missing on trainer, try validator.metrics
        if "metrics" in payload and payload.get("metrics", None) is None:
            try:
                v = getattr(obj, "validator", None) if not hasattr(obj, "metrics") else obj
                if v is not None and getattr(v, "metrics", None) is not None:
                    payload["metrics"] = v.metrics
            except Exception:
                pass

        # compact metrics if they are DetMetrics to avoid huge logs
        # compact metrics/results to avoid huge log spam
        try:
            m = payload.get("metrics", None)
            rd = None
            if hasattr(m, "results_dict"):
                rd_attr = m.results_dict
                rd = rd_attr() if callable(rd_attr) else rd_attr
            summary = None
            if isinstance(rd, dict):
                sel_keys = [
                    "metrics/precision(B)",
                    "metrics/recall(B)",
                    "metrics/mAP50(B)",
                    "metrics/mAP50-95(B)",
                    "fitness",
                    "val/loss",
                ]
                summary = {k: rd.get(k) for k in sel_keys if k in rd}
            payload["metrics"] = summary if summary is not None else None
            # trim results.csv tail if present
            if "results.csv.tail2" in payload:
                tail = payload["results.csv.tail2"]
                if isinstance(tail, (list, tuple)) and len(tail) > 0:
                    payload["results.csv.tail2"] = tail[-1]
            # drop heavy fields if they creep in
            for k in list(payload.keys()):
                if any(tok in k for tok in ("curves_results", "confusion_matrix", "ap_class_index", "box")):
                    payload[k] = None
        except Exception:
            pass

        # drop/compact stats to avoid huge tensors
        if "stats" in payload:
            try:
                st = payload.get("stats")
                if isinstance(st, (list, tuple)):
                    payload["stats"] = f"len={len(st)}"
                else:
                    payload["stats"] = "SKIPPED"
            except Exception:
                payload["stats"] = "SKIPPED"

        # console
        try:
            msg = f"[ValTrap/{tag}] " + " | ".join(f"{k}={v}" for k, v in payload.items())
            LOGGER.info(msg)
        except Exception:
            msg = None

        # file
        try:
            if self._logfile is None:
                self._init_logfile(trainer)
            if self._logfile is not None and msg is not None:
                with open(self._logfile, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
        except Exception:
            pass

    @staticmethod
    def _safe_shapes(xs):
        try:
            out = []
            for t in (xs or []):
                try:
                    out.append(tuple(getattr(t, "shape", None)))
                except Exception:
                    out.append(type(t).__name__)
            return out
        except Exception:
            return None

    @staticmethod
    def _tensor_stats(x):
        try:
            import torch
            if not isinstance(x, torch.Tensor):
                return None
            return dict(
                dtype=str(x.dtype),
                shape=tuple(x.shape),
                min=float(torch.nan_to_num(x).min().item()),
                max=float(torch.nan_to_num(x).max().item()),
                has_nan=bool(torch.isnan(x).any().item()),
                has_inf=bool(torch.isinf(x).any().item()),
            )
        except Exception:
            return None

    @staticmethod
    def _bn_train_count(model):
        try:
            import torch.nn as nn
            cnt = 0
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and getattr(m, "training", False):
                    cnt += 1
            return cnt
        except Exception:
            return None

    def _csv_filesize(self, trainer):
        try:
            from pathlib import Path
            p = Path(trainer.save_dir) / "results.csv"
            return p.stat().st_size if p.exists() else None
        except Exception:
            return None

    # ---------------- YOLO callbacks ----------------
    def on_val_start(self, obj):
        """obj may be validator or trainer."""
        self._wrote_first_batch = False
        self._t0 = __import__("time").time()
        v = obj if hasattr(obj, "model") and hasattr(obj, "metrics") else getattr(obj, "validator", None)
        payload = {}

        if v is not None:
            # các cờ/thuộc tính quan trọng của validator
            for k in ["_compute_loss_during_val", "_use_training_model", "training"]:
                payload[k] = getattr(v, k, None)

            # criterion có sẵn chưa?
            try:
                crit = getattr(trainer, "criterion", None)
                payload["criterion"] = crit.__class__.__name__ if crit is not None else None
            except Exception as e:
                payload["criterion_err"] = str(e)

            # trạng thái model trong validator (eval/train) + BN train count
            try:
                m = getattr(v, "model", None)
                payload["model.training"] = (getattr(m, "training", None) if m is not None else None)
                bn_train = self._bn_train_count(m) if m is not None else None
                payload["bn.train_count"] = bn_train
            except Exception as e:
                payload["model.training_err"] = str(e)

            # dataloader length
            try:
                dl = getattr(v, "dataloader", None)
                payload["dataloader_len"] = (len(dl) if dl is not None else None)
            except Exception as e:
                payload["dataloader_len_err"] = str(e)

            # args rút gọn
            args = getattr(v, "args", None)
            if args is not None:
                for a in ["split", "conf", "iou", "agnostic", "max_det", "half", "nms"]:
                    payload[f"args.{a}"] = getattr(args, a, None)

            # dataset info (names/nc) nếu có
            try:
                data = getattr(v, "data", None) or getattr(trainer, "data", None) or {}
                names = data.get("names", None)
                nc = data.get("nc", (len(names) if isinstance(names, (list, tuple)) else None))
                if names and isinstance(names, (list, tuple)):
                    payload["data.nc"] = nc
                    payload["data.names.head"] = names[:min(5, len(names))]
                else:
                    payload["data.nc"] = nc
            except Exception:
                pass

        # snapshot kích thước results.csv trước khi val
        self._csv_size_before = self._csv_filesize(trainer)
        payload["results.csv.size_before"] = self._csv_size_before

        self._write(obj, "start", payload)

    def on_val_batch_end(self, obj):
        # chỉ log batch đầu
        if self._wrote_first_batch:
            return
        v = obj if hasattr(obj, "model") and hasattr(obj, "metrics") else getattr(obj, "validator", None)
        payload = {}

        # feats: khi validator tính val-loss (legacy path) sẽ có
        try:
            feats = getattr(v, "_last_feats_for_loss", None)
            if feats is not None:
                payload["feats_shapes"] = self._safe_shapes(feats)
                if len(feats) > 0:
                    payload["feats[0]_stats"] = self._tensor_stats(feats[0])
        except Exception as e:
            payload["feats_err"] = str(e)

        # preds: khi chạy nhánh predict+NMS để tính mAP
        try:
            preds = getattr(v, "_last_preds", None)
            if preds is not None:
                if isinstance(preds, (list, tuple)) and len(preds) and hasattr(preds[0], "shape"):
                    payload["preds_shapes"] = [tuple(p.shape) for p in preds]
                    payload["preds[0]_stats"] = self._tensor_stats(preds[0])
                else:
                    payload["preds_type"] = type(preds).__name__
        except Exception as e:
            payload["preds_err"] = str(e)

        self._write(obj, "first_batch", payload)
        self._wrote_first_batch = True

    def on_val_end(self, obj):
        v = obj if hasattr(obj, "model") and hasattr(obj, "metrics") else getattr(obj, "validator", None)
        payload = {}

        # metrics summary only
        try:
            m = getattr(v, "metrics", None)
            rd = None
            if hasattr(m, "results_dict"):
                rd_attr = m.results_dict
                rd = rd_attr() if callable(rd_attr) else rd_attr
            if isinstance(rd, dict):
                sel = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness", "val/loss"]
                payload["metrics"] = {k: rd.get(k) for k in sel if k in rd}
            else:
                payload["metrics"] = None
        except Exception:
            payload["metrics"] = None

        # thời lượng validate
        try:
            import time
            if self._t0 is not None:
                payload["duration_s"] = round(time.time() - self._t0, 3)
        except Exception:
            pass

        self._write(obj, "end", payload)


# -----------------------------------------------------------------------------
# Các callback "an toàn" cho validate: ép NMS + ưu tiên nhánh predict + post val-loss
# -----------------------------------------------------------------------------
class ForceValArgsNMS:
    """Ép các tham số quan trọng cho validator (đặc biệt là nms=True)."""
    def __init__(self, *, conf=0.01, iou=0.5, max_det=300, half_if_cuda=True, nms=True):
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.half_if_cuda = bool(half_if_cuda)
        self.nms = bool(nms)

    def __call__(self, trainer):
        try:
            v = getattr(trainer, "validator", None)
            if v is None:
                return
            v.args.conf = self.conf
            v.args.iou = self.iou
            v.args.max_det = self.max_det
            v.args.nms = self.nms
            if self.half_if_cuda and getattr(trainer.device, "type", "cpu") != "cpu":
                v.args.half = True
            from ultralytics.utils import LOGGER
            LOGGER.info(f"[VAL/Args] nms={v.args.nms} conf={v.args.conf} iou={v.args.iou} "
                        f"max_det={v.args.max_det} half={getattr(v.args,'half',None)}")
        except Exception as e:
            from ultralytics.utils import LOGGER
            LOGGER.warning(f"[VAL/Args] set failed: {e}")


class PreferPredictPath:
    """
    Đảm bảo vòng validate luôn chạy nhánh predict+NMS để có mAP.
    (Val-loss sẽ được tính sau bằng post-hook nếu cần.)
    """
    def __call__(self, trainer):
        try:
            v = getattr(trainer, "validator", None)
            if v is None:
                return
            # tắt compute-loss-trong-val để nhánh predict chạy chắc chắn
            if hasattr(v, "_compute_loss_during_val"):
                setattr(v, "_compute_loss_during_val", False)
            # dùng eval model cho nhánh predict thay vì train graph
            if hasattr(v, "_use_training_model"):
                setattr(v, "_use_training_model", False)
            from ultralytics.utils import LOGGER
            LOGGER.info("[VAL/Mode] prefer predict path: _compute_loss_during_val=False, _use_training_model=False")
        except Exception as e:
            from ultralytics.utils import LOGGER
            LOGGER.warning(f"[VAL/Mode] prefer predict path failed: {e}")


class ValLossPostLite:
    """
    Tính val-loss nhanh sau khi đã tính mAP (không ảnh hưởng path predict).
    Duyệt k batch đầu của dataloader val bằng training criterion và in loss tổng quan.
    """
    def __init__(self, k=2):
        self.k = int(k)

    def on_val_end(self, trainer):
        if self.k <= 0:
            return
        try:
            from ultralytics.utils import LOGGER
            import torch
            v = getattr(trainer, "validator", None)
            crit = getattr(trainer, "criterion", None)
            if v is None or crit is None:
                return
            dl = getattr(v, "dataloader", None)
            if dl is None:
                return

            dev = getattr(trainer, "device", torch.device("cpu"))
            n = 0
            losses = []

            model_train = trainer.model
            # tạm thời đặt BN sang eval để không update running stats
            bns, bn_modes = [], []
            import torch.nn as nn
            for m in model_train.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    bns.append(m)
                    bn_modes.append(m.training)
                    m.eval()

            model_train.train()  # dùng train graph nhưng BN không update
            with torch.no_grad():
                for i, batch in enumerate(dl):
                    if i >= self.k:
                        break
                    imgs = batch["img"].to(dev, non_blocking=True).float() / 255.0
                    feats = model_train(imgs)  # train graph
                    loss = crit(feats, batch)
                    # Chuẩn hóa thành float tổng quát để báo cáo
                    try:
                        if torch.is_tensor(loss):
                            losses.append(float(loss.detach().item()))
                        elif isinstance(loss, (list, tuple)):
                            s = 0.0
                            for x in loss:
                                try:
                                    s += float(x.detach().item() if torch.is_tensor(x) else float(x))
                                except Exception:
                                    pass
                            losses.append(float(s))
                        else:
                            losses.append(float(loss))
                    except Exception:
                        pass
                    n += 1

            # restore BN modes
            for m, t in zip(bns, bn_modes):
                m.train(t)

            if n > 0 and len(losses) > 0:
                avg = sum(losses) / len(losses)
                LOGGER.info(f"[VAL/LOSS post] k={n} avg_total={avg:.4f}")
        except Exception as e:
            from ultralytics.utils import LOGGER
            LOGGER.warning(f"[VAL/LOSS post] failed: {e}")


# -----------------------------------------------------------------------------
# Tiện ích: gắn ValTrap + safety callback vào hệ callback của YOLO
# -----------------------------------------------------------------------------
def attach_valtrap(trainer):
    """Đăng ký ValTrap trực tiếp vào hệ callback của YOLO (dùng khi bạn có 'trainer' sẵn)."""
    try:
        from ultralytics.utils import callbacks, LOGGER
        cb = ValTrap()
        reg = {
            "on_val_start": [cb.on_val_start],
            "on_val_batch_end": [cb.on_val_batch_end],
            "on_val_end": [cb.on_val_end],
        }
        callbacks.add_integration_callbacks(trainer, reg)
        LOGGER.info("[Attach] ValTrap attached")
        return cb
    except Exception as e:
        from ultralytics.utils import LOGGER
        LOGGER.warning(f"[Attach] ValTrap failed: {e}")
        return None


def register_val_trap_and_safety(
    yolo,
    *,
    conf=0.01,
    iou=0.50,
    max_det=300,
    half_if_cuda=True,
    nms=True,
    post_loss_k=2,
    log_file=None,
):
    """
    Dùng khi bạn có object 'yolo' (YOLO(...)). Hàm này sẽ:
      - Gắn ValTrap để bạn có log start/first_batch/end cho vòng validate.
      - Ép NMS + tham số val "an toàn" (ForceValArgsNMS).
      - Ưu tiên nhánh predict để luôn có mAP (PreferPredictPath).
      - (Tuỳ chọn) Tính val-loss nhẹ sau khi mAP xong (ValLossPostLite).
    """
    # ValTrap
    vt = ValTrap(log_path=log_file)

    # attach ValTrap to YOLO callback system (some trainers forward these)
    yolo.add_callback("on_val_start", vt.on_val_start)
    yolo.add_callback("on_val_batch_end", vt.on_val_batch_end)
    yolo.add_callback("on_val_end", vt.on_val_end)

    # ensure validator (once constructed) also has ValTrap callbacks
    def _attach_vt_to_validator(trainer):
        try:
            v = getattr(trainer, "validator", None)
            if v is None:
                return
            for name, fn in (
                ("on_val_start", vt.on_val_start),
                ("on_val_batch_end", vt.on_val_batch_end),
                ("on_val_end", vt.on_val_end),
            ):
                try:
                    cblist = v.callbacks.get(name, [])
                    if fn not in cblist:
                        cblist.append(fn)
                        v.callbacks[name] = cblist
                except Exception:
                    pass
        except Exception:
            return

    yolo.add_callback("on_train_start", _attach_vt_to_validator)
    yolo.add_callback("on_pretrain_routine_end", _attach_vt_to_validator)

    # Force NMS + args
    fna = ForceValArgsNMS(conf=conf, iou=iou, max_det=max_det, half_if_cuda=half_if_cuda, nms=nms)
    yolo.add_callback("on_val_start", fna)

    # Prefer predict path
    ppp = PreferPredictPath()
    yolo.add_callback("on_val_start", ppp)

    # Post val-loss (nhẹ)
    if int(post_loss_k) > 0:
        vlp = ValLossPostLite(k=int(post_loss_k))
        yolo.add_callback("on_val_end", vlp.on_val_end)

    from ultralytics.utils import LOGGER
    LOGGER.info("[Attach] ValTrap+Safety registered (nms=%s, post_loss_k=%s, log_file=%s)", nms, post_loss_k, log_file)

# =============================================================================
# 13) Minimal attach macro (gọn, có tham số)
# =============================================================================
def attach_minimal_callbacks(yolo,
                             *,
                             stn_cfg: Dict[str, Any] | None = None,
                             supcon_schedule: Dict[str, Any] | None = None,
                             debug_images: Dict[str, Any] | None = None,
                             enable_val_loss: bool = True,
                             results_csv_guard: bool = True):
    """Tối giản: STNControl + SupConScheduler + DebugImages + ValLoss + CSV guard."""
    attach_callbacks(
        yolo,
        stn_cfg=stn_cfg or {},
        supcon_schedule=(supcon_schedule or {}).get("schedule_str", None),
        debug_images=debug_images or {},
        enable_val_loss=enable_val_loss,
        results_csv_guard=results_csv_guard,
        publish_theta=True,
        link_trainer_to_loss=True,
        sync_epoch_to_loss=True,
    )

from __future__ import annotations
import os, time, math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFont, ImageDraw, Image
from ultralytics import nn

from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import smart_inference_mode


# ──────────────────────────────────────────────────────────────────────────────
#  Constant labels & colors
# ──────────────────────────────────────────────────────────────────────────────
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
FONT_PATH = "arial.ttf"


# ──────────────────────────────────────────────────────────────────────────────
#  Small helpers
# ──────────────────────────────────────────────────────────────────────────────
def _put_text(img, txt, pos, col, size=22):
    pil = Image.fromarray(img)
    d   = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(FONT_PATH, size)
    except Exception:
        font = ImageFont.load_default()
    d.text(pos, txt, font=font, fill=col)
    return np.asarray(pil).copy()


# ──────────────────────────────────────────────────────────────────────────────
#  1) STN controller: freeze/warmup + trust region + val bypass
# ──────────────────────────────────────────────────────────────────────────────
try:
    from ultralytics.nn.modules.block import SpatialTransformer as STN
except Exception:
    STN = type("STN", (), {})

class STNControl:
    """
    - Freeze STN N epoch đầu (identity).
    - Sau đó bật dần theo warmup (alpha 0→1), với clamp theta (dịch & tỉ lệ).
    - Khi VALIDATION: ép STN chạy identity, sau val khôi phục lại trạng thái trước đó.
    - Patch forward TENSOR-duy-nhất, theta ghi ra qua m.record_theta(...) nếu có.
    """
    def __init__(self, freeze_epochs=0, stn_warmup=20, tmax=0.20, smin=0.85, smax=1.15, log=False):
        self.freeze_epochs = max(0, int(freeze_epochs))
        self.stn_warmup = max(0, int(stn_warmup))
        self.tmax, self.smin, self.smax = float(tmax), float(smin), float(smax)
        self.log = bool(log)
        self._last_alpha = 0.0
        self._last_mode  = "identity"  # "identity" | "blend"

    def _dbg(self, msg: str):
        if self.log:
            LOGGER.info(msg)

    @staticmethod
    def _unwrap(m):
        return STNControl._unwrap(m.module) if hasattr(m, "module") else m

    def _resolve_root(self, obj):
        """Tìm root model bền vững cho cả train/val."""
        cand = []
        m = getattr(obj, "model", None)
        t = getattr(obj, "trainer", None)
        v = getattr(obj, "validator", None)

        cand += [m, getattr(m, "ema", None)]
        cand += [getattr(obj, "ema", None)]
        if t:
            tm = getattr(t, "model", None)
            cand += [tm, getattr(tm, "ema", None)]
        if v:
            vm = getattr(v, "model", None)
            cand += [vm, getattr(vm, "ema", None)]
        if hasattr(obj, "modules"):
            cand.append(obj)

        for c in cand:
            if c is not None:
                return self._unwrap(c)
        return None

    @staticmethod
    def _is_stn(m) -> bool:
        return m.__class__.__name__ in {"SpatialTransformer", "STN", "SpatialTransformer2D", "SpatialTransformerBlock"}

    def _iter_stn(self, model):
        if model is None:
            return
        for m in model.modules():
            if self._is_stn(m):
                yield m

    def _alpha_for_epoch(self, e: int) -> float:
        if e < self.freeze_epochs:
            return 0.0
        if self.stn_warmup <= 0:
            return 1.0
        k = (e - self.freeze_epochs) / float(self.stn_warmup)
        return float(max(0.0, min(1.0, k)))

    def _record_theta(self, m, theta):
        if hasattr(m, "record_theta") and callable(getattr(m, "record_theta")):
            try:
                if isinstance(theta, torch.Tensor):
                    m.record_theta(theta.detach())
                else:
                    m.record_theta(theta)
            except Exception:
                pass

    @staticmethod
    def _ensure_orig_forward(m):
        if not hasattr(m, "_stn_forward_orig"):
            m._stn_forward_orig = m.forward

    @staticmethod
    def _set_patch_flag(m, mode: str | None):
        m._stn_patched_identity = bool(mode == "identity")
        m._stn_patched_blend    = bool(mode == "blend")

    def _patch_identity(self, m):
        self._ensure_orig_forward(m)

        def _forward_identity(x, *_, **__):
            B, _, _, _ = x.shape
            theta_I = torch.tensor([[1, 0, 0], [0, 1, 0]], device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
            self._record_theta(m, theta_I)
            return x

        m.forward = _forward_identity
        self._set_patch_flag(m, "identity")

    def _patch_blend(self, m, alpha: float):
        self._ensure_orig_forward(m)
        alpha = float(max(0.0, min(1.0, alpha)))
        tmax, smin, smax = self.tmax, self.smin, self.smax

        def _forward_blend(x, *args, **kwargs):
            out = m._stn_forward_orig(x, *args, **kwargs)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                x_t, theta = out[0], out[1]
            else:
                x_t, theta = (out if isinstance(out, torch.Tensor) else x), None

            # thiếu theta -> không squash
            if not isinstance(theta, torch.Tensor):
                B = x.shape[0]
                theta_I = torch.tensor([[1, 0, 0], [0, 1, 0]], device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
                self._record_theta(m, theta_I)
                return x if alpha < 1.0 else x_t

            B, C, H, W = x.shape
            T = theta.view(-1, 2, 3).to(dtype=x.dtype)
            M, t = T[:, :, :2], T[:, :, 2]
            t = t.tanh() * tmax  # clamp dịch chuyển

            # clamp scale/shear qua SVD
            try:
                U, S, Vh = torch.linalg.svd(M)
                S = S.clamp(min=smin, max=smax)
                M_safe = U @ torch.diag_embed(S) @ Vh
            except Exception:
                I = torch.eye(2, device=x.device, dtype=x.dtype).unsqueeze(0)
                M_safe = 0.75 * (M - I) + I

            theta_safe = torch.cat([M_safe, t.unsqueeze(-1)], dim=-1)
            grid = torch.nn.functional.affine_grid(theta_safe, size=(B, C, H, W), align_corners=False)
            x_stab = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            y = x_stab if alpha >= 1.0 else (x if alpha <= 0.0 else x + (x_stab - x) * alpha)

            self._record_theta(m, theta_safe)
            return y

        m.forward = _forward_blend
        self._set_patch_flag(m, "blend")

    @staticmethod
    def _restore(m):
        if hasattr(m, "_stn_forward_orig"):
            m.forward = m._stn_forward_orig
        STNControl._set_patch_flag(m, None)

    def _apply_identity(self, model, enable: bool):
        for m in self._iter_stn(model):
            if enable:
                if not getattr(m, "_stn_patched_identity", False):
                    self._patch_identity(m)
            else:
                if getattr(m, "_stn_patched_identity", False):
                    self._restore(m)

    def _apply_blend(self, model, alpha: float):
        for m in self._iter_stn(model):
            self._patch_blend(m, alpha)

    # Ultralytics callbacks
    def on_train_epoch_start(self, trainer):
        model = self._resolve_root(trainer)
        if model is None:
            self._dbg("[STN] on_train_epoch_start: cannot resolve model")
            return

        e = int(getattr(trainer, "epoch", 0))
        if e < self.freeze_epochs:
            self._apply_identity(model, True)
            self._last_mode, self._last_alpha = "identity", 0.0
            self._dbg(f"[STN] frozen -> IDENTITY at epoch {e}")
        else:
            alpha = self._alpha_for_epoch(e)
            self._apply_identity(model, False)
            self._apply_blend(model, alpha)
            self._last_mode, self._last_alpha = "blend", alpha
            self._dbg(f"[STN] ENABLED blend alpha={alpha:.3f} at epoch {e} (tmax={self.tmax}, S∈[{self.smin},{self.smax}])")

    def on_val_start(self, validator):
        model = self._resolve_root(validator)
        if model is None:
            self._dbg("[STN] on_val_start: cannot resolve model")
            return
        self._apply_identity(model, True)
        self._dbg("[STN] validation bypass ON (identity)")

    def on_val_end(self, validator):
        model = self._resolve_root(validator)
        if model is None:
            self._dbg("[STN] on_val_end: cannot resolve model")
            return
        if self._last_mode == "blend":
            self._apply_identity(model, False)
            self._apply_blend(model, self._last_alpha)
            self._dbg(f"[STN] validation bypass OFF -> restore BLEND alpha={self._last_alpha:.3f}")
        else:
            self._dbg("[STN] validation bypass OFF -> keep IDENTITY (frozen)")

class EnsureThetaOnValBatch:
    """
    Bảo hiểm cho nhánh VAL: nếu chưa có 'stn_theta' ở state thì cấp identity (B,2,3).
    Lưu ý: Ultralytics sẽ gọi on_val_batch_start(self, validator) chỉ với 1 tham số.
    """
    def __init__(self, verbose=True):
        self.verbose = verbose

    def on_val_batch_start(self, validator):
        owner = getattr(validator, "trainer", validator)
        if not hasattr(owner, "state"):
            owner.state = {}
        st = owner.state

        # Lấy batch hiện tại từ validator (chuẩn Ultralytics)
        batch = getattr(validator, "batch", None)
        if batch is None:
            return

        # Trích imgs và B
        if isinstance(batch, dict):
            imgs = batch.get("img", None)
        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            imgs = batch[0]
        else:
            imgs = None
        if imgs is None:
            return
        B = imgs.shape[0]

        # Nếu theta không có hoặc sai batch size -> cấp identity đúng B
        theta = st.get("stn_theta", None)
        need_fallback = (
            theta is None
            or not hasattr(theta, "shape")
            or theta.shape[0] != B
            or theta.shape[-2:] != (2, 3)
        )
        if need_fallback:
            theta = imgs.new_zeros((B, 2, 3))
            theta[:, 0, 0] = 1.0
            theta[:, 1, 1] = 1.0
            st["stn_theta"] = theta

        if self.verbose:
            bidx = getattr(validator, "batch_i", -1)
            LOGGER.info(f"[ValTheta] batch={bidx} B={B} fallback_identity={need_fallback}")

class PublishThetaToStateV2:
    """
    Gắn hook record_theta vào mọi SpatialTransformer.
    - train_start: attach vào trainer.model (lưu ở _prev_train)
    - val_start:   attach vào validator.model (lưu ở _prev_val)
    - val_end:     chỉ gỡ hook đã gắn ở VAL (không đụng hook TRAIN)
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        self._prev_train = {}  # {module: prev_record_theta}
        self._prev_val = {}

    @staticmethod
    def _root(obj):
        # Ultralytics: model có thể nằm ở .model hoặc .model.model
        model = getattr(obj, "model", None) or getattr(getattr(obj, "trainer", None), "model", None)
        return getattr(model, "model", model) if model is not None else None

    @staticmethod
    def _iter_stn_modules(root):
        for name, m in root.named_modules():
            if type(m).__name__ in {"SpatialTransformer", "STN", "SpatialTransformerModule"}:
                yield name, m

    def _attach(self, owner, store_dict):
        root = self._root(owner)
        if root is None:
            return 0
        if not hasattr(owner, "state"):
            owner.state = {}
        cnt = 0
        for name, m in self._iter_stn_modules(root):
            if m in store_dict:
                continue
            prev = getattr(m, "record_theta", None)
            store_dict[m] = prev

            def _rec(theta, _prev=prev, _owner=owner):
                try:
                    if callable(_prev):
                        _prev(theta)  # chain cũ
                except Exception:
                    pass
                try:
                    _owner.state["stn_theta"] = theta.detach() if hasattr(theta, "detach") else theta
                except Exception:
                    _owner.state["stn_theta"] = theta

            setattr(m, "record_theta", _rec)
            cnt += 1
        return cnt

    def _detach(self, store_dict):
        restored = 0
        for m, prev in list(store_dict.items()):
            try:
                if prev is not None:
                    setattr(m, "record_theta", prev)
                elif hasattr(m, "record_theta"):
                    delattr(m, "record_theta")
            except Exception:
                pass
            restored += 1
        store_dict.clear()
        return restored

    # === callbacks ===
    def on_train_start(self, trainer):
        n = self._attach(trainer, self._prev_train)
        if self.verbose:
            LOGGER.info(f"[ThetaPub] train_start: attached to {n} STN module(s)")

    def on_val_start(self, validator):
        n = self._attach(validator, self._prev_val)
        if self.verbose:
            LOGGER.info(f"[ThetaPub] val_start: attached to {n} STN module(s)")

    def on_val_end(self, validator):
        r = self._detach(self._prev_val)
        if self.verbose:
            LOGGER.info(f"[ThetaPub] val_end: restored {r} STN module(s)")

    def on_train_end(self, trainer):
        r = self._detach(self._prev_train)
        if self.verbose:
            LOGGER.info(f"[ThetaPub] train_end: restored {r} STN module(s)")


class DumpModelWiringOnVal:
    """
    In ra đường đi ở VAL: tìm các module STN & DetectSDTN, log tên-đường dẫn đầy đủ.
    Không sửa gì vào mô hình.
    """
    def __init__(self, verbose=True, max_lines=64):
        self.verbose = verbose
        self.max_lines = max_lines
        self._paths = {"stn": [], "head": []}

    @staticmethod
    def _root(obj):
        model = getattr(obj, "model", None) or getattr(getattr(obj, "trainer", None), "model", None)
        return getattr(model, "model", model) if model is not None else None

    def on_val_start(self, validator):
        root = self._root(validator)
        self._paths = {"stn": [], "head": []}
        if root is None:
            LOGGER.warning("[WireDump] root model is None")
            return
        for name, m in root.named_modules():
            t = type(m).__name__
            if t in {"SpatialTransformer", "STN", "SpatialTransformerModule"}:
                self._paths["stn"].append((name, t))
            if ("DetectSDTN" in t) or ("DetectSDTN" in name):
                self._paths["head"].append((name, t))

        if self.verbose:
            LOGGER.info(f"[WireDump] VAL model: found STN={len(self._paths['stn'])}, HEAD={len(self._paths['head'])}")
            for tag in ("stn", "head"):
                lines = self._paths[tag][: self.max_lines]
                for i, (name, t) in enumerate(lines):
                    LOGGER.info(f"[WireDump] {tag.upper()}[{i}] path='{name}' type='{t}'")


class WireThetaToSDTNHead:
    """
    Mỗi batch VAL: nếu có 'stn_theta' trong state thì thử bơm theta vào head DetectSDTN.
    Không đoán mò API: lần lượt thử các cách quen thuộc; nếu không có API tương thích -> chỉ log.
    Không tạo theta fallback (không ép).
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        self._heads = []

    @staticmethod
    def _root(obj):
        model = getattr(obj, "model", None) or getattr(getattr(obj, "trainer", None), "model", None)
        return getattr(model, "model", model) if model is not None else None

    def on_val_start(self, validator):
        model = self._resolve_root(validator)
        if model is None:
            self._dbg("[STN] on_val_start: cannot resolve model")
            return
        # mimic TRAIN during VAL: enable stabilized blend with same alpha schedule
        e = int(getattr(getattr(validator, "trainer", validator), "epoch", 0))
        alpha = self._alpha_for_epoch(e)
        self._apply_identity(model, False)
        self._apply_blend(model, alpha)
        self._last_mode, self._last_alpha = "blend", alpha
        self._dbg(f"[STN] validation mimic TRAIN -> blend alpha={alpha:.3f} at epoch {e}")

    def on_val_batch_start(self, validator):
        owner = getattr(validator, "trainer", validator)
        theta = getattr(owner, "state", {}).get("stn_theta", None)
        if theta is None:
            if self.verbose:
                bi = getattr(validator, "batch_i", -1)
                LOGGER.warning(f"[WireTheta] batch={bi}: NO theta in state -> skip wiring")
            return

        wired = 0
        for name, m in self._heads:
            ok = False
            # thứ tự các API quen thuộc
            for api in ("set_theta", "receive_theta", "update_theta", "register_theta"):
                fn = getattr(m, api, None)
                if callable(fn):
                    try:
                        fn(theta)
                        ok = True
                        if self.verbose:
                            LOGGER.info(f"[WireTheta] -> {name}.{api}(B={theta.shape[0]})")
                        break
                    except Exception as e:
                        LOGGER.warning(f"[WireTheta] {name}.{api} failed: {e}")

            if not ok:
                # thử set attribute phổ biến
                for attr in ("theta", "theta_buffer", "_theta", "stn_theta"):
                    try:
                        setattr(m, attr, theta)
                        ok = True
                        if self.verbose:
                            LOGGER.info(f"[WireTheta] -> {name}.{attr} = theta(B={theta.shape[0]})")
                        break
                    except Exception as e:
                        LOGGER.warning(f"[WireTheta] set {name}.{attr} failed: {e}")

            wired += int(ok)

        if self.verbose:
            bi = getattr(validator, "batch_i", -1)
            LOGGER.info(f"[WireTheta] batch={bi}: wired {wired}/{len(self._heads)} head(s)")

# Giữ ForceSTNIdentityOnVal (gọn, ít log) để tương thích code hiện có
class ForceSTNIdentityOnVal:
    def __init__(self, log=False):
        self.ctrl = STNControl(log=log)

    def on_val_start(self, validator):
        self.ctrl.on_val_start(validator)

    def on_val_end(self, validator):
        self.ctrl.on_val_end(validator)


# ──────────────────────────────────────────────────────────────────────────────
#  2) Debug ảnh STN
# ──────────────────────────────────────────────────────────────────────────────
class DebugImages:
    def __init__(self, epochs=(0, 5, 10, 15, 20), max_images=5):
        self.epochs = set(int(e) for e in epochs)
        self.max_images = int(max_images)
        self.samples = []
        self.dbg_dir = None

    @staticmethod
    def _grab_loader(trainer):
        return getattr(getattr(trainer, "validator", None), "dataloader", None) or trainer.train_loader

    def _cache_samples(self, trainer):
        loader = self._grab_loader(trainer)
        for batch in loader:
            imgs = batch["img"] if isinstance(batch, dict) else batch[0]
            tgts = batch if isinstance(batch, dict) else batch[1]
            idx0 = (tgts["batch_idx"] == 0)
            self.samples.append((
                imgs[0].cpu(),
                tgts["bboxes"][idx0].cpu(),
                tgts["cls"][idx0].cpu(),
                (tgts.get("im_file") or [None])[0]
            ))
            if len(self.samples) >= self.max_images:
                break

    @staticmethod
    def _scale_xyxy(xyxy_t: np.ndarray, rx: float, ry: float) -> np.ndarray:
        out = xyxy_t.astype(np.float32).copy()
        out[:, [0, 2]] *= float(rx)
        out[:, [1, 3]] *= float(ry)
        return out

    @staticmethod
    def _warp_boxes_xywh_with_theta(bxywh: torch.Tensor, W: int, H: int, theta_2x3: torch.Tensor) -> torch.Tensor:
        if theta_2x3 is None:
            x, y, w, h = bxywh.unbind(-1)
            return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=1)

        device, dtype = bxywh.device, bxywh.dtype
        x, y, w, h = bxywh.unbind(-1)
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2

        Xs = torch.stack([x1, x2, x2, x1], dim=1)
        Ys = torch.stack([y1, y1, y2, y2], dim=1)

        pix2norm = lambda px, L: (px / max(L - 1, 1) * 2.0) - 1.0
        Xn, Yn = pix2norm(Xs, W), pix2norm(Ys, H)

        th = theta_2x3.detach().to(dtype=torch.float32, device=device)
        A  = torch.tensor([[th[0, 0], th[0, 1], th[0, 2]],
                           [th[1, 0], th[1, 1], th[1, 2]],
                           [0.0,      0.0,      1.0     ]], dtype=torch.float32, device=device)
        Ainv = torch.linalg.inv(A)

        ones  = torch.ones_like(Xn)
        P_in  = torch.stack([Xn, Yn, ones], dim=1)         # (N,3,4)
        P_out = torch.einsum("ij,njk->nik", Ainv, P_in)    # (N,3,4)
        Xo, Yo = P_out[:, 0, :], P_out[:, 1, :]

        norm2pix = lambda pn, L: (pn + 1.0) * 0.5 * (L - 1)
        Xp, Yp = norm2pix(Xo, W), norm2pix(Yo, H)

        x1p, x2p = Xp.min(dim=1).values, Xp.max(dim=1).values
        y1p, y2p = Yp.min(dim=1).values, Yp.max(dim=1).values
        x1p = x1p.clamp(0, W - 1);  y1p = y1p.clamp(0, H - 1)
        x2p = x2p.clamp(1, W);      y2p = y2p.clamp(1, H)

        return torch.stack([x1p, y1p, x2p, y2p], dim=1)

    @smart_inference_mode()
    def on_train_epoch_end(self, trainer):
        ep = int(getattr(trainer, "epoch", 0))
        if ep not in self.epochs:
            return

        if self.dbg_dir is None:
            self.dbg_dir = os.path.join(str(trainer.save_dir), "stn_dbg")
            os.makedirs(self.dbg_dir, exist_ok=True)
        if not self.samples:
            self._cache_samples(trainer)

        model = trainer.model
        device = next(model.parameters()).device
        was_train = model.training
        model.eval()

        H_panel = 672
        pad_color = (40, 40, 40)

        try:
            for i, (img0, bxywh, bcls, path0) in enumerate(self.samples):
                x = img0.clone().to(device).unsqueeze(0).float()
                if x.max() > 1:
                    x = x / 255.0
                if hasattr(trainer, "state"):
                    trainer.state.pop("stn_theta", None)
                    trainer.state.pop("stn_out", None)
                _ = model(x)
                theta = getattr(trainer, "state", {}).get("stn_theta", None)
                stn_img = getattr(trainer, "state", {}).get("stn_out", None)

                ori_np = img0.detach().cpu().numpy()
                if ori_np.ndim == 3 and ori_np.shape[0] in (1, 3):
                    ori_np = np.transpose(ori_np, (1, 2, 0))
                if ori_np.max() <= 1.5:
                    ori_np = (ori_np * 255.0).round()
                ori_np = np.clip(ori_np, 0, 255).astype(np.uint8)
                if ori_np.ndim == 2:
                    ori_np = np.repeat(ori_np[..., None], 3, axis=2)
                if ori_np.shape[2] == 1:
                    ori_np = np.repeat(ori_np, 3, axis=2)
                ori = ori_np

                H_src, W_src = ori.shape[:2]

                if torch.is_tensor(stn_img):
                    vis = stn_img[0].detach().cpu().numpy()
                    if vis.ndim == 3 and vis.shape[0] in (1, 3):
                        vis = np.transpose(vis, (1, 2, 0))
                    vis = (vis * 255.0).clip(0, 255).astype(np.uint8)
                    if vis.ndim == 2:
                        vis = np.repeat(vis[..., None], 3, axis=2)
                    if vis.shape[2] == 1:
                        vis = np.repeat(vis, 3, axis=2)
                else:
                    vis = ori.copy()

                def _resize_h(im, H=H_panel):
                    h, w = im.shape[:2]
                    if h == H:
                        return im
                    new_w = int(round(w * H / float(h)))
                    return cv2.resize(im, (new_w, H), interpolation=cv2.INTER_LINEAR)

                L = _resize_h(ori, H_panel)
                R = _resize_h(vis, H_panel)
                H0, W_L = L.shape[:2]
                _,  W_R = R.shape[:2]

                bxywh_src = (bxywh.detach().cpu().numpy() *
                             np.array([W_src, H_src, W_src, H_src], dtype=np.float32))

                xyxy_src = xywh2xyxy(torch.from_numpy(bxywh_src)).numpy().astype(np.float32)
                rx_L = float(W_L) / float(W_src)
                ry_L = float(H_panel) / float(H_src)
                xyxy_left = self._scale_xyxy(xyxy_src, rx_L, ry_L).round().clip(0, 10**9).astype(int)

                xyxy_right = xyxy_left.copy()
                if isinstance(theta, torch.Tensor):
                    th = theta[0] if (theta.dim() == 3 and theta.shape[0] >= 1) else (theta if theta.dim() == 2 else None)
                    if th is not None:
                        with torch.no_grad():
                            tb = torch.from_numpy(bxywh_src).to(dtype=torch.float32)
                            tbw_src = self._warp_boxes_xywh_with_theta(tb, W_src, H_src, th)
                            rx_R = float(W_R) / float(W_src)
                            ry_R = float(H_panel) / float(H_src)
                            xyxy_right = self._scale_xyxy(tbw_src.cpu().numpy().astype(np.float32), rx_R, ry_R)\
                                             .round().clip(0, 10**9).astype(int)

                def _draw(img, xyxy_use):
                    out = img.copy()
                    labels = bcls.view(-1).tolist()
                    for (x1, y1, x2, y2), c in zip(xyxy_use, labels):
                        x1 = int(max(0, min(img.shape[1] - 1, x1)))
                        y1 = int(max(0, min(img.shape[0] - 1, y1)))
                        x2 = int(max(1, min(img.shape[1],     x2)))
                        y2 = int(max(1, min(img.shape[0],     y2)))
                        col = CLASS_COLORS[int(c) % len(CLASS_COLORS)]
                        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
                        out = _put_text(out, CLASS_LABELS.get(int(c), str(int(c))), (x1 + 4, max(0, y1 - 22)), col, 28)
                    return out

                Ld = _draw(L, xyxy_left)
                Rd = _draw(R, xyxy_right)

                Ld = _put_text(Ld, "ORIGINAL", (10, 10), (255, 255, 0), 42)
                Rd = _put_text(Rd, "STN",      (10, 10), (255, 255, 0), 42)

                if not isinstance(theta, torch.Tensor) or theta.numel() == 0:
                    theta_show = torch.tensor([[1, 0, 0],[0, 1, 0]], dtype=torch.float32)
                else:
                    theta_show = theta[0] if theta.dim() == 3 else theta

                t = theta_show.detach().cpu().view(2, 3).numpy() if isinstance(theta_show, torch.Tensor) \
                    else np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]], dtype=np.float32)
                Rd = _put_text(Rd, f"θ[0]: {t[0,0]:+0.3f} {t[0,1]:+0.3f} {t[0,2]:+0.3f}", (10, 60), (120, 255, 120), 30)
                Rd = _put_text(Rd, f"θ[1]: {t[1,0]:+0.3f} {t[1,1]:+0.3f} {t[1,2]:+0.3f}", (10, 95), (120, 255, 120), 30)

                both = np.concatenate([Ld, Rd], axis=1)
                pad = 16
                Hh, Ww = both.shape[:2]
                canvas = np.full((Hh + 2 * pad, Ww + 2 * pad, 3), (40, 40, 40), dtype=np.uint8)
                canvas[pad:pad + Hh, pad:pad + Ww] = both

                base = os.path.splitext(os.path.basename(str(path0) or f"sample{i:02d}"))[0]
                out_path = os.path.join(self.dbg_dir, f"{base}_epoch_{ep:03d}_{i:02d}.png")
                cv2.imwrite(out_path, canvas)
        finally:
            model.train(was_train)

        LOGGER.info(f"[DebugImages] saved samples for epoch {ep}")


# ──────────────────────────────────────────────────────────────────────────────
#  3) PeekBatch: kiểm DataLoader nhanh
# ──────────────────────────────────────────────────────────────────────────────
class PeekBatch:
    def __init__(self, save_first_n=1, max_images=4, panels=2, save_dir_name="peek_batch"):
        self.save_first_n = save_first_n
        self.max_images = max_images
        self.panels = panels
        self.save_dir_name = save_dir_name
        self._done = 0

    def _get_names(self, trainer):
        names = getattr(trainer, "names", None)
        if names is None:
            names = getattr(getattr(trainer, "validator", None), "names", None)
        if names is None:
            names = getattr(getattr(trainer, "model", None), "names", None)
        return names

    def _summarize_value(self, k, v):
        try:
            if torch.is_tensor(v):
                v = v.detach()
                return (f"{k}: tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device} "
                        f"min={float(v.min()) if v.numel() else 'n/a'} max={float(v.max()) if v.numel() else 'n/a'}")
            if isinstance(v, (list, tuple)):
                return f"{k}: {type(v).__name__} len={len(v)} sample_head={list(v)[:3]}"
            if isinstance(v, dict):
                return f"{k}: dict keys={list(v.keys())[:10]}"
            return f"{k}: {type(v).__name__} value={v}"
        except Exception as e:
            return f"{k}: <error summarizing: {e}>"

    def _pack_like_loader(self, items):
        imgs = torch.stack([torch.as_tensor(it["img"]) for it in items], 0)
        bxs, cls, bidx = [], [], []
        for i, it in enumerate(items):
            bi = it.get("bboxes", None)
            ci = it.get("cls", None)
            if isinstance(bi, torch.Tensor) and bi.ndim == 2 and bi.size(-1) == 4:
                bxs.append(bi); bidx.append(torch.full((bi.shape[0],), i))
            if isinstance(ci, torch.Tensor):
                cls.append(ci)
        bxs  = torch.cat(bxs, 0)  if len(bxs)  else torch.zeros((0, 4))
        cls  = torch.cat(cls, 0)  if len(cls)  else torch.zeros((0,))
        bidx = torch.cat(bidx, 0) if len(bidx) else torch.zeros((0,))
        paths = [it.get("im_file", it.get("path", "")) for it in items]
        return {"img": imgs, "bboxes": bxs, "cls": cls, "batch_idx": bidx, "im_file": paths}

    def _save_grid_panel(self, trainer, batch, img_ids, save_path: Path):
        imgs = batch["img"].detach().float().cpu()
        if imgs.max() > 1.5:
            imgs = imgs / 255.0
        sel = torch.tensor(img_ids, dtype=torch.long)
        imgs_sub = imgs.index_select(0, sel)

        labels = None
        if all(k in batch for k in ("batch_idx", "cls", "bboxes")):
            bidx = batch["batch_idx"].long().cpu()
            mask = torch.isin(bidx, sel)
            if mask.any():
                old = bidx[mask]
                mapping = {int(v): i for i, v in enumerate(img_ids)}
                new_bidx = torch.tensor([mapping[int(v)] for v in old], dtype=torch.float32).unsqueeze(1)
                cls = batch["cls"][mask].float().unsqueeze(1).floor()
                bxs = batch["bboxes"][mask].float()
                labels = torch.cat((new_bidx, cls, bxs), 1).cpu()

        paths = batch.get("im_file", batch.get("paths", None))
        if isinstance(paths, (list, tuple)) and len(paths) == imgs.shape[0]:
            paths = [paths[i] for i in img_ids]
        names = self._get_names(trainer)

        plot_images(images=imgs_sub, labels=labels, paths=paths, names=names,
                    fname=str(save_path), max_subplots=len(img_ids))

    def on_fit_epoch_start(self, trainer):
        if self._done >= self.save_first_n:
            return
        self._done += 1

        ds = getattr(getattr(trainer, "train_loader", None), "dataset", None)
        if ds is None:
            return

        bs = int(getattr(trainer, "batch_size", self.max_images * self.panels))
        need = min(bs, self.max_images * self.panels)

        try:
            idxs = np.random.choice(len(ds), need, replace=False).tolist()
        except Exception:
            idxs = list(range(min(need, len(ds))))

        items = [ds[i] for i in idxs]
        if hasattr(ds, "collate_fn"):
            try:
                batch = ds.collate_fn(items)
            except Exception:
                batch = self._pack_like_loader(items)
        else:
            batch = self._pack_like_loader(items)

        save_dir = Path(str(trainer.save_dir)) / self.save_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)

        meta_lines = [f"[peek] epoch={trainer.epoch} imgsz={getattr(trainer, 'imgsz', None)} "
                      f"(synthetic batch, bs~{bs}, dumped={need})"]
        for k in sorted(batch.keys()):
            meta_lines.append(self._summarize_value(k, batch[k]))
        with open(save_dir / f"batch_epoch{int(trainer.epoch):03d}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(meta_lines))

        nimg = int(batch["img"].shape[0])
        take = min(nimg, self.max_images * self.panels)
        cursor, panel_id = 0, 0
        while cursor < take and panel_id < self.panels:
            end = min(cursor + self.max_images, take)
            img_ids = list(range(cursor, end))
            out = save_dir / f"batch_epoch{int(trainer.epoch):03d}_p{panel_id}.jpg"
            self._save_grid_panel(trainer, batch, img_ids, out)
            cursor = end
            panel_id += 1


# ──────────────────────────────────────────────────────────────────────────────
#  4) SupCon hooks (inject + schedule + sync)
# ──────────────────────────────────────────────────────────────────────────────
class InjectSupConArgsMinimal:
    """Cache supcon_* sớm, mirror sang model.args & criterion.hyp khi model/loss sẵn sàng."""
    def __init__(self, **cfg):
        self.cfg = dict(cfg)

    def _ensure_ns(self, obj, attr):
        val = getattr(obj, attr, None)
        if val is None or isinstance(val, dict):
            ns = SimpleNamespace(**(val or {}))
            setattr(obj, attr, ns)
            return ns
        return val

    def on_pretrain_routine_start(self, trainer):
        trainer._supcon_cfg = SimpleNamespace(**self.cfg)
        LOGGER.info("[INJECT@start] cached supcon_* -> trainer._supcon_cfg")

    def on_pretrain_routine_end(self, trainer):
        ma = self._ensure_ns(trainer.model, "args")
        for k, v in self.cfg.items():
            setattr(ma, k, v)
        crit = getattr(trainer.model, "criterion", None)
        if crit is not None:
            hyp = self._ensure_ns(crit, "hyp")
            for k, v in self.cfg.items():
                setattr(hyp, k, v)
        trainer._supcon_cfg = SimpleNamespace(**self.cfg)
        LOGGER.info(f"[INJECT@end] model.args.supcon_feat={getattr(ma, 'supcon_feat', None)} | on={getattr(ma, 'supcon_on', None)}")

    def on_train_start(self, trainer):
        cache = getattr(trainer, "_supcon_cfg", None)

        def _is_default(c):
            try:
                return (
                    getattr(c, "supcon_feat", None) in (None, "") and
                    int(getattr(c, "supcon_on", 0)) == 0 and
                    int(getattr(c, "supcon_proj_dim", 0)) == 0 and
                    int(getattr(c, "supcon_proj_hidden", 0)) == 0 and
                    str(getattr(c, "supcon_schedule", "")) == ""
                )
            except Exception:
                return True

        use_cache = cache is not None and not _is_default(cache)
        cfg = vars(cache) if use_cache else self.cfg

        ma = self._ensure_ns(trainer.model, "args")
        for k, v in cfg.items():
            setattr(ma, k, v)

        crit = getattr(trainer.model, "criterion", None)
        if crit is not None:
            hyp = self._ensure_ns(crit, "hyp")
            for k, v in cfg.items():
                setattr(hyp, k, v)

        LOGGER.info(f"[INJECT@train] source={'cache' if use_cache else 'self.cfg'} | model.args.supcon_feat={getattr(ma, 'supcon_feat', None)} | on={getattr(ma, 'supcon_on', None)}")


class SupConScheduler:
    """Bật/tắt SupCon theo epoch display (1-based)."""
    def __init__(self, schedule_str: str, default_on: int = 0):
        self.default_on=1 if default_on else 0
        self.ranges=self._parse(schedule_str or "")

    @staticmethod
    def _parse(spec: str):
        out=[]
        for tok in spec.replace(" ","").split(","):
            if not tok: continue
            if "-" in tok:
                a,b=tok.split("-",1); a=a.strip(); b=b.strip()
                if a=="": lo,hi=1,int(b)
                elif b=="": lo,hi=int(a),10**9
                else: lo,hi=int(a),int(b)
            else: lo=hi=int(tok)
            out.append((lo,hi))
        return out

    def _enabled_for_display_epoch(self, e_disp:int)->int:
        if not self.ranges: return self.default_on
        for lo,hi in self.ranges:
            if lo<=e_disp<=hi: return 1
        return 0

    def on_train_epoch_start(self,trainer):
        e_disp=int(getattr(trainer,"epoch",0))+1
        want_on=self._enabled_for_display_epoch(e_disp)
        ma=getattr(trainer.model,"args",None)
        if ma is None or isinstance(ma,dict):
            ma=SimpleNamespace(**(ma or {})); trainer.model.args=ma
        setattr(ma,"supcon_on",int(want_on))
        if want_on==1 and hasattr(trainer,"loss_names"):
            if not trainer.loss_names or trainer.loss_names[-1]!="supcon_loss":
                trainer.loss_names=("box_loss","cls_loss","dfl_loss","supcon_loss")
        LOGGER.info(f"[SupCon/schedule] epoch_display={e_disp} -> supcon_on={want_on}")


class ReinforceSupConToLoss:
    """Mirror lại supcon_* từ model.args sang criterion.hyp mỗi epoch (phòng reset)."""
    def __init__(self, keys): self.keys=tuple(keys)
    def on_train_epoch_start(self, trainer):
        ma=getattr(trainer.model,"args",None)
        crit=getattr(trainer.model,"criterion",None)
        if ma is None or crit is None: return
        if not hasattr(crit,"hyp") or crit.hyp is None: crit.hyp=SimpleNamespace()
        for k in self.keys:
            if hasattr(ma,k): setattr(crit.hyp,k,getattr(ma,k))
        try:
            sc_on=int(getattr(ma,"supcon_on",0) or 0)
            if sc_on==1 and hasattr(trainer,"loss_names") and trainer.loss_names[-1]!="supcon_loss":
                trainer.loss_names=("box_loss","cls_loss","dfl_loss","supcon_loss")
        except Exception:
            pass


class LinkTrainerToLoss:
    def on_train_start(self, trainer):
        if getattr(trainer,"loss",None) is not None:
            trainer.loss._trainer=trainer


class SyncEpochToLoss:
    def on_train_epoch_start(self, trainer):
        if getattr(trainer,"loss",None) is not None:
            trainer.loss.epoch=int(trainer.epoch)
    def on_train_batch_start(self, trainer):
        if getattr(trainer,"loss",None) is not None:
            trainer.loss.epoch=int(trainer.epoch)


# ──────────────────────────────────────────────────────────────────────────────
#  5) Tap STN feature (GAP) – VRAM-friendly
# ──────────────────────────────────────────────────────────────────────────────
class TapSTNFeat:
    """Hook lấy đặc trưng GAP [B,C] tại layer được chọn, cắt tham chiếu sau mỗi batch để giảm VRAM."""
    def __init__(self, out_idx: int | None = None, out_name: str | None = None):
        self.out_idx = out_idx
        self.out_name = (out_name or None)
        self.hook_handle = None
        self.latest = None
        self.attached_to = None

    def _cfg_get(self, trainer, key: str, default=None):
        cfg = getattr(trainer, "_supcon_cfg", None)
        if cfg is None: return default
        if isinstance(cfg, dict): return cfg.get(key, default)
        if isinstance(cfg, SimpleNamespace): return getattr(cfg, key, default)
        return getattr(cfg, key, default)

    def _hook(self, module, inputs, output):
        gap = F.adaptive_avg_pool2d(output, (1, 1)).flatten(1)
        self.latest = gap

    def _resolve_module(self, model, cfg_idx: int | None, cfg_name: str | None):
        modlist = getattr(model, "model", None)
        if modlist is None: return None, None
        if isinstance(cfg_idx, int) and 0 <= cfg_idx < len(modlist):
            return modlist[cfg_idx], f"model.model[{cfg_idx}]"
        if cfg_name:
            low = str(cfg_name).lower()
            for i, m in enumerate(modlist):
                if low in m.__class__.__name__.lower():
                    return m, f"model.model[{i}]<{m.__class__.__name__}>"
        for i, m in enumerate(modlist):
            if hasattr(m, "forward"):
                return m, f"model.model[{i}]<{m.__class__.__name__}>"
        return None, None

    def on_train_start(self, trainer, *args, **kwargs):
        idx = self._cfg_get(trainer, "supcon_out", self.out_idx)
        name = self._cfg_get(trainer, "supcon_name", self.out_name)
        name = name.strip() or None if isinstance(name, str) else None

        module, where = self._resolve_module(trainer.model, idx, name)
        if module is None:
            LOGGER.info("[TapSTNFeat] could not attach (layer not found)")
            return

        if self.hook_handle is not None:
            try: self.hook_handle.remove()
            except Exception: pass
            self.hook_handle = None

        self.hook_handle = module.register_forward_hook(self._hook)
        self.attached_to = where
        LOGGER.info(f"[TapSTNFeat] hook attached at {where}")

    def on_train_end(self, trainer, *args, **kwargs):
        if self.hook_handle is not None:
            try: self.hook_handle.remove()
            except Exception: pass
            self.hook_handle = None
        self.latest = None
        self.attached_to = None

    def on_train_batch_end(self, trainer, *args, **kwargs):
        self.latest = None


# ──────────────────────────────────────────────────────────────────────────────
#  6) (Optional) t-SNE embedding – giữ nguyên cho ai còn dùng
# ──────────────────────────────────────────────────────────────────────────────
class TSNEEmbeddings:
    """
    Lấy feature map sâu sau STN, pool ROI, chuẩn hoá, t-SNE metric 'cosine', vẽ PNG ở ep 0,10,20,...
    Dùng khi bạn vẫn cần bản t-SNE đầy đủ hơn SafeTSNE.
    """
    def __init__(self, every=10, loader="val", per_class=150, total_max=10000, max_batches=250,
                 use_roialign=True, roialign_out=1, min_feat_wh=1, pca_dim=128,
                 min_channels=128, min_downsample=4, verbose=True):
        self.every = 10 if int(every) in (0, 1) else max(1, int(every))
        self.loader = loader
        self.per_class = int(per_class)
        self.total_max = int(total_max)
        self.max_batches = int(max_batches)
        self.use_roialign = bool(use_roialign)
        self.roialign_out = int(roialign_out)
        self.min_feat_wh = int(min_feat_wh)
        self.pca_dim = int(pca_dim)
        self.min_channels = int(min_channels)
        self.min_downsample = int(min_downsample)
        self.verbose = bool(verbose)
        self._feat_mod = None

    def _get_loader(self, trainer):
        if self.loader == "train":
            return trainer.train_loader
        val = getattr(getattr(trainer, "validator", None), "dataloader", None)
        return val or trainer.train_loader

    def _choose_after_stn_conv(self, trainer, device):
        if self._feat_mod is not None:
            return self._feat_mod

        core = trainer.model
        seq = list(core.modules())
        stn_seen, hooks, outs = False, [], []

        def _mk_hook(m):
            def _h(_m, _i, o):
                if torch.is_tensor(o):
                    outs.append((m, o))
                elif isinstance(o, (list, tuple)):
                    for t in o:
                        if torch.is_tensor(t):
                            outs.append((m, t))
            return _h

        for m in seq:
            if isinstance(m, STN):
                stn_seen = True
                continue
            if not stn_seen:
                continue
            if hasattr(m, "forward"):
                hooks.append(m.register_forward_hook(_mk_hook(m)))

        loader = self._get_loader(trainer)
        batch = next(iter(loader))
        img = (batch["img"] if isinstance(batch, dict) else batch[0]).to(device).float()
        if img.max() > 1:
            img = img / 255.0
        B, C, H, W = img.shape

        was_train = core.training
        core.eval()
        try:
            with torch.no_grad():
                _ = core(img[:1])
        finally:
            for h in hooks: h.remove()
            core.train(was_train)

        candidates = []
        for m, o in outs:
            if not (torch.is_tensor(o) and o.dim() == 4):
                continue
            _, Cf, Hf, Wf = o.shape
            deep_enough = (Cf >= self.min_channels) and (Hf <= H//self.min_downsample) and (Wf <= W//self.min_downsample)
            if deep_enough:
                candidates.append((m, (1, Cf, Hf, Wf)))

        if not candidates and outs:
            for m, o in reversed(outs):
                if torch.is_tensor(o) and o.dim() == 4:
                    _, Cf, Hf, Wf = o.shape
                    candidates.append((m, (1, Cf, Hf, Wf)))
                    break

        if not candidates:
            LOGGER.warning("[TSNE] can't find feature after STN; skip")
            return None

        self._feat_mod = candidates[0][0]
        if self.verbose:
            LOGGER.info(f"[TSNE] feature module = {self._feat_mod.__class__.__name__} with shape {candidates[0][1]}")
        return self._feat_mod

    @torch.no_grad()
    def on_train_epoch_end(self, trainer):
        ep = int(getattr(trainer, "epoch", 0))
        if (ep % self.every) != 0:
            return

        dl = self._get_loader(trainer)
        if dl is None:
            LOGGER.warning("[TSNE] dataloader not found; skip")
            return

        model = trainer.model
        device = next(model.parameters()).device
        save_dir = os.path.join(str(trainer.save_dir), "embeddings")
        os.makedirs(save_dir, exist_ok=True)
        png_path = os.path.join(save_dir, f"tsne_epoch_{ep:03d}.png")

        feat_mod = self._choose_after_stn_conv(trainer, device)
        if feat_mod is None:
            return

        bucket = []
        handle = feat_mod.register_forward_hook(lambda m, i, o: bucket.append(o))

        was_train = model.training
        model.eval()

        from collections import defaultdict
        per_cls = defaultdict(int)
        feats, labels = [], []
        stats = dict(collected=0, skip_no_feat=0, skip_not4d=0,
                     skip_empty_targets=0, skip_too_small=0,
                     skip_quota_class=0, stop_total_max=0, stop_max_batches=0)

        try:
            for bi, batch in enumerate(dl, start=1):
                if self.max_batches and bi > self.max_batches:
                    stats["stop_max_batches"] += 1
                    break

                img = (batch["img"] if isinstance(batch, dict) else batch[0]).to(device).float()
                if img.max() > 1:
                    img = img / 255.0
                B, C, H, W = img.shape

                tg = batch if isinstance(batch, dict) else batch[1]
                if tg["bboxes"].numel() == 0:
                    stats["skip_empty_targets"] += 1
                    continue

                bucket.clear()
                _ = model(img)

                if not bucket:
                    stats["skip_no_feat"] += 1
                    continue
                fmap = bucket[-1]
                if not torch.is_tensor(fmap) or fmap.dim() != 4:
                    stats["skip_not4d"] += 1
                    continue

                _, Cf, Hf, Wf = fmap.shape
                bidx  = tg["batch_idx"].to(torch.long)
                bxywh = tg["bboxes"]
                bcls  = tg["cls"].view(-1).to(torch.long)

                xywh_pix = bxywh * torch.tensor([W, H, W, H], device=bxywh.device)
                x, y, w, h = xywh_pix.unbind(-1)
                x1 = (x - w/2).clamp(0, W-1);  y1 = (y - h/2).clamp(0, H-1)
                x2 = (x + w/2).clamp(1, W);    y2 = (y + h/2).clamp(1, H)

                sx, sy = Wf / float(W), Hf / float(H)
                fx1 = (x1 * sx).floor().clamp(0, Wf-1).to(torch.long)
                fy1 = (y1 * sy).floor().clamp(0, Hf-1).to(torch.long)
                fx2 = (x2 * sx).ceil().clamp(1,  Wf).to(torch.long)
                fy2 = (y2 * sy).ceil().clamp(1,  Hf).to(torch.long)

                if self.use_roialign:
                    from torchvision.ops import roi_align
                    rois = torch.stack([bidx.float(), fx1.float(), fy1.float(), fx2.float(), fy2.float()], dim=1).to(fmap.device)
                    pooled = roi_align(fmap, rois, output_size=(self.roialign_out, self.roialign_out),
                                       spatial_scale=1.0, sampling_ratio=-1, aligned=True)
                    vecs = pooled.mean(dim=(2, 3))
                    proj = getattr(getattr(trainer.model, "criterion", None), "supcon_proj", None)
                    if proj is not None:
                        in_dim = getattr(proj, "in_dim", None)
                        if in_dim is None:
                            first_linear = None
                            try:
                                first_linear = next((m for m in proj.net if isinstance(m, nn.Linear)), None)
                            except Exception:
                                pass
                            if first_linear is not None:
                                in_dim = first_linear.in_features
                        if in_dim is not None and vecs.shape[-1] == in_dim:
                            with torch.cuda.amp.autocast(enabled=False):
                                vecs = proj(vecs.float())
                        else:
                            LOGGER.warning(f"[TSNE] Skip SupConProj: dim mismatch vecs={vecs.shape[-1]} vs proj_in={in_dim}")

                    for vi, c in zip(vecs, bcls):
                        if self.per_class and per_cls[int(c)] >= self.per_class:
                            stats["skip_quota_class"] += 1; continue
                        feats.append(vi.cpu().numpy()); labels.append(int(c))
                        per_cls[int(c)] += 1; stats["collected"] += 1
                else:
                    for bi2, c, xi1, yi1, xi2, yi2 in zip(bidx, bcls, fx1, fy1, fx2, fy2):
                        xi1, yi1, xi2, yi2 = map(int, (xi1, yi1, xi2, yi2))
                        if (xi2 - xi1) <  self.min_feat_wh or (yi2 - yi1) < self.min_feat_wh:
                            stats["skip_too_small"] += 1; continue
                        roi = fmap[int(bi2), :, yi1:yi2, xi1:xi2]
                        v = roi.mean(dim=(1, 2))
                        if self.per_class and per_cls[int(c)] >= self.per_class:
                            stats["skip_quota_class"] += 1; continue
                        feats.append(v.cpu().numpy()); labels.append(int(c))
                        per_cls[int(c)] += 1; stats["collected"] += 1

                if self.total_max and stats["collected"] >= self.total_max:
                    stats["stop_total_max"] += 1; break
        finally:
            handle.remove()
            model.train(was_train)

        if self.verbose:
            LOGGER.info(
                "[TSNE] collected: {collected} | skip_no_feat={skip_no_feat}, skip_not4d={skip_not4d}, "
                "skip_empty_targets={skip_empty_targets}, skip_too_small={skip_too_small}, "
                "skip_quota_class={skip_quota_class}, stop_total_max={stop_total_max}, "
                "stop_max_batches={stop_max_batches}".format(**stats)
            )

        if len(feats) < 10:
            LOGGER.warning(f"[TSNE] only {len(feats)} vectors; skip plot")
            return

        X = np.stack(feats, 0).astype(np.float32)
        y = np.array(labels, dtype=np.int32)
        mu, sd = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True) + 1e-6
        X = (X - mu) / sd
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)

        feat_dim = X.shape[1]
        if self.verbose:
            LOGGER.info(f"[TSNE] vector_dim={feat_dim}, samples={len(X)}")

        if self.pca_dim > 0 and feat_dim > self.pca_dim:
            try:
                from sklearn.decomposition import PCA
                X = PCA(n_components=self.pca_dim, whiten=False, random_state=42).fit_transform(X)
                if self.verbose:
                    LOGGER.info(f"[TSNE] PCA -> {self.pca_dim} dims")
            except Exception:
                pass

        try:
            from sklearn.manifold import TSNE
        except Exception:
            LOGGER.warning("[TSNE] scikit-learn not installed")
            return

        n = X.shape[0]
        perplexity = max(5, min(50, n // 3))
        X2 = TSNE(n_components=2, init="pca", perplexity=perplexity, learning_rate=200, metric="cosine",
                  early_exaggeration=12.0, random_state=42).fit_transform(X)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        for cid, name in CLASS_LABELS.items():
            m = (y == cid)
            if m.any():
                plt.scatter(X2[m, 0], X2[m, 1], s=8, alpha=0.8, label=name)
        plt.title(f"ROI Embeddings (after STN) — epoch {ep}")
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        plt.tight_layout()
        plt.savefig(png_path, dpi=220)
        plt.close()
        LOGGER.info(f"[TSNE] saved {png_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  7) SupCon % Logger
# ──────────────────────────────────────────────────────────────────────────────
class SupConPercentLogger:
    def __init__(self):
        self.sum_num=0.0; self.sum_den=0.0

    def _read_batch_losses(self, trainer):
        names,vals=getattr(trainer,"loss_names",None),getattr(trainer,"tloss",None)
        if names and vals is not None:
            vals=vals.detach().cpu().tolist() if torch.is_tensor(vals) else list(vals)
            m={n:float(v) for n,v in zip(names,vals)}
            sup=m.get("supcon_loss",0.0); den=sum(m.values())
            return sup,den
        return None,None

    def on_train_batch_end(self,trainer):
        sup,den=self._read_batch_losses(trainer)
        if sup is not None:
            self.sum_num+=sup; self.sum_den+=max(den,1e-9)

    def on_train_epoch_end(self,trainer):
        if self.sum_den>0:
            LOGGER.info(f"[SupCon%] epoch {trainer.epoch+1}: ≈ {100*self.sum_num/self.sum_den:.1f}%")
        self.sum_num=self.sum_den=0.0


# ──────────────────────────────────────────────────────────────────────────────
#  8) SupCon projector registry + attacher (đã PATCH LR)
# ──────────────────────────────────────────────────────────────────────────────
_SUPCON_PROJ_GLOBAL = None
def supcon_register_projector(proj):
    global _SUPCON_PROJ_GLOBAL
    _SUPCON_PROJ_GLOBAL = proj
    LOGGER.info("[SupConProj] projector registered (global)")

def _supcon_get_global_projector():
    return _SUPCON_PROJ_GLOBAL


class AttachSupConProjToOptim:
    """Chỉ thêm param_group RIÊNG cho projector; KHÔNG làm thay đổi LR nhóm khác."""
    def __init__(self):
        self.attached = False
        LOGGER.info("[SupConProj] attacher ready")

    def on_train_start(self, trainer, *a, **kw):
        self._try_attach(trainer, when="on_train_start")
    def on_train_batch_start(self, trainer, *a, **kw):
        if not self.attached: self._try_attach(trainer, when="on_train_batch_start")
    def on_train_batch_end(self, trainer, *a, **kw):
        if not self.attached: self._try_attach(trainer, when="on_train_batch_end")

    def _resolve_projector(self, trainer):
        cand = []
        crit = getattr(trainer, "criterion", None)
        if crit is not None:
            cand += [getattr(crit, k, None) for k in ("_supcon_proj", "supcon_projector", "projector", "supcon_proj")]
        loss = getattr(trainer, "loss", None)
        if loss is not None:
            cand += [getattr(loss, k, None) for k in ("_supcon_proj", "supcon_projector", "projector", "supcon_proj")]
        cand += [getattr(trainer, k, None) for k in ("_supcon_proj", "supcon_projector", "supcon_proj")]
        model = getattr(trainer, "model", None)
        if model is not None:
            cand += [getattr(model, k, None) for k in ("_supcon_proj", "supcon_projector", "supcon_proj")]
        cand.append(_supcon_get_global_projector())
        for c in cand:
            if c is not None:
                return c
        return None

    def _attach(self, trainer, proj, when: str):
        from ultralytics.utils import LOGGER
        import torch.nn as nn

        opt = getattr(trainer, "optimizer", None)
        if opt is None:
            LOGGER.info(f"[SupConProj] optimizer not ready ({when}) -> defer")
            return False

        # Resolve projector if not provided
        if proj is None or not isinstance(proj, nn.Module):
            proj = self._resolve_projector(trainer)
        if proj is None or not isinstance(proj, nn.Module):
            LOGGER.info(f"[SupConProj] projector not ready ({when}) -> defer")
            return False

        params = [p for p in proj.parameters() if p.requires_grad]
        if not params:
            LOGGER.info("[SupConProj] projector has no trainable params -> skip")
            return False

        # Snapshot LR/initial_lr hiện tại (để lát khôi phục các nhóm cũ)
        old_lrs   = [g.get("lr", None) for g in opt.param_groups]
        old_inits = [g.get("initial_lr", None) for g in opt.param_groups]

        # derive from initial (pre-warmup) lr to avoid bias warmup lr=0.1 bleed
        train_args = getattr(trainer, 'args', SimpleNamespace())
        base_init = float(getattr(train_args, 'lr0', 0.001))
        base_lr = base_init  # dùng pre-warmup làm lr hiện tại cho projector
        base_wd = 0.0

        # Tìm xem projector đã nằm trong nhóm nào chưa
        target_pg = None
        proj_param_ids = {id(p) for p in params}
        for g in opt.param_groups:
            gids = {id(p) for p in g.get("params", [])}
            if proj_param_ids & gids:
                target_pg = g
                break

        # Helper: bơm defaults của optimizer vào group nếu thiếu
        def _fill_defaults(pg):
            defaults = getattr(opt, "defaults", {}) or {}
            for k, v in defaults.items():
                if k not in pg:
                    pg[k] = v
            if "initial_lr" not in pg:
                pg["initial_lr"] = pg.get("lr", base_init)
            if "weight_decay" not in pg:
                pg["weight_decay"] = base_wd
            if "name" not in pg:
                pg["name"] = "supcon_proj"
            return pg

        if target_pg is None:
            # TẠO NHÓM MỚI bằng add_param_group để tự nhận 'betas', 'eps', ...
            new_pg = {
                "params": list(params),
                "lr": float(base_lr),
                "weight_decay": float(0.0),   # projector thường không decay
                "initial_lr": float(base_init),
                "name": "supcon_proj",
            }
            new_pg = _fill_defaults(new_pg)
            opt.add_param_group(new_pg)  # <-- QUAN TRỌNG: dùng API chính thống
            LOGGER.info(f"[SupConProj] ADDED (when={when}) | n_params={sum(p.numel() for p in params)} | lr={base_lr} wd=0.0")
        else:
            # Đảm bảo đủ params projector & bơm defaults nếu thiếu
            existed = {id(p) for p in target_pg["params"]}
            for p in params:
                if id(p) not in existed:
                    target_pg["params"].append(p)

            _fill_defaults(target_pg)
            if float(target_pg.get("lr", 0.0)) == 0.0:
                target_pg["lr"] = float(base_lr)
            LOGGER.info(f"[SupConProj] REUSED (when={when}) | lr={target_pg['lr']} wd={target_pg.get('weight_decay', 0.0)}")

        # Khôi phục lr/initial_lr cho các nhóm KHÔNG phải supcon_proj (an toàn với scheduler)
        for i, g in enumerate(opt.param_groups):
            if g.get("name", "") == "supcon_proj":
                continue
            if old_lrs[i] is not None:
                g["lr"] = old_lrs[i]
            if old_inits[i] is not None:
                g["initial_lr"] = old_inits[i]

        # Log groups
        try:
            names = ", ".join([f"pg{i} lr={g.get('lr', None)} wd={g.get('weight_decay', None)} "
                               f"name={g.get('name', '')} init={g.get('initial_lr', None)}"
                               for i, g in enumerate(opt.param_groups)])
            LOGGER.info(f"[OPT/LR groups] {names}")
        except Exception:
            pass

        # Global registry
        try:
            from ultralytics.utils.stn_utils import supcon_register_projector
            supcon_register_projector(proj)
        except Exception:
            pass
        return True

    def _try_attach(self, trainer, when):
        if self.attached: return
        proj = self._resolve_projector(trainer)
        if proj is None:
            LOGGER.info(f"[SupConProj] projector not ready ({when}) -> defer"); return
        if self._attach(trainer, proj, when): self.attached = True


# ──────────────────────────────────────────────────────────────────────────────
#  9) NaN guard + batch sanity
# ──────────────────────────────────────────────────────────────────────────────
class LossNaNGuard:
    def __init__(self, stop_on_nan=True, save_bad_batch=True):
        self.stop_on_nan = stop_on_nan
        self.save_bad_batch = save_bad_batch

    def on_train_batch_end(self, trainer, *args, **kwargs):
        items = getattr(trainer, "loss_items", None)
        if items is None: return

        def _tofloat(v):
            if isinstance(v, float): return v
            if torch.is_tensor(v):   return float(v.detach().item())
            try: return float(v)
            except Exception: return float("nan")

        vals = {}
        if isinstance(items, dict):
            for k, v in items.items(): vals[k] = _tofloat(v)
        else:
            keys = ["box_loss", "cls_loss", "dfl_loss", "supcon_loss"]
            for i, v in enumerate(items):
                if i < len(keys): vals[keys[i]] = _tofloat(v)

        bad = {k: v for k, v in vals.items() if (math.isnan(v) or math.isinf(v))}
        if not bad: return

        epoch = getattr(trainer, "epoch", -1)
        step  = getattr(trainer, "batch_i", getattr(trainer, "ni", -1))
        LOGGER.error(f"[NaNGuard] epoch={epoch} step={step} NaN/Inf in loss: {bad}")

        batch = getattr(trainer, "batch", None)
        if self.save_bad_batch and isinstance(batch, dict) and ("img" in batch):
            try:
                save_dir = Path(getattr(trainer, "save_dir", Path(".")))
                fname = save_dir / f"nan_batch_e{epoch:03d}_i{int(step):06d}.jpg"
                plot_images(images=batch["img"], batch=batch, fname=fname)
                LOGGER.error(f"[NaNGuard] saved bad batch -> {fname}")
            except Exception as e:
                LOGGER.error(f"[NaNGuard] save-batch failed: {e}")

        if self.stop_on_nan:
            raise RuntimeError(f"[NaNGuard] Stop due to NaN in {list(bad.keys())}")


class BatchSanityFilter:
    """Lọc nhẹ NaN/Inf & clamp bbox trước khi tính loss (an toàn, không đổi logic)."""
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def _extract_batch_ni(self, trainer, *args, **kwargs):
        batch = kwargs.get("batch", None)
        ni    = kwargs.get("ni", None)
        if batch is None and len(args) >= 1:
            batch = args[0]
        if ni is None and len(args) >= 2:
            ni = args[1]
        if batch is None:
            batch = getattr(trainer, "batch", None)
        if ni is None:
            ni = getattr(trainer, "batch_i", getattr(trainer, "ni", None))
        return batch, ni

    def on_train_batch_start(self, trainer, *args, **kwargs):
        batch, ni = self._extract_batch_ni(trainer, *args, **kwargs)
        if batch is None:
            return
        for k in ("img", "bboxes", "cls"):
            v = batch.get(k, None) if isinstance(batch, dict) else None
            if torch.is_tensor(v):
                batch[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if isinstance(batch, dict) and torch.is_tensor(batch.get("bboxes", None)):
            bb = batch["bboxes"]
            if torch.all(bb.abs() <= 2.0):
                batch["bboxes"] = torch.clamp(bb, 0.0 - self.eps, 1.0 + self.eps)


# ──────────────────────────────────────────────────────────────────────────────
#  10) Preview BG pairing
# ──────────────────────────────────────────────────────────────────────────────
class DebugBgPairROIs:
    """
    Lưu ảnh preview theo cặp [FG | BG] mỗi epoch (ít cặp) để kiểm pairing & negative ROIs.
    """
    def __init__(self, epochs=(0,1,2,5,10), max_pairs=6):
        self.epochs = set(int(e) for e in epochs)
        self.max_pairs = int(max_pairs)

    def _label_path(self, img_path: str) -> str | None:
        p = Path(img_path)
        lbl = Path(str(p).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)).with_suffix(".txt")
        return str(lbl) if lbl.exists() else None

    def _read_yolo_xyxy(self, lbl_path: str, W: int, H: int):
        boxes = []
        if not lbl_path or not os.path.exists(lbl_path):
            return boxes
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int(max(0, (cx - bw / 2) * W))
                    y1 = int(max(0, (cy - bh / 2) * H))
                    x2 = int(min(W - 1, (cx + bw / 2) * W))
                    y2 = int(min(H - 1, (cy + bh / 2) * H))
                    boxes.append((x1, y1, x2, y2, cls_id))
        return boxes

    def _draw(self, img, boxes, default_color=(0, 0, 255), thick=2, tag=None, names: dict | None = None, per_class_color: bool = False):
        out = img.copy()
        for (x1, y1, x2, y2, ci) in boxes:
            col = (CLASS_COLORS[int(ci) % len(CLASS_COLORS)] if per_class_color else default_color)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, thick)
            name = (names.get(int(ci)) if names and int(ci) in names else CLASS_LABELS.get(int(ci), str(int(ci))))
            cv2.putText(out, name, (x1 + 4, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        if tag:
            cv2.putText(out, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
        return out

    def on_train_epoch_end(self, trainer):
        e = int(getattr(trainer, "epoch", 0))
        if e not in self.epochs:
            return

        try:
            loader = trainer.train_loader
            batch = next(iter(loader))
        except Exception as ex:
            LOGGER.warning(f"[DebugBgPairROIs] cannot fetch batch: {ex}")
            return

        pair_idx = batch.get("pair_idx", None)
        abn_mask = batch.get("abn_mask", None)
        im_files = batch.get("im_files", None)
        if pair_idx is None or abn_mask is None or im_files is None:
            LOGGER.warning("[DebugBgPairROIs] missing pair_idx/abn_mask/im_files in batch. Is paired loader active?")
            return

        save_dir = Path(str(trainer.save_dir)) / "bgpair_preview"
        save_dir.mkdir(parents=True, exist_ok=True)

        rows, take = [], 0
        for (i, j) in pair_idx.tolist():
            if take >= self.max_pairs: break
            fg_i, bg_j = (i, j) if bool(abn_mask[i]) and not bool(abn_mask[j]) else ((j, i) if bool(abn_mask[j]) and not bool(abn_mask[i]) else (None, None))
            if fg_i is None: continue

            fg_p, bg_p = im_files[fg_i], im_files[bg_j]
            if not (fg_p and bg_p and os.path.exists(fg_p) and os.path.exists(bg_p)): continue

            L = cv2.imread(fg_p, cv2.IMREAD_COLOR); R = cv2.imread(bg_p, cv2.IMREAD_COLOR)
            if L is None or R is None: continue
            H, W = L.shape[:2]

            fg_boxes = self._read_yolo_xyxy(self._label_path(fg_p), W, H)

            H2, W2 = R.shape[:2]
            bg_boxes = []
            for (x1, y1, x2, y2, ci) in fg_boxes:
                x1b = max(0, min(W2 - 1, x1)); y1b = max(0, min(H2 - 1, y1))
                x2b = max(1, min(W2,     x2)); y2b = max(1, min(H2,     y2))
                if x2b > x1b and y2b > y1b:
                    bg_boxes.append((x1b, y1b, x2b, y2b, ci))

            names = None
            try:
                names = getattr(getattr(trainer, "model", None), "names", None) or getattr(getattr(trainer, "validator", None), "names", None)
            except Exception:
                names = None

            Ld = self._draw(L, fg_boxes, default_color=(0, 0, 255), tag="FG", names=names, per_class_color=True)
            Rd = self._draw(R, bg_boxes, default_color=(0, 255, 255), tag="BG (pseudo bbox)", names=names, per_class_color=False)

            h = max(Ld.shape[0], Rd.shape[0]); w = Ld.shape[1] + Rd.shape[1]
            row = np.zeros((h, w, 3), dtype=np.uint8)
            row[:Ld.shape[0], :Ld.shape[1]] = Ld
            row[:Rd.shape[0], Ld.shape[1]:] = Rd
            rows.append(row)

            take += 1
            if take >= self.max_pairs: break

        if not rows:
            return

        w = max(r.shape[1] for r in rows)
        out_h = sum(r.shape[0] for r in rows)
        out = np.zeros((out_h, w, 3), dtype=np.uint8)
        y = 0
        for r in rows:
            out[y:y+r.shape[0], :r.shape[1]] = r
            y += r.shape[0]

        fp = save_dir / f"epoch_{e:03d}.jpg"
        cv2.imwrite(str(fp), out)
        LOGGER.info(f"[DebugBgPairROIs] saved {fp} ({len(rows)} pairs)")

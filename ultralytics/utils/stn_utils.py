from __future__ import annotations
import os, json, time, cv2, torch, numpy as np
from types import SimpleNamespace
from PIL import ImageFont, ImageDraw, Image

from ultralytics import nn
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils.ops import xywh2xyxy
from pathlib import Path
import torch
import torch.nn.functional as F
# --- FIX: Bypass STN on validation (ép identity) ---
import types

def _find_stn_modules(model):
    stns = []
    try:
        for m in model.modules():
            # tên lớp có thể là 'SpatialTransformer' (ultralytics.nn.modules.block.SpatialTransformer)
            if m.__class__.__name__.lower() in ("spatialtransformer", "stn", "spatialtransform"):
                stns.append(m)
    except Exception:
        pass
    return stns

class ForceSTNIdentityOnVal:
    def __init__(self):
        self._patched = []

    def on_val_start(self, validator):
        # cố lấy model từ validator hoặc trainer
        model = getattr(validator, "model", None)
        if model is None:
            model = getattr(getattr(validator, "trainer", None), "model", None)
        if model is None:
            LOGGER.warning("[STN/FIX] on_val_start: cannot resolve model -> skip")
            return

        for stn in _find_stn_modules(model):
            if hasattr(stn, "_orig_forward"):
                continue
            stn._orig_forward = stn.forward
            stn.forward = types.MethodType(lambda self, x, *a, **k: x, stn)
            self._patched.append(stn)
        if self._patched:
            LOGGER.info(f"[STN/FIX] Bypass {len(self._patched)} STN module(s) during validation")

    def on_val_end(self, validator):
        # khôi phục
        for stn in self._patched:
            if hasattr(stn, "_orig_forward"):
                stn.forward = stn._orig_forward
                delattr(stn, "_orig_forward")
        if self._patched:
            LOGGER.info("[STN/FIX] Restore STN after validation")
        self._patched = []
# STN
try:
    from ultralytics.nn.modules.block import SpatialTransformer as STN
except Exception:
    STN = type("STN", (), {})  # fallback an toàn

# ───────── Labels & colors ─────────
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

# ───────── Helpers ─────────
class _Timer:
    def __init__(self, tag): self.tag = tag
    def __enter__(self): self.t0 = time.time()
    def __exit__(self, *_): LOGGER.info(f"[{self.tag}] {(time.time()-self.t0):.1f}s")

def _put_text(img, txt, pos, col, size=22):
    pil = Image.fromarray(img)
    d   = ImageDraw.Draw(pil)
    try: font = ImageFont.truetype(FONT_PATH, size)
    except: font = ImageFont.load_default()
    d.text(pos, txt, font=font, fill=col)
    return np.asarray(pil).copy()

# ───────── 1) Bật/tắt STN ─────────
class STNControl:
    """
    - Freeze STN N epoch đầu (freeze_epochs).
    - Khi VALIDATION: ép STN chạy identity để tránh lệch hệ tọa độ.
    - Sau khi mở STN: ổn định theta bằng SVD clamp + clamp tịnh tiến, và blend warmup (alpha 0->1).
    - Tất cả patch forward của STN đều TRẢ VỀ TENSOR DUY NHẤT (không trả tuple).
      Theta (đã squash) sẽ chuyển ra ngoài qua m.record_theta(...) nếu có.
    """

    # ------------------------------------------------------------
    # BLOCK 1 — Config & state
    # ------------------------------------------------------------
    def __init__(
        self,
        freeze_epochs: int = 0,
        stn_warmup: int = 20,
        tmax: float = 0.20,
        smin: float = 0.85,
        smax: float = 1.15,
        log: bool = False,  # chỉ in log khi thật sự cần
    ):
        self.freeze_epochs = max(0, int(freeze_epochs))
        self.stn_warmup = max(0, int(stn_warmup))
        self.tmax, self.smin, self.smax = float(tmax), float(smin), float(smax)
        self.log = bool(log)

        self._last_alpha: float = 0.0
        self._last_mode: str = "identity"  # "identity" | "blend"

    def _dbg(self, msg: str):
        if self.log:
            LOGGER.info(msg)

    # ------------------------------------------------------------
    # BLOCK 2 — Model resolve & helpers
    # ------------------------------------------------------------
    @staticmethod
    def _unwrap(m):
        return STNControl._unwrap(m.module) if hasattr(m, "module") else m

    def _resolve_root(self, obj):
        """
        Trả về root model ở cả train/val:
        - trainer.model; trainer.model.ema; trainer.ema
        - validator.model; validator.trainer.model; validator.model.ema
        - hoặc obj nếu obj là nn.Module
        """
        cand = []
        m = getattr(obj, "model", None)
        t = getattr(obj, "trainer", None)
        v = getattr(obj, "validator", None)

        cand += [m, getattr(m, "ema", None)]
        cand += [getattr(obj, "ema", None)]
        if t:
            cand += [getattr(t, "model", None), getattr(getattr(t, "model", None), "ema", None)]
        if v:
            cand += [getattr(v, "model", None), getattr(getattr(v, "model", None), "ema", None)]

        if hasattr(obj, "modules"):  # nn.Module
            cand.append(obj)

        for c in cand:
            if c is not None:
                return self._unwrap(c)
        return None

    @staticmethod
    def _is_stn(m) -> bool:
        n = m.__class__.__name__
        return n in {"SpatialTransformer", "STN", "SpatialTransformer2D", "SpatialTransformerBlock"}

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

    # ------------------------------------------------------------
    # BLOCK 3 — Patchers (identity / blend) + restore
    # ------------------------------------------------------------
    @staticmethod
    def _ensure_orig_forward(m):
        if not hasattr(m, "_stn_forward_orig"):
            m._stn_forward_orig = m.forward

    @staticmethod
    def _set_patch_flag(m, mode: str | None):
        m._stn_patched_identity = bool(mode == "identity")
        m._stn_patched_blend = bool(mode == "blend")

    def _patch_identity(self, m):
        self._ensure_orig_forward(m)

        def _forward_identity(x, *_, **__):
            # không warp, nhưng vẫn ghi theta=I cho loss/debug nếu cần
            B, C, H, W = x.shape
            theta_I = torch.tensor([[1, 0, 0], [0, 1, 0]], device=x.device, dtype=x.dtype)\
                        .unsqueeze(0).repeat(B, 1, 1)
            self._record_theta(m, theta_I)
            return x  # TENSOR DUY NHẤT

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

            if theta is None or not isinstance(theta, torch.Tensor):
                # không có theta -> không squash, coi như identity/blend không đổi
                B = x.shape[0]
                theta_I = torch.tensor([[1, 0, 0], [0, 1, 0]], device=x.device, dtype=x.dtype)\
                            .unsqueeze(0).repeat(B, 1, 1)
                self._record_theta(m, theta_I)
                return x if alpha < 1.0 else x_t

            B, C, H, W = x.shape
            T = theta.view(-1, 2, 3).to(dtype=x.dtype)
            M, t = T[:, :, :2], T[:, :, 2]

            # clamp translation (tanh để mượt)
            t = t.tanh() * tmax

            # clamp scale/shear bằng SVD
            try:
                U, S, Vh = torch.linalg.svd(M)  # (B,2,2), (B,2), (B,2,2)
                S = S.clamp(min=smin, max=smax)
                M_safe = U @ torch.diag_embed(S) @ Vh
            except Exception:
                I = torch.eye(2, device=x.device, dtype=x.dtype).unsqueeze(0)
                M_safe = 0.75 * (M - I) + I

            theta_safe = torch.cat([M_safe, t.unsqueeze(-1)], dim=-1)  # (B,2,3)

            # warp ảnh với theta_safe
            grid = torch.nn.functional.affine_grid(theta_safe, size=(B, C, H, W), align_corners=False)
            x_stab = torch.nn.functional.grid_sample(
                x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )

            # blend: y = x + (x_stab - x) * alpha
            y = x_stab if alpha >= 1.0 else (x if alpha <= 0.0 else x + (x_stab - x) * alpha)

            self._record_theta(m, theta_safe)
            return y  # TENSOR DUY NHẤT

        m.forward = _forward_blend
        self._set_patch_flag(m, "blend")

    @staticmethod
    def _restore(m):
        if hasattr(m, "_stn_forward_orig"):
            m.forward = m._stn_forward_orig
        STNControl._set_patch_flag(m, None)

    # ------------------------------------------------------------
    # BLOCK 4 — Apply patch to all STN modules
    # ------------------------------------------------------------
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
            self._patch_blend(m, alpha)  # ghi đè nếu đang identity

    # ------------------------------------------------------------
    # BLOCK 5 — Ultralytics callbacks
    # ------------------------------------------------------------
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

# ───────── 2) Debug ảnh STN (FULL) ─────────
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
                imgs[0].cpu(),                         # 0) ảnh tensor CxHxW, thang 0..1 (thường là vậy)
                tgts["bboxes"][idx0].cpu(),            # 1) bboxes của ảnh 0 (xywh, 0..1)
                tgts["cls"][idx0].cpu(),               # 2) labels của ảnh 0
                (tgts.get("im_file") or [None])[0]     # 3) đường dẫn gốc (nếu có)
            ))
            if len(self.samples) >= self.max_images:
                break

    # --- helper: scale xyxy theo tỉ lệ w/h ---
    @staticmethod
    def _scale_xyxy(xyxy_t: "np.ndarray", rx: float, ry: float) -> "np.ndarray":
        out = xyxy_t.astype(np.float32).copy()
        out[:, [0, 2]] *= float(rx)
        out[:, [1, 3]] *= float(ry)
        return out

    # --- helper: warp bbox (xywh px) bằng theta (2x3) → trả ra xyxy px ---
    @staticmethod
    def _warp_boxes_xywh_with_theta(bxywh: "torch.Tensor", W: int, H: int, theta_2x3: "torch.Tensor") -> "torch.Tensor":
        """
        bxywh: (N,4) ở pixel (x,y,w,h) của ảnh gốc (input)
        W,H : kích thước ảnh (pixel)
        theta_2x3: (2,3) từ STN (mapping output->input trong hệ [-1,1])

        Trả về xyxy (N,4) ở pixel của ảnh STN (output).
        """
        if theta_2x3 is None:
            # không warp
            x, y, w, h = bxywh.unbind(-1)
            x1, y1 = x - w / 2, y - h / 2
            x2, y2 = x + w / 2, y + h / 2
            return torch.stack([x1, y1, x2, y2], dim=1)

        device = bxywh.device
        dtype  = bxywh.dtype

        x, y, w, h = bxywh.unbind(-1)
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2

        # 4 góc bbox (pixel input)
        Xs = torch.stack([x1, x2, x2, x1], dim=1)  # (N,4)
        Ys = torch.stack([y1, y1, y2, y2], dim=1)

        # pixel -> [-1,1]
        def pix2norm(px, L):
            return (px / max(L - 1, 1) * 2.0) - 1.0

        Xn = pix2norm(Xs, W)
        Yn = pix2norm(Ys, H)

        # nghịch đảo θ (đưa input->output)
        th = theta_2x3.detach().to(dtype=torch.float32, device=device)
        A  = torch.tensor([[th[0, 0], th[0, 1], th[0, 2]],
                           [th[1, 0], th[1, 1], th[1, 2]],
                           [0.0,      0.0,      1.0     ]],
                          dtype=torch.float32, device=device)
        Ainv = torch.linalg.inv(A)  # 3x3

        ones  = torch.ones_like(Xn)
        P_in  = torch.stack([Xn, Yn, ones], dim=1)         # (N,3,4)
        P_out = torch.einsum("ij,njk->nik", Ainv, P_in)    # (N,3,4)
        Xo, Yo = P_out[:, 0, :], P_out[:, 1, :]            # (N,4)

        # [-1,1] -> pixel
        def norm2pix(pn, L):
            return (pn + 1.0) * 0.5 * (L - 1)

        Xp = norm2pix(Xo, W)
        Yp = norm2pix(Yo, H)

        # đóng khung + clamp
        x1p, x2p = Xp.min(dim=1).values, Xp.max(dim=1).values
        y1p, y2p = Yp.min(dim=1).values, Yp.max(dim=1).values
        x1p = x1p.clamp(0, W - 1);  y1p = y1p.clamp(0, H - 1)
        x2p = x2p.clamp(1, W);      y2p = y2p.clamp(1, H)

        return torch.stack([x1p, y1p, x2p, y2p], dim=1)  # (N,4)

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
                # --- chạy 1 forward để hook ghi theta/stn_out cho riêng ảnh này
                x = img0.clone().to(device).unsqueeze(0).float()
                if x.max() > 1:
                    x = x / 255.0
                if hasattr(trainer, "state"):
                    trainer.state.pop("stn_theta", None)
                    trainer.state.pop("stn_out", None)
                _ = model(x)
                theta = getattr(trainer, "state", {}).get("stn_theta", None)  # [B,2,3] hoặc [2,3]
                stn_img = getattr(trainer, "state", {}).get("stn_out", None)

                # --- trái = ảnh gốc (GIỮ NGUYÊN)
                ori_np = img0.detach().cpu().numpy()
                if ori_np.ndim == 3 and ori_np.shape[0] in (1, 3):  # CxHxW -> HxWxC
                    ori_np = np.transpose(ori_np, (1, 2, 0))
                if ori_np.max() <= 1.5:  # thang 0..1
                    ori_np = (ori_np * 255.0).round()
                ori_np = np.clip(ori_np, 0, 255).astype(np.uint8)
                if ori_np.ndim == 2:
                    ori_np = np.repeat(ori_np[..., None], 3, axis=2)
                if ori_np.shape[2] == 1:
                    ori_np = np.repeat(ori_np, 3, axis=2)
                ori = ori_np  # (H_src, W_src, 3)

                H_src, W_src = ori.shape[:2]

                # --- phải = ảnh sau STN (nếu không có thì dùng lại ảnh gốc)
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

                # scale về cùng chiều cao
                def _resize_h(im, H=H_panel):
                    h, w = im.shape[:2]
                    if h == H:
                        return im
                    new_w = int(round(w * H / float(h)))
                    return cv2.resize(im, (new_w, H), interpolation=cv2.INTER_LINEAR)

                L = _resize_h(ori, H_panel)  # (H_panel, W_L, 3)
                R = _resize_h(vis, H_panel)  # (H_panel, W_R, 3)
                H0, W_L = L.shape[:2]
                _,  W_R = R.shape[:2]

                # --- bbox gốc theo pixel ảnh nguồn (H_src, W_src)
                bxywh_src = (bxywh.detach().cpu().numpy() *
                             np.array([W_src, H_src, W_src, H_src], dtype=np.float32))  # (N,4) xywh px

                # xyxy cho panel trái: scale từ (W_src,H_src) -> (W_L,H_panel)
                xyxy_src = xywh2xyxy(torch.from_numpy(bxywh_src)).numpy().astype(np.float32)  # (N,4)
                rx_L = float(W_L) / float(W_src)
                ry_L = float(H_panel) / float(H_src)
                xyxy_left = self._scale_xyxy(xyxy_src, rx_L, ry_L).round().clip(0, 10**9).astype(int)

                # xyxy cho panel phải:
                xyxy_right = xyxy_left.copy()
                if theta is not None and torch.is_tensor(theta):
                    # chọn theta[0] nếu có batch dim
                    th = theta
                    if th.dim() == 3 and th.shape[0] >= 1:
                        th = th[0]
                    elif th.dim() == 2:
                        pass
                    else:
                        th = None  # format lạ -> bỏ warp

                    if th is not None:
                        with torch.no_grad():
                            tb = torch.from_numpy(bxywh_src).to(dtype=torch.float32)  # (N,4) xywh (px, ảnh nguồn)
                            tbw_src = self._warp_boxes_xywh_with_theta(tb, W_src, H_src, th)  # (N,4) xyxy (px nguồn sau warp)
                            rx_R = float(W_R) / float(W_src)
                            ry_R = float(H_panel) / float(H_src)
                            xyxy_right = (tbw_src.detach().cpu().numpy().astype(np.float32))
                            xyxy_right = self._scale_xyxy(xyxy_right, rx_R, ry_R).round().clip(0, 10**9).astype(int)

                def _draw(img, xyxy_use):
                    out = img.copy()
                    labels = bcls.view(-1).tolist()
                    for (x1, y1, x2, y2), c in zip(xyxy_use, labels):
                        x1 = int(max(0, min(img.shape[1] - 1, x1)))
                        y1 = int(max(0, min(img.shape[0] - 1, y1)))
                        x2 = int(max(1, min(img.shape[1],     x2)))
                        y2 = int(max(1, min(img.shape[0],     y2)))
                        cv2.rectangle(out, (x1, y1), (x2, y2), CLASS_COLORS[int(c) % len(CLASS_COLORS)], 2)
                        out = _put_text(out, CLASS_LABELS.get(int(c), str(int(c))), (x1 + 4, max(0, y1 - 22)),
                                        CLASS_COLORS[int(c) % len(CLASS_COLORS)], size=28)
                    return out

                Ld = _draw(L, xyxy_left)      # trái: giữ nguyên
                Rd = _draw(R, xyxy_right)     # phải: đã warp nếu có theta

                # tiêu đề
                Ld = _put_text(Ld, "ORIGINAL", (10, 10), (255, 255, 0), size=42)
                Rd = _put_text(Rd, "STN",      (10, 10), (255, 255, 0), size=42)

                # in ma trận θ (nếu không có -> đơn vị)
                if theta is None or (torch.is_tensor(theta) and theta.numel() == 0):
                    theta_show = torch.tensor([[1, 0, 0],
                                               [0, 1, 0]], dtype=torch.float32)
                else:
                    theta_show = theta[0] if (torch.is_tensor(theta) and theta.dim() == 3) else theta  # (2,3)

                if torch.is_tensor(theta_show):
                    t = theta_show.detach().cpu().view(2, 3).numpy()
                    txt0 = f"θ[0]: {t[0, 0]:+0.3f} {t[0, 1]:+0.3f} {t[0, 2]:+0.3f}"
                    txt1 = f"θ[1]: {t[1, 0]:+0.3f} {t[1, 1]:+0.3f} {t[1, 2]:+0.3f}"
                else:
                    t = np.array([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]], dtype=np.float32)
                    txt0 = f"θ[0]: {t[0, 0]:+0.3f} {t[0, 1]:+0.3f} {t[0, 2]:+0.3f}"
                    txt1 = f"θ[1]: {t[1, 0]:+0.3f} {t[1, 1]:+0.3f} {t[1, 2]:+0.3f}"

                Rd = _put_text(Rd, txt0, (10, 60), (120, 255, 120), size=30)
                Rd = _put_text(Rd, txt1, (10, 95), (120, 255, 120), size=30)

                # ghép 2 panel
                both = np.concatenate([Ld, Rd], axis=1)

                # viền nền
                pad = 16
                Hh, Ww = both.shape[:2]
                canvas = np.full((Hh + 2 * pad, Ww + 2 * pad, 3), pad_color, dtype=np.uint8)
                canvas[pad:pad + Hh, pad:pad + Ww] = both

                # tên file có epoch
                base = os.path.splitext(os.path.basename(str(path0) or f"sample{i:02d}"))[0]
                out_path = os.path.join(self.dbg_dir, f"{base}_epoch_{ep:03d}_{i:02d}.png")
                cv2.imwrite(out_path, canvas)

        finally:
            model.train(was_train)

        LOGGER.info(f"[DebugImages] saved samples for epoch {ep}")

# ==== PeekBatch: dump batch mô phỏng để kiểm DataLoader ====
class PeekBatch:
    """
    Lưu 1 batch mô phỏng ngay đầu mỗi epoch (không tiêu tốn batch train):
      - Tạo 'panels' tấm, mỗi tấm chứa tối đa 'max_images' ảnh (ví dụ 2 panel x 4 ảnh = 2 tấm 2x2).
      - Vẽ bbox + TÊN LỚP dựa vào trainer.names/model.names.
      - Ghi kèm metadata keys/shapes.
    Files:
      runs_stn/<run>/peek_batch/batch_epochXXX_pY.jpg / .txt
    """
    def __init__(self, save_first_n: int = 1, max_images: int = 4, panels: int = 2,
                 save_dir_name: str = "peek_batch"):
        self.save_first_n = save_first_n
        self.max_images = max_images         # ảnh / panel
        self.panels = panels                 # số panel / epoch
        self.save_dir_name = save_dir_name
        self._done = 0

    # ---------- utils ----------
    def _get_names(self, trainer):
        # cố lấy theo thứ tự: trainer.names -> validator.names -> model.names
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
                head = v[:3]
                return f"{k}: {type(v).__name__} len={len(v)} sample_head={head}"
            if isinstance(v, dict):
                return f"{k}: dict keys={list(v.keys())[:10]}"
            return f"{k}: {type(v).__name__} value={v}"
        except Exception as e:
            return f"{k}: <error summarizing: {e}>"

    def _pack_like_loader(self, items):
        """Fallback collate nếu ds.collate_fn không dùng được."""
        imgs = [torch.as_tensor(it["img"]) for it in items]
        imgs = torch.stack(imgs, 0)

        bxs, cls, bidx = [], [], []
        for i, it in enumerate(items):
            bi = it.get("bboxes", None)
            ci = it.get("cls", None)
            if isinstance(bi, torch.Tensor) and bi.ndim == 2 and bi.size(-1) == 4:
                bxs.append(bi)
                bidx.append(torch.full((bi.shape[0],), i))
            if isinstance(ci, torch.Tensor):
                cls.append(ci)
        bxs  = torch.cat(bxs, 0)  if len(bxs)  else torch.zeros((0, 4))
        cls  = torch.cat(cls, 0)  if len(cls)  else torch.zeros((0,))
        bidx = torch.cat(bidx, 0) if len(bidx) else torch.zeros((0,))
        paths = [it.get("im_file", it.get("path", "")) for it in items]
        return {"img": imgs, "bboxes": bxs, "cls": cls, "batch_idx": bidx, "im_file": paths}

    def _save_grid_panel(self, trainer, batch, img_ids, save_path: Path):
        """
        Lưu 1 panel gồm các ảnh có index trong 'img_ids' (theo batch hiện tại).
        Remap batch_idx về [0..len(img_ids)-1] để plot_images hoạt động đúng.
        """
        from ultralytics.utils.plotting import plot_images

        imgs = batch["img"].detach().float().cpu()
        if imgs.max() > 1.5:
            imgs = imgs / 255.0

        # subset ảnh
        sel = torch.tensor(img_ids, dtype=torch.long)
        imgs_sub = imgs.index_select(0, sel)

        # subset labels theo batch_idx và remap
        labels = None
        if all(k in batch for k in ("batch_idx", "cls", "bboxes")):
            bidx = batch["batch_idx"].long().cpu()
            mask = torch.isin(bidx, sel)
            if mask.any():
                old = bidx[mask]
                # map cũ->mới
                mapping = {int(v): i for i, v in enumerate(img_ids)}
                new_bidx = torch.tensor([mapping[int(v)] for v in old], dtype=torch.float32).unsqueeze(1)
                cls = batch["cls"][mask].float().unsqueeze(1)
                # cast cls -> int để in tên nhãn
                cls = cls.floor()  # an toàn
                bxs = batch["bboxes"][mask].float()
                labels = torch.cat((new_bidx, cls, bxs), 1).cpu()

        paths = batch.get("im_file", batch.get("paths", None))
        # subset path nếu có
        if isinstance(paths, (list, tuple)) and len(paths) == imgs.shape[0]:
            paths = [paths[i] for i in img_ids]

        names = self._get_names(trainer)

        plot_images(
            images=imgs_sub,
            labels=labels,
            paths=paths,
            names=names,                    # => in TÊN NHÃN
            fname=str(save_path),
            max_subplots=len(img_ids),      # 2x2 khi img_ids<=4
        )

    # ---------- main hook ----------
    def on_fit_epoch_start(self, trainer):
        if self._done >= self.save_first_n:
            return
        self._done += 1

        ds = getattr(getattr(trainer, "train_loader", None), "dataset", None)
        if ds is None:
            return

        # Lấy đúng batch_size ảnh để mô phỏng 1 batch thật, sau đó chia panel 2x2
        bs = int(getattr(trainer, "batch_size", self.max_images * self.panels))
        need = min(bs, self.max_images * self.panels)  # chỉ lấy vừa đủ cho số panel

        try:
            import numpy as np
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

        # metadata
        meta_lines = [f"[peek] epoch={trainer.epoch} imgsz={getattr(trainer, 'imgsz', None)} "
                      f"(synthetic batch, bs~{bs}, dumped={need})"]
        for k in sorted(batch.keys()):
            meta_lines.append(self._summarize_value(k, batch[k]))
        with open(save_dir / f"batch_epoch{int(trainer.epoch):03d}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(meta_lines))

        # panel hóa: mỗi panel tối đa self.max_images ảnh (=> 2x2)
        nimg = int(batch["img"].shape[0])
        take = min(nimg, self.max_images * self.panels)
        cursor = 0
        panel_id = 0
        while cursor < take and panel_id < self.panels:
            end = min(cursor + self.max_images, take)
            img_ids = list(range(cursor, end))
            out = save_dir / f"batch_epoch{int(trainer.epoch):03d}_p{panel_id}.jpg"
            self._save_grid_panel(trainer, batch, img_ids, out)
            cursor = end
            panel_id += 1

# ───────── 3) SupCon callbacks ─────────
class InjectSupConArgsMinimal:
    """
    3-hook injector:
      - on_pretrain_routine_start: chỉ CACHE vào trainer._supcon_cfg (lúc này model/crit có thể chưa sẵn sàng)
      - on_pretrain_routine_end: model/criterion đã build -> mirror supcon_* vào model.args & criterion.hyp
      - on_train_start: nhắc lại phòng TH criterion bị recreate
    """
    def __init__(self, **cfg):
        self.cfg = dict(cfg)

    # tiện ích: đảm bảo attr là SimpleNamespace
    def _ensure_ns(self, obj, attr):
        val = getattr(obj, attr, None)
        if val is None or isinstance(val, dict):
            ns = SimpleNamespace(**(val or {}))
            setattr(obj, attr, ns)
            return ns
        return val

    # 1) cache sớm để _setup_train có cái mà merge (A)
    def on_pretrain_routine_start(self, trainer):
        trainer._supcon_cfg = SimpleNamespace(**self.cfg)
        LOGGER.info("[INJECT@start] cached supcon_* -> trainer._supcon_cfg")

    # 2) sau setup_model xong: mirror vào model & criterion, đồng thời recache (B)
    def on_pretrain_routine_end(self, trainer):
        # đảm bảo model.args tồn tại dạng SimpleNamespace
        ma = self._ensure_ns(trainer.model, "args")
        # mirror supcon_* vào model.args
        for k, v in self.cfg.items():
            setattr(ma, k, v)

        # criterion.hyp nếu đã có
        crit = getattr(trainer.model, "criterion", None)
        if crit is not None:
            hyp = self._ensure_ns(crit, "hyp")
            for k, v in self.cfg.items():
                setattr(hyp, k, v)

        # Quan trọng: recache lại để “sống sót” nếu trainer vừa reset
        trainer._supcon_cfg = SimpleNamespace(**self.cfg)

        LOGGER.info(f"[INJECT@end] model.args.supcon_feat={getattr(ma, 'supcon_feat', None)} "
                    f"| on={getattr(ma, 'supcon_on', None)}")

    # 3) đầu train: ưu tiên cache NẾU cache không phải default; nếu default thì rơi về self.cfg (C)
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

        # model.args
        ma = self._ensure_ns(trainer.model, "args")
        for k, v in cfg.items():
            setattr(ma, k, v)

        # criterion.hyp
        crit = getattr(trainer.model, "criterion", None)
        if crit is not None:
            hyp = self._ensure_ns(crit, "hyp")
            for k, v in cfg.items():
                setattr(hyp, k, v)

        LOGGER.info(
            f"[INJECT@train] source={'cache' if use_cache else 'self.cfg'} | "
            f"model.args.supcon_feat={getattr(ma, 'supcon_feat', None)} | on={getattr(ma, 'supcon_on', None)}"
        )

class SupConScheduler:
    def __init__(self, schedule_str: str, default_on: int = 0):
        self.default_on=1 if default_on else 0; self.ranges=self._parse(schedule_str or "")
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
        if ma is None or isinstance(ma,dict): ma=SimpleNamespace(**(ma or {})); trainer.model.args=ma
        setattr(ma,"supcon_on",int(want_on))
        if want_on==1 and hasattr(trainer,"loss_names"):
            if not trainer.loss_names or trainer.loss_names[-1]!="supcon_loss":
                trainer.loss_names=("box_loss","cls_loss","dfl_loss","supcon_loss")
        LOGGER.info(f"[SupCon/schedule] epoch_display={e_disp} -> supcon_on={want_on}")

class ReinforceSupConToLoss:
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
        except: pass

# ───────── 4) Trainer ↔ Loss ─────────
class LinkTrainerToLoss:
    def on_train_start(self, trainer):
        if getattr(trainer,"loss",None) is not None: trainer.loss._trainer=trainer

class SyncEpochToLoss:
    def on_train_epoch_start(self, trainer):
        if getattr(trainer,"loss",None) is not None: trainer.loss.epoch=int(trainer.epoch)
    def on_train_batch_start(self, trainer):
        if getattr(trainer,"loss",None) is not None: trainer.loss.epoch=int(trainer.epoch)

# ───────── 5) Tap STN Feat ─────────
class TapSTNFeat:
    def __init__(self):
        self.handle = None

    def on_train_start(self, trainer):
        stn = None
        for m in trainer.model.modules():
            if isinstance(m, STN):
                stn = m
                break
        if stn is None:
            LOGGER.warning("[TapSTNFeat] STN not found")
            return

        def _hook(mod, inp, out):
            # Lưu fmap sau STN như cũ
            if not hasattr(trainer, "state"):
                trainer.state = {}
            trainer.state["stn_out"] = out

            # === FIX: lấy theta một cách an toàn, không dùng `or` trên Tensor ===
            theta = None
            for name in ("theta", "_theta", "last_theta"):
                val = getattr(mod, name, None)
                if isinstance(val, torch.Tensor):
                    theta = val
                    break
            # nếu không có theta thì dùng ma trận đồng nhất 2x3
            if theta is None:
                device = out.device if torch.is_tensor(out) else "cpu"
                theta = torch.tensor([[1., 0., 0.],
                                      [0., 1., 0.]], device=device)

            # Lưu vào state để DebugImages/TSNE… đọc
            trainer.state["stn_theta"] = theta.detach().clone()

        self.handle = stn.register_forward_hook(_hook)
        LOGGER.info("[TapSTNFeat] hook attached at STN")

    def on_train_end(self, trainer):
        if self.handle:
            self.handle.remove()
            self.handle = None
        LOGGER.info("[TapSTNFeat] hook removed")

# ───────── TSNE Embedding CHUẨN: lấy feature map sâu sau STN ─────────
class TSNEEmbeddings:
    """
    Duyệt loader (train/val), hook module NGAY SAU STN nhưng phải là feature *sâu*:
      - tensor 4D, C >= min_channels, và downsample >= min_downsample so với ảnh vào.
    Pool ROI bằng ROIAlign -> vector, z-score + L2-norm -> t-SNE(metric='cosine').
    Lưu PNG vào runs/.../embeddings, vẽ ở epoch 0,10,20,...
    """
    def __init__(self,
                 every=10,                 # vẽ ở ep 0,10,20,...
                 loader="val",             # "train" | "val"
                 per_class=150,
                 total_max=10000,
                 max_batches=250,
                 use_roialign=True,
                 roialign_out=1,
                 min_feat_wh=1,
                 pca_dim=128,
                 min_channels=128,         # ép chọn feature sâu (tránh C=3/16)
                 min_downsample=4,         # yêu cầu Hf<=H/4 và Wf<=W/4
                 verbose=True):
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

    # ---------- helpers ----------
    def _get_loader(self, trainer):
        if self.loader == "train":
            return trainer.train_loader
        val = getattr(getattr(trainer, "validator", None), "dataloader", None)
        return val or trainer.train_loader

    def _choose_after_stn_conv(self, trainer, device):
        """Chọn module đầu tiên SAU STN có out 4D với C>=min_channels và downsample >= min_downsample."""
        if self._feat_mod is not None:
            return self._feat_mod

        core = trainer.model
        seq = list(core.model.modules())
        stn_seen, hooks, outs = False, [], []

        def _mk_hook(m):
            def _h(_m, _i, o):
                # nhiều module trả tuple/list -> gom tất cả tensor con
                if torch.is_tensor(o):
                    outs.append((m, o))
                elif isinstance(o, (list, tuple)):
                    for t in o:
                        if torch.is_tensor(t):
                            outs.append((m, t))
            return _h

        # hook tất cả module sau khi gặp STN
        for m in seq:
            if isinstance(m, STN):
                stn_seen = True
                continue
            if not stn_seen:
                continue
            if hasattr(m, "forward"):
                hooks.append(m.register_forward_hook(_mk_hook(m)))

        # chạy probe 1 ảnh để lấy shape thật
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
            for h in hooks:
                h.remove()
            core.train(was_train)

        # lọc ứng viên “đủ sâu”
        candidates = []
        for m, o in outs:
            if not (torch.is_tensor(o) and o.dim() == 4):
                continue
            _, Cf, Hf, Wf = o.shape
            deep_enough = (Cf >= self.min_channels) and (Hf <= H//self.min_downsample) and (Wf <= W//self.min_downsample)
            if deep_enough:
                candidates.append((m, (1, Cf, Hf, Wf)))

        # fallback: nếu không có, lấy 4D cuối cùng để vẫn chạy được
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

    # ---------- callback ----------
    @torch.no_grad()
    def on_train_epoch_end(self, trainer):
        ep = int(getattr(trainer, "epoch", 0))
        if (ep % self.every) != 0:
            return  # chỉ vẽ ở 0,10,20,...

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

        # Hook cố định vào feat_mod để gom fmap
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

                # ảnh->pixel->fmap
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
                    rois = torch.stack([bidx.float(),
                                        fx1.float(), fy1.float(),
                                        fx2.float(), fy2.float()], dim=1).to(fmap.device)
                    pooled = roi_align(
                        fmap, rois,
                        output_size=(self.roialign_out, self.roialign_out),
                        spatial_scale=1.0, sampling_ratio=-1, aligned=True
                    )  # (N, Cf, r, r)
                    vecs = pooled.mean(dim=(2, 3))  # (N, Cf)
                    # nếu model có projection head cho SupCon thì dùng luôn để vẽ
                    # --- Projection guard: chỉ dùng SupConProj khi khớp chiều ---
                    proj = getattr(getattr(trainer.model, "criterion", None), "supcon_proj", None)
                    if proj is not None:
                        # cố gắng suy luận in_dim mong đợi của projection head
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
                            # ép về float32 + tắt autocast để tránh Half/Float mismatch
                            with torch.cuda.amp.autocast(enabled=False):
                                vecs = proj(vecs.float())
                        else:
                            LOGGER.warning(
                                f"[TSNE] Skip SupConProj: dim mismatch vecs={vecs.shape[-1]} vs proj_in={in_dim}")
                    # --- end guard ---

                    for vi, c in zip(vecs, bcls):
                        if self.per_class and per_cls[int(c)] >= self.per_class:
                            stats["skip_quota_class"] += 1
                            continue
                        feats.append(vi.cpu().numpy())
                        labels.append(int(c))
                        per_cls[int(c)] += 1
                        stats["collected"] += 1
                else:
                    for bi2, c, xi1, yi1, xi2, yi2 in zip(bidx, bcls, fx1, fy1, fx2, fy2):
                        xi1, yi1, xi2, yi2 = map(int, (xi1, yi1, xi2, yi2))
                        if (xi2 - xi1) < self.min_feat_wh or (yi2 - yi1) < self.min_feat_wh:
                            stats["skip_too_small"] += 1
                            continue
                        roi = fmap[int(bi2), :, yi1:yi2, xi1:xi2]
                        v = roi.mean(dim=(1, 2))
                        if self.per_class and per_cls[int(c)] >= self.per_class:
                            stats["skip_quota_class"] += 1
                            continue
                        feats.append(v.cpu().numpy())
                        labels.append(int(c))
                        per_cls[int(c)] += 1
                        stats["collected"] += 1

                if self.total_max and stats["collected"] >= self.total_max:
                    stats["stop_total_max"] += 1
                    break
        finally:
            handle.remove()
            model.train(was_train)

        if self.verbose:
            LOGGER.info(
                "[TSNE] collected: {collected} | "
                "skip_no_feat={skip_no_feat}, skip_not4d={skip_not4d}, "
                "skip_empty_targets={skip_empty_targets}, skip_too_small={skip_too_small}, "
                "skip_quota_class={skip_quota_class}, stop_total_max={stop_total_max}, "
                "stop_max_batches={stop_max_batches}".format(**stats)
            )

        if len(feats) < 10:
            LOGGER.warning(f"[TSNE] only {len(feats)} vectors; skip plot")
            return

        # --- Chuẩn hoá: z-score theo kênh toàn bộ + L2 ---
        X = np.stack(feats, 0).astype(np.float32)      # (N, C)
        y = np.array(labels, dtype=np.int32)
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-6
        X = (X - mu) / sd
        # L2
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)

        feat_dim = X.shape[1]
        if self.verbose:
            LOGGER.info(f"[TSNE] vector_dim={feat_dim}, samples={len(X)}")

        # PCA (nếu cần)
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
        X2 = TSNE(
            n_components=2, init="pca", perplexity=perplexity,
            learning_rate=200, metric="cosine",
            early_exaggeration=12.0, random_state=42
        ).fit_transform(X)

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

# ───────── 7) SupCon % Logger ─────────
class SupConPercentLogger:
    def __init__(self): self.sum_num=0.0; self.sum_den=0.0
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
        if sup is not None: self.sum_num+=sup; self.sum_den+=max(den,1e-9)
    def on_train_epoch_end(self,trainer):
        if self.sum_den>0:
            LOGGER.info(f"[SupCon%] epoch {trainer.epoch+1}: ≈ {100*self.sum_num/self.sum_den:.1f}%")
        self.sum_num=self.sum_den=0.0

class AttachSupConProjToOptim:
    """Gắn SupCon projector vào optimizer với LR/WD khớp các param group đang chạy."""

    def __call__(self, trainer, projector):
        opt = getattr(trainer, 'optimizer', None)
        if opt is None:
            LOGGER.warning("[SupConProj] no optimizer -> skip attach")
            return

        # Lấy 1 param group đang active để suy ra lr/wd “hợp lý”
        # Ưu tiên group có lr lớn nhất hiện tại (thường là non-bias)
        base_pg = max(opt.param_groups, key=lambda pg: float(pg.get('lr', 0.0)))
        base_lr = float(base_pg.get('lr', 0.0) or getattr(trainer.args, 'lr0', 0.001))
        base_wd = float(base_pg.get('weight_decay', 0.0))

        params = [p for p in projector.parameters() if p.requires_grad]
        if not params:
            LOGGER.warning("[SupConProj] projector has no trainable params")
            return

        new_pg = {
            'params': params,
            'lr': base_lr,
            'weight_decay': base_wd,
            'name': 'supcon_proj'
        }
        opt.add_param_group(new_pg)

        # Nếu có scheduler, thêm base_lr tương ứng để scheduler step không bị lệch độ dài
        sch = getattr(trainer, 'scheduler', None)
        if sch is not None and hasattr(sch, 'base_lrs'):
            try:
                sch.base_lrs.append(base_lr)
            except Exception as e:
                LOGGER.warning(f"[SupConProj] cannot extend scheduler.base_lrs: {e}")

        # In debug toàn bộ các nhóm LR để bạn check ngay
        def _pg_str(pg, i):
            return f"pg{i} lr={float(pg.get('lr',0)):.5g} wd={float(pg.get('weight_decay',0))} n={len(pg.get('params',[]))} name={pg.get('name','')}"
        pg_dbg = ', '.join(_pg_str(pg, i) for i, pg in enumerate(opt.param_groups))

        n_params = sum(p.numel() for p in params)
        LOGGER.info(f"[SupConProj] parameters added | n_params={n_params} | groups={len(opt.param_groups)} | lr={base_lr} wd={base_wd}")
        LOGGER.info(f"[OPT/LR groups] {pg_dbg}")

# === STN SupCon injector (dict-callback, không cần base class) ===============
def _attach_supcon_cfg(trainer, cfg_ns):
    crit = getattr(trainer.model, "criterion", None)
    if crit is None:
        return
    # Gắn trực tiếp vào loss (không đi vòng self.hyp)
    crit.supcon_cfg = cfg_ns
    # Mirror 1 flag cho dễ debug
    if hasattr(trainer.model, "args"):
        try:
            trainer.model.args.supcon_on = int(getattr(cfg_ns, "on", 0))
        except Exception:
            pass
    LOGGER.info("[INJECT OK] SupCon -> criterion.supcon_cfg")

def make_supcon_injector(supcon_cfg: dict):
    """
    Trả về 'callback dict' hợp lệ với Ultralytics v8.
    Dùng trong main_stn.py: callbacks=[make_supcon_injector(cfg)]
    """
    cfg_ns = SimpleNamespace(**supcon_cfg)

    def _cb(trainer, *a, **k):
        _attach_supcon_cfg(trainer, cfg_ns)

    # Gọi ở nhiều mốc để chắc chắn kể cả khi criterion bị recreate
    return {
        "on_fit_start": _cb,
        "on_train_start": _cb,
        "on_train_epoch_start": _cb,
    }
# ============================================================================
# ───────── NEW: Preview BG pairing + pseudo-ROIs ─────────
class DebugBgPairROIs:
    """
    Lưu ảnh preview theo cặp [FG | BG] mỗi epoch (ít cặp) để kiểm tra:
      - Đúng file ghép cặp từ paired loader (pair_idx, abn_mask, im_files)
      - BBox của FG, và 'bbox nền' copy sang BG (để bạn xem triết lý có hợp lý)
    Lưu ý: ảnh lấy trực tiếp từ disk theo im_files (không qua augment),
           bbox đọc từ file .txt của FG và scale theo ảnh gốc.
    """
    def __init__(self, epochs=(0,1,2,5,10), max_pairs=6):
        self.epochs = set(int(e) for e in epochs)
        self.max_pairs = int(max_pairs)

    def _label_path(self, img_path: str) -> str | None:
        p = Path(img_path)
        # images -> labels, đổi đuôi sang .txt
        lbl = Path(str(p).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)).with_suffix(".txt")
        return str(lbl) if lbl.exists() else None

    def _read_yolo_xyxy(self, lbl_path: str, W: int, H: int):
        """
        Đọc file YOLO *.txt -> trả list [(x1,y1,x2,y2,cls), ...] theo pixel.
        """
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


    def _draw(self, img, boxes, default_color=(0, 0, 255), thick=2,
              tag=None, names: dict | None = None, per_class_color: bool = False):
        """
        boxes: [(x1,y1,x2,y2,cls), ...]
        per_class_color=False -> dùng default_color; True -> tô theo CLASS_COLORS[cls]
        """
        import cv2
        out = img.copy()
        for (x1, y1, x2, y2, ci) in boxes:
            col = (CLASS_COLORS[int(ci) % len(CLASS_COLORS)]
                   if per_class_color else default_color)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, thick)
            # nhãn
            name = None
            if names and int(ci) in names:
                name = str(names[int(ci)])
            elif int(ci) in CLASS_LABELS:
                name = CLASS_LABELS[int(ci)]
            else:
                name = str(int(ci))
            cv2.putText(out, name, (x1 + 4, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        if tag:
            cv2.putText(out, tag, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
        return out


    def on_train_epoch_end(self, trainer):
        e = int(getattr(trainer, "epoch", 0))
        if e not in self.epochs:
            return

        # Lấy 1 batch từ train_loader đang dùng (đã là paired loader)
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

        import cv2, numpy as np
        save_dir = Path(str(trainer.save_dir)) / "bgpair_preview"
        save_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        take = 0
        for (i, j) in pair_idx.tolist():
            if take >= self.max_pairs:
                break
            # Xác định FG và BG theo abn_mask
            fg_i, bg_j = (i, j) if bool(abn_mask[i]) and not bool(abn_mask[j]) else \
                         ((j, i) if bool(abn_mask[j]) and not bool(abn_mask[i]) else (None, None))
            if fg_i is None:
                continue

            fg_p = im_files[fg_i]; bg_p = im_files[bg_j]
            if not (fg_p and bg_p and os.path.exists(fg_p) and os.path.exists(bg_p)):
                continue

            L = cv2.imread(fg_p, cv2.IMREAD_COLOR)
            R = cv2.imread(bg_p, cv2.IMREAD_COLOR)
            if L is None or R is None:
                continue
            H, W = L.shape[:2]

            # đọc bbox của FG (kèm cls)
            fg_lbl   = self._label_path(fg_p)
            fg_boxes = self._read_yolo_xyxy(fg_lbl, W, H)

            # "bbox nền" = copy bbox FG sang BG và clip theo kích thước BG (giữ nguyên cls)
            H2, W2 = R.shape[:2]
            bg_boxes = []
            for (x1, y1, x2, y2, ci) in fg_boxes:
                x1b = max(0, min(W2 - 1, x1))
                y1b = max(0, min(H2 - 1, y1))
                x2b = max(1, min(W2,     x2))
                y2b = max(1, min(H2,     y2))
                if x2b > x1b and y2b > y1b:
                    bg_boxes.append((x1b, y1b, x2b, y2b, ci))

            # lấy names từ trainer nếu có; fallback CLASS_LABELS
            names = None
            try:
                names = getattr(getattr(trainer, "model", None), "names", None)
                if names is None:
                    names = getattr(getattr(trainer, "validator", None), "names", None)
            except Exception:
                names = None

            # vẽ: FG màu theo LỚP; BG giữ màu vàng để "khác màu bbox trước" (FG)
            Ld = self._draw(L, fg_boxes, default_color=(0, 0, 255),
                            tag="FG", names=names, per_class_color=True)
            Rd = self._draw(R, bg_boxes, default_color=(0, 255, 255),
                            tag="BG (pseudo bbox)", names=names, per_class_color=False)

            # ghép 2 cột
            h = max(Ld.shape[0], Rd.shape[0])
            w = Ld.shape[1] + Rd.shape[1]
            row = np.zeros((h, w, 3), dtype=np.uint8)
            row[:Ld.shape[0], :Ld.shape[1]] = Ld
            row[:Rd.shape[0], Ld.shape[1]:] = Rd
            rows.append(row)

            take += 1
            if take >= self.max_pairs:
                break

        # xếp hàng dọc
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

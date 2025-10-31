from __future__ import annotations
import os, math, time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFont, ImageDraw, Image
from ultralytics.engine import trainer

from ultralytics import nn
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import smart_inference_mode
import torchvision

import os
from torchvision.utils import save_image
from ultralytics.utils.plotting import Annotator

# ──────────────────────────────────────────────────────────────────────────────
# Public constants (kept for BC)
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
# Small helpers (shared by all callbacks)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from ultralytics.nn.modules.block import SpatialTransformer as STN
except Exception:  # pragma: no cover
    STN = type("STN", (), {})

class _Ctx:
    @staticmethod
    def root(obj):
        m = getattr(obj, "model", None) or getattr(getattr(obj, "trainer", None), "model", None)
        return getattr(m, "model", m) if m is not None else None

    @staticmethod
    def unwrap(m):
        return _Ctx.unwrap(m.module) if hasattr(m, "module") else m

    @staticmethod
    def is_stn(m) -> bool:
        return m.__class__.__name__ in {"SpatialTransformer", "STN", "SpatialTransformer2D", "SpatialTransformerBlock"}

    @staticmethod
    def stn_modules(model):
        if model is None: return []
        for m in model.modules():
            if _Ctx.is_stn(m):
                yield m

    @staticmethod
    def state(owner):
        if not hasattr(owner, "state"): owner.state = {}
        return owner.state

    @staticmethod
    def put_text(img, txt, pos, col, size=22):
        pil = Image.fromarray(img); d = ImageDraw.Draw(pil)
        try: font = ImageFont.truetype(FONT_PATH, size)
        except Exception: font = ImageFont.load_default()
        d.text(pos, txt, font=font, fill=col)
        return np.asarray(pil).copy()

# ──────────────────────────────────────────────────────────────────────────────
# 1) STN controller: freeze/warmup + trust region + val bypass
# ──────────────────────────────────────────────────────────────────────────────
class STNControl(_Ctx):
    """Enable identity during early epochs & validation, then blend stabilized STN output with clamp."""
    def __init__(self, freeze_epochs=0, stn_warmup=20, tmax=0.20, smin=0.90, smax=1.10, log=False):
        self.freeze_epochs = max(0, int(freeze_epochs))
        self.stn_warmup = max(0, int(stn_warmup))
        self.tmax, self.smin, self.smax = float(tmax), float(smin), float(smax)
        self.log = bool(log)
        self._mode, self._alpha = "identity", 0.0
        self._epoch = -1  # chỉ để log nếu cần

    def _alpha_for(self, e: int) -> float:
        if e < self.freeze_epochs:
            return 0.0
        if self.stn_warmup <= 0:
            return 1.0
        k = (e - self.freeze_epochs) / float(self.stn_warmup)
        return float(max(0.0, min(1.0, k)))

    @staticmethod
    def _ensure_orig_forward(m):
        if not hasattr(m, "_stn_forward_orig"):
            m._stn_forward_orig = m.forward

    def _patch_identity(self, m):
        self._ensure_orig_forward(m)

        def f(x, *_, **__):
            B = x.shape[0]
            theta_I = x.new_tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0).repeat(B, 1, 1)
            if hasattr(m, "record_theta"):
                try:
                    m.record_theta(theta_I)
                except Exception:
                    pass
            return x

        m.forward = f
        m._stn_mode = "identity"

    def _patch_blend(self, m, alpha: float):
        """Blend output with strength alpha. If theta exists, clamp (t, s) -> stabilized grid_sample."""
        self._ensure_orig_forward(m)
        # chốt alpha cho closure (không phụ thuộc biến ngoài)
        alpha = float(max(0.0, min(1.0, alpha)))
        tmax, smin, smax = self.tmax, self.smin, self.smax

        def f(x, *a, **kw):
            from torch.nn import functional as F  # tránh yêu cầu import ở đầu file
            out = m._stn_forward_orig(x, *a, **kw)

            # 1) Unpack out -> (x_t, theta) nếu có
            x_t, theta = None, None
            if isinstance(out, (tuple, list)):
                if len(out) >= 2:
                    x_t, theta = out[0], out[1]
                elif len(out) == 1 and torch.is_tensor(out[0]):
                    x_t = out[0]
            elif torch.is_tensor(out):
                x_t = out

            # 2) Nếu có theta, kẹp để lấy x_stab
            if theta is not None and torch.is_tensor(theta):
                try:
                    B, C, H, W = x.shape
                    T = theta.view(-1, 2, 3).to(dtype=x.dtype, device=x.device)
                    M, t = T[:, :, :2], T[:, :, 2].tanh() * tmax
                    try:
                        U, S, Vh = torch.linalg.svd(M)
                        S = S.clamp(smin, smax)
                        M = U @ torch.diag_embed(S) @ Vh
                    except Exception:
                        I = torch.eye(2, device=x.device, dtype=x.dtype).unsqueeze(0)
                        M = 0.75 * (M - I) + I
                    theta_safe = torch.cat([M, t.unsqueeze(-1)], -1)

                    grid = F.affine_grid(theta_safe, size=(B, C, H, W), align_corners=False)
                    x_stab = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                    if hasattr(m, "record_theta"):
                        try:
                            m.record_theta(theta_safe)
                        except Exception:
                            pass
                    x_t = x_stab  # dùng bản ổn định để blend
                except Exception:
                    # nếu có lỗi khi clamp, cứ dùng x_t (nếu đã có) hoặc fallback ở bước 3
                    pass

            # 3) Đảm bảo luôn có x_t (fallback identity)
            if x_t is None:
                x_t = x
                try:
                    LOGGER.info("[STN] STN forward: no theta output, using identity transform.")
                except Exception:
                    pass

            # 4) Trả về theo alpha (blend an toàn)
            if alpha <= 0.0:
                return x
            if alpha >= 1.0:
                return x_t
            return x.lerp(x_t, alpha)

        m.forward = f
        m._stn_mode = "blend"

    def _apply_identity(self, model, on: bool):
        for m in self.stn_modules(model):
            if on:
                self._patch_identity(m)
            elif getattr(m, "_stn_forward_orig", None):
                m.forward = m._stn_forward_orig

    def _apply_blend(self, model, alpha: float):
        for m in self.stn_modules(model):
            self._patch_blend(m, alpha)

    def on_train_epoch_start(self, trainer):
        model = self.root(trainer)  # self.root() trả về trainer.model.model (đúng)

        # --- SỬA LẠI LOGIC TÌM EMA MODEL ---
        ema_model_raw = getattr(getattr(trainer, "ema", None), "ema", None)
        ema_model_seq = getattr(ema_model_raw, "model", None)  # Lấy .model (nn.Sequential) từ bên trong EMA
        # --- KẾT THÚC SỬA ---

        if model is None:
            return
        e = int(getattr(trainer, "epoch", 0))
        self._epoch = e
        if e < self.freeze_epochs:
            self._apply_identity(model, True)
            if ema_model_seq: self._apply_identity(ema_model_seq, True)  # <-- Dùng ema_model_seq
            self._mode, self._alpha = "identity", 0.0
            if self.log:
                LOGGER.info(f"[STN] identity @ epoch {e}")
        else:
            a = self._alpha_for(e)
            self._apply_identity(model, False)
            if ema_model_seq: self._apply_identity(ema_model_seq, False)  # <-- Dùng ema_model_seq
            self._apply_blend(model, a)
            if ema_model_seq: self._apply_blend(ema_model_seq, a)  # <-- Dùng ema_model_seq
            self._mode, self._alpha = "blend", a
            if self.log:
                LOGGER.info(f"[STN] blend α={a:.3f} (t≤{self.tmax:.2f}, s∈[{self.smin:.2f},{self.smax:.2f}])")

    def on_val_start(self, validator):
        m = self.root(validator)  # self.root() trả về trainer.model.model (đúng)

        # --- SỬA LẠI LOGIC TÌM EMA MODEL ---
        ema_m_raw = getattr(getattr(validator, "trainer", None), "ema", None)
        ema_model_raw = getattr(ema_m_raw, "ema", None)
        ema_model_seq = getattr(ema_model_raw, "model", None)  # Lấy .model (nn.Sequential) từ bên trong EMA
        # --- KẾT THÚC SỬA ---

        if m is not None:
            self._apply_identity(m, True)
        if ema_model_seq is not None: self._apply_identity(ema_model_seq, True)  # <-- Dùng ema_model_seq
        if self.log:
            LOGGER.info("[STN] validation: identity")

    def on_val_end(self, validator):
        m = self.root(validator) # self.root() trả về trainer.model.model (đúng)

        # --- SỬA LẠI LOGIC TÌM EMA MODEL ---
        ema_m_raw = getattr(getattr(validator, "trainer", None), "ema", None)
        ema_model_raw = getattr(ema_m_raw, "ema", None)
        ema_model_seq = getattr(ema_model_raw, "model", None) # Lấy .model (nn.Sequential) từ bên trong EMA
        # --- KẾT THÚC SỬA ---

        if m is None:
            return
        if self._mode == "blend":
            self._apply_identity(m, False)
            if ema_model_seq: self._apply_identity(ema_model_seq, False) # <-- Dùng ema_model_seq
            self._apply_blend(m, self._alpha)
            if ema_model_seq: self._apply_blend(ema_model_seq, self._alpha) # <-- Dùng ema_model_seq
            if self.log:
                LOGGER.info(f"[STN] restore blend α={self._alpha:.3f}")

class ForceSTNIdentityOnVal:  # BC wrapper
    def __init__(self, log=False): self.ctrl = STNControl(log=log)
    def on_val_start(self, v): self.ctrl.on_val_start(v)
    def on_val_end(self, v):   self.ctrl.on_val_end(v)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Theta plumbing (publish → ensure → wire)
# ──────────────────────────────────────────────────────────────────────────────
class PublishThetaToStateV2(_Ctx):
    def __init__(self, verbose=True):
        self.verbose = verbose; self._prev = {"train": {}, "val": {}}

    def _attach(self, owner, slot: str):
        root = self.root(owner);
        if root is None: return 0
        st = self._prev[slot]; cnt = 0
        S = self.state(owner)
        for m in self.stn_modules(root):
            if m in st: continue
            prev = getattr(m, "record_theta", None); st[m] = prev
            def rec(theta, _prev=prev):
                try:
                    _prev and _prev(theta)
                except Exception:
                    pass
                try:
                    # <<< THÊM LOG DEBUG >>>
                    theta_shape = theta.shape if hasattr(theta, "shape") else "N/A"
                    LOGGER.info(f"[PublishTheta DEBUG] Callback received theta with shape: {theta_shape}")
                    # <<< KẾT THÚC LOG DEBUG >>>
                    S["stn_theta"] = theta.detach() if hasattr(theta, "detach") else theta
                except Exception as e:  # Thêm exception handling cụ thể hơn
                    LOGGER.error(f"[PublishTheta DEBUG] Failed to set stn_theta in state: {e}")
                    S["stn_theta"] = theta  # Vẫn thử gán nếu detach lỗi
            setattr(m, "record_theta", rec); cnt += 1
        if self.verbose: LOGGER.info(f"[ThetaPub] attach[{slot}] -> {cnt}")
        return cnt

    def _detach(self, slot: str):
        st = self._prev[slot]; r = 0
        for m, prev in list(st.items()):
            try:
                if prev is not None: setattr(m, "record_theta", prev)
                elif hasattr(m, "record_theta"): delattr(m, "record_theta")
            except Exception: pass
            r += 1
        st.clear()
        if self.verbose: LOGGER.info(f"[ThetaPub] detach[{slot}] -> {r}")

    def on_train_start(self, t): self._attach(t, "train")
    def on_train_end(self, t):   self._detach("train")
    def on_val_start(self, v):   self._attach(v, "val")
    def on_val_end(self, v):     self._detach("val")

class EnsureThetaOnValBatch(_Ctx):
    def __init__(self, verbose=True): self.verbose = verbose
    def on_val_batch_start(self, validator):
        owner = getattr(validator, "trainer", validator); S = self.state(owner)
        batch = getattr(validator, "batch", None)
        imgs = (batch.get("img") if isinstance(batch, dict) else (batch[0] if batch else None))
        if imgs is None: return
        B = imgs.shape[0]; th = S.get("stn_theta")
        need = (th is None or not hasattr(th, "shape") or th.shape[0] != B or th.shape[-2:] != (2,3))
        if need:
            th = imgs.new_zeros((B,2,3)); th[:,0,0]=1.0; th[:,1,1]=1.0; S["stn_theta"] = th
        if self.verbose:
            bi = getattr(validator, "batch_i", -1); LOGGER.info(f"[ValTheta] batch={bi} B={B} fallback={need}")

class DumpModelWiringOnVal(_Ctx):
    def __init__(self, verbose=True, max_lines=64): self.verbose, self.max_lines = verbose, max_lines
    def on_val_start(self, validator):
        r = self.root(validator)
        if r is None: return
        stn, head = [], []
        for name, m in r.named_modules():
            t = type(m).__name__
            if self.is_stn(m): stn.append((name, t))
            if ("DetectSDTN" in t) or ("DetectSDTN" in name): head.append((name, t))
        if self.verbose:
            LOGGER.info(f"[WireDump] STN={len(stn)} HEAD={len(head)}")
            for tag, arr in ("STN", stn), ("HEAD", head):
                for i, (n, t) in enumerate(arr[: self.max_lines]): LOGGER.info(f"[WireDump] {tag}[{i}] {n} <{t}>")

class WireThetaToSDTNHead(_Ctx):
    def __init__(self, verbose=True): self.verbose, self._heads = verbose, []
    def on_val_start(self, v):
        self._heads.clear(); r = self.root(v)
        if r is None: return
        for name, m in r.named_modules():
            t = type(m).__name__
            if ("DetectSDTN" in t) or ("DetectSDTN" in name): self._heads.append((name, m))
        if self.verbose: LOGGER.info(f"[WireTheta] cached {len(self._heads)} heads")
    def on_val_batch_start(self, v):
        owner = getattr(v, "trainer", v); th = self.state(owner).get("stn_theta")
        if th is None: return
        wired = 0
        for name, m in self._heads:
            ok = False
            for api in ("set_theta", "receive_theta", "update_theta", "register_theta"):
                fn = getattr(m, api, None)
                if callable(fn):
                    try: fn(th); ok = True; break
                    except Exception as e: LOGGER.warning(f"[WireTheta] {name}.{api} failed: {e}")
            if not ok:
                for attr in ("theta", "theta_buffer", "_theta", "stn_theta"):
                    try: setattr(m, attr, th); ok = True; break
                    except Exception as e: LOGGER.warning(f"[WireTheta] set {name}.{attr} failed: {e}")
            wired += int(ok)
        if self.verbose:
            bi = getattr(v, "batch_i", -1); LOGGER.info(f"[WireTheta] batch={bi}: wired {wired}/{len(self._heads)}")

# ──────────────────────────────────────────────────────────────────────────────
# 3) Debugging visuals (kept compact)
# ──────────────────────────────────────────────────────────────────────────────
class DebugImages(_Ctx):
    def __init__(self, epochs=(0,5,10,15,20), max_images=5):
        self.epochs, self.max_images = set(int(e) for e in epochs), int(max_images)
        self.samples, self.dbg_dir = [], None

    @staticmethod
    def _grab_loader(trainer):
        return getattr(getattr(trainer, "validator", None), "dataloader", None) or trainer.train_loader

    def _cache_samples(self, trainer):
        for batch in self._grab_loader(trainer):
            imgs = batch["img"] if isinstance(batch, dict) else batch[0]
            tgts = batch if isinstance(batch, dict) else batch[1]
            idx0 = (tgts["batch_idx"] == 0)
            self.samples.append((imgs[0].cpu(), tgts["bboxes"][idx0].cpu(), tgts["cls"][idx0].cpu(), (tgts.get("im_file") or [None])[0]))
            if len(self.samples) >= self.max_images: break

    @staticmethod
    def _scale_xyxy(x, rx, ry):
        x = x.astype(np.float32).copy(); x[:,[0,2]]*=float(rx); x[:,[1,3]]*=float(ry); return x

    @staticmethod
    def _warp_boxes_xywh_with_theta(bxywh, W, H, th):
        if th is None:
            x,y,w,h = bxywh.unbind(-1)
            return torch.stack([x-w/2, y-h/2, x+w/2, y+h/2], 1)
        device, dtype = bxywh.device, bxywh.dtype
        x,y,w,h = bxywh.unbind(-1)
        x1,y1,x2,y2 = x-w/2, y-h/2, x+w/2, y+h/2
        Xs = torch.stack([x1,x2,x2,x1],1); Ys = torch.stack([y1,y1,y2,y2],1)
        pix2norm = lambda p,L: (p/max(L-1,1)*2.0)-1.0
        Xn, Yn = pix2norm(Xs,W), pix2norm(Ys,H)
        th = th.detach().to(torch.float32, device=device)
        A = torch.tensor([[th[0,0],th[0,1],th[0,2]],[th[1,0],th[1,1],th[1,2]],[0.0,0.0,1.0]], device=device)
        Ainv = torch.linalg.inv(A)
        ones = torch.ones_like(Xn)
        P_in = torch.stack([Xn, Yn, ones],1)
        P_out = torch.einsum("ij,njk->nik", Ainv, P_in)
        Xo, Yo = P_out[:,0,:], P_out[:,1,:]
        norm2pix = lambda p,L: (p+1.0)*0.5*(L-1)
        Xp, Yp = norm2pix(Xo,W), norm2pix(Yo,H)
        x1p,x2p = Xp.min(1).values, Xp.max(1).values
        y1p,y2p = Yp.min(1).values, Yp.max(1).values
        x1p = x1p.clamp(0, W-1); y1p = y1p.clamp(0, H-1)
        x2p = x2p.clamp(1, W);   y2p = y2p.clamp(1, H)
        return torch.stack([x1p,y1p,x2p,y2p],1)

    @smart_inference_mode()
    def on_train_epoch_end(self, trainer):
        ep = int(getattr(trainer, "epoch", 0))
        if ep not in self.epochs: return
        if self.dbg_dir is None:
            self.dbg_dir = os.path.join(str(trainer.save_dir), "stn_dbg"); os.makedirs(self.dbg_dir, exist_ok=True)
        if not self.samples: self._cache_samples(trainer)

        model = trainer.model; device = next(model.parameters()).device
        was_train = model.training; model.eval()
        # Lấy STNControl callback để biết trạng thái alpha hiện tại
        stn_control = None
        for cb in trainer.callbacks.get("on_train_epoch_start", []):
            if isinstance(cb, STNControl):  # STNControl được import ở đầu file
                stn_control = cb
                break

        # Áp dụng thủ công STN patch (blend/identity) CHÍNH XÁC cho việc visualize
        if stn_control:
            if stn_control._mode == "blend":
                stn_control._apply_identity(model, False)  # Tắt identity patch (nếu có)
                stn_control._apply_blend(model, stn_control._alpha)  # Áp dụng blend patch
                LOGGER.info(f"[DebugImages] Applied STN blend patch (α={stn_control._alpha:.3f}) for visualization.")
            else:  # mode == "identity"
                stn_control._apply_identity(model, True)  # Áp dụng identity patch
                LOGGER.info(f"[DebugImages] Applied STN identity patch for visualization.")
        H_panel = 672
        try:
            for i,(img0,bxywh,bcls,path0) in enumerate(self.samples):
                x = img0.clone().to(device).unsqueeze(0).float(); x = x/255.0 if x.max()>1 else x
                if hasattr(trainer, "state"): trainer.state.pop("stn_theta", None); trainer.state.pop("stn_out", None)
                _ = model(x); theta = getattr(trainer, "state", {}).get("stn_theta", None); stn_img = getattr(trainer, "state", {}).get("stn_out", None)
                ori = img0.detach().cpu().numpy();
                if ori.ndim==3 and ori.shape[0] in (1,3): ori = np.transpose(ori, (1,2,0))
                if ori.max() <= 1.5: ori = (ori*255.0).round()
                ori = np.clip(ori,0,255).astype(np.uint8);
                if ori.ndim==2: ori = np.repeat(ori[...,None], 3, 2)
                if ori.shape[2]==1: ori = np.repeat(ori, 3, 2)
                Hs,Ws = ori.shape[:2]
                if torch.is_tensor(stn_img):
                    vis = stn_img[0].detach().cpu().numpy()
                    if vis.ndim==3 and vis.shape[0] in (1,3): vis = np.transpose(vis,(1,2,0))
                    vis = (vis*255.0).clip(0,255).astype(np.uint8)
                    if vis.ndim==2: vis = np.repeat(vis[...,None], 3, 2)
                    if vis.shape[2]==1: vis = np.repeat(vis, 3, 2)
                else:
                    vis = ori.copy()
                def rz(im,H=H_panel):
                    h,w = im.shape[:2]
                    if h==H: return im
                    return cv2.resize(im, (int(round(w*H/float(h))), H), interpolation=cv2.INTER_LINEAR)
                L,R = rz(ori), rz(vis)
                WL,WR = L.shape[1], R.shape[1]
                bxywh_src = (bxywh.detach().cpu().numpy() * np.array([Ws,Hs,Ws,Hs], np.float32))
                xyxy_src = xywh2xyxy(torch.from_numpy(bxywh_src)).numpy().astype(np.float32)
                xyxy_left  = self._scale_xyxy(xyxy_src, float(WL)/Ws, float(H_panel)/Hs).round().clip(0,1e9).astype(int)
                xyxy_right = xyxy_left.copy()
                if isinstance(theta, torch.Tensor):
                    th = theta[0] if (theta.dim()==3 and theta.shape[0]>=1) else (theta if theta.dim()==2 else None)
                    if th is not None:
                        with torch.no_grad():
                            tb = torch.from_numpy(bxywh_src).to(torch.float32)
                            tbw = self._warp_boxes_xywh_with_theta(tb, Ws, Hs, th).cpu().numpy().astype(np.float32)
                            xyxy_right = self._scale_xyxy(tbw, float(WR)/Ws, float(H_panel)/Hs).round().clip(0,1e9).astype(int)
                def draw(img, boxes):
                    out = img.copy(); labels = bcls.view(-1).tolist()
                    for (x1,y1,x2,y2),c in zip(boxes, labels):
                        x1=int(max(0,min(img.shape[1]-1,x1))); y1=int(max(0,min(img.shape[0]-1,y1)))
                        x2=int(max(1,min(img.shape[1],x2)));   y2=int(max(1,min(img.shape[0],y2)))
                        col = CLASS_COLORS[int(c)%len(CLASS_COLORS)]
                        cv2.rectangle(out,(x1,y1),(x2,y2),col,2)
                        out = self.put_text(out, CLASS_LABELS.get(int(c), str(int(c))), (x1+4, max(0,y1-22)), col, 28)
                    return out
                Ld,Rd = draw(L, xyxy_left), draw(R, xyxy_right)
                Ld = self.put_text(Ld, "ORIGINAL", (10,10), (255,255,0), 42)
                Rd = self.put_text(Rd, "STN",      (10,10), (255,255,0), 42)
                t = (theta[0] if (isinstance(theta, torch.Tensor) and theta.dim()==3) else theta)
                if not isinstance(t, torch.Tensor) or t.numel()==0:
                    t = torch.tensor([[1,0,0],[0,1,0]], dtype=torch.float32)
                t = t.detach().cpu().view(2,3).numpy()
                Rd = self.put_text(Rd, f"θ0: {t[0, 0]:+0.5f} {t[0, 1]:+0.5f} {t[0, 2]:+0.5f}", (10, 60),
                                   (120, 255, 120), 28)
                Rd = self.put_text(Rd, f"θ1: {t[1, 0]:+0.5f} {t[1, 1]:+0.5f} {t[1, 2]:+0.5f}", (10, 94),
                                   (120, 255, 120), 28)
                dx = float(t[0, 2]) * (Ws / 2.0)  # ~ pixel theo chiều ngang
                dy = float(t[1, 2]) * (Hs / 2.0)  # ~ pixel theo chiều dọc
                Rd = self.put_text(Rd, f"Δt ≈ ({dx:+.2f}px, {dy:+.2f}px)", (10, 128), (180, 220, 255), 26)

                both = np.concatenate([Ld,Rd],1); pad=16
                Hh,Ww = both.shape[:2]; canvas = np.full((Hh+2*pad, Ww+2*pad, 3), (40,40,40), np.uint8)
                canvas[pad:pad+Hh, pad:pad+Ww] = both
                base = os.path.splitext(os.path.basename(str(path0) or f"sample{i:02d}"))[0]
                out = os.path.join(self.dbg_dir, f"{base}_epoch_{ep:03d}_{i:02d}.png"); cv2.imwrite(out, canvas)
        finally:
            model.train(was_train)
        LOGGER.info(f"[DebugImages] saved for epoch {ep}")

# ──────────────────────────────────────────────────────────────────────────────
# 4) Quick dataloader peek
# ──────────────────────────────────────────────────────────────────────────────
class PeekBatch(_Ctx):
    def __init__(self, save_first_n=1, max_images=4, panels=2, save_dir_name="peek_batch"):
        self.save_first_n, self.max_images, self.panels, self.save_dir_name = save_first_n, max_images, panels, save_dir_name
        self._done = 0
    def _names(self, trainer):
        return getattr(trainer, "names", None) or getattr(getattr(trainer, "validator", None), "names", None) or getattr(getattr(trainer, "model", None), "names", None)
    def _pack_like_loader(self, items):
        imgs = torch.stack([torch.as_tensor(it["img"]) for it in items], 0)
        bxs, cls, bidx = [], [], []
        for i,it in enumerate(items):
            bi,ci = it.get("bboxes"), it.get("cls")
            if torch.is_tensor(bi) and bi.ndim==2 and bi.size(-1)==4:
                bxs.append(bi); bidx.append(torch.full((bi.shape[0],), i))
            if torch.is_tensor(ci): cls.append(ci)
        bxs  = torch.cat(bxs,0)  if bxs  else torch.zeros((0,4))
        cls  = torch.cat(cls,0)  if cls  else torch.zeros((0,))
        bidx = torch.cat(bidx,0) if bidx else torch.zeros((0,))
        paths = [it.get("im_file", it.get("path","")) for it in items]
        return {"img":imgs, "bboxes":bxs, "cls":cls, "batch_idx":bidx, "im_file":paths}
    def _save_grid(self, trainer, batch, img_ids, out):
        imgs = batch["img"].detach().float().cpu(); imgs = imgs/255.0 if imgs.max()>1.5 else imgs
        sel = torch.tensor(img_ids, dtype=torch.long); imgs_sub = imgs.index_select(0, sel)
        labels = None
        if all(k in batch for k in ("batch_idx","cls","bboxes")):
            bidx = batch["batch_idx"].long().cpu(); mask = torch.isin(bidx, sel)
            if mask.any():
                old = bidx[mask]; mapping = {int(v): i for i,v in enumerate(img_ids)}
                new_bidx = torch.tensor([mapping[int(v)] for v in old], dtype=torch.float32).unsqueeze(1)
                cls = batch["cls"][mask].float().unsqueeze(1).floor(); bxs = batch["bboxes"][mask].float()
                labels = torch.cat((new_bidx, cls, bxs), 1).cpu()
        paths = batch.get("im_file", batch.get("paths", None))
        if isinstance(paths,(list,tuple)) and len(paths)==imgs.shape[0]: paths=[paths[i] for i in img_ids]
        plot_images(images=imgs_sub, labels=labels, paths=paths, names=self._names(trainer), fname=str(out), max_subplots=len(img_ids))
    def on_fit_epoch_start(self, trainer):
        if self._done >= self.save_first_n: return
        self._done += 1
        ds = getattr(getattr(trainer, "train_loader", None), "dataset", None)
        if ds is None: return
        bs = int(getattr(trainer, "batch_size", self.max_images*self.panels)); need = min(bs, self.max_images*self.panels)
        try: idxs = np.random.choice(len(ds), need, replace=False).tolist()
        except Exception: idxs = list(range(min(need, len(ds))))
        items = [ds[i] for i in idxs]
        batch = ds.collate_fn(items) if hasattr(ds, "collate_fn") else self._pack_like_loader(items)
        save_dir = Path(str(trainer.save_dir))/self.save_dir_name; save_dir.mkdir(parents=True, exist_ok=True)
        meta = [f"[peek] epoch={trainer.epoch} imgsz={getattr(trainer,'imgsz',None)} (synthetic bs~{bs}, dumped={need})"]
        for k in sorted(batch.keys()):
            v = batch[k]
            if torch.is_tensor(v):
                v = v.detach(); meta.append(f"{k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
            elif isinstance(v, (list,tuple)): meta.append(f"{k}: {type(v).__name__} len={len(v)}")
            elif isinstance(v, dict): meta.append(f"{k}: dict keys={list(v.keys())[:10]}")
            else: meta.append(f"{k}: {type(v).__name__}")
        (save_dir/f"batch_epoch{int(trainer.epoch):03d}.txt").write_text("\n".join(meta), encoding="utf-8")
        nimg = int(batch["img"].shape[0]); take = min(nimg, self.max_images*self.panels)
        cursor, pid = 0, 0
        while cursor < take and pid < self.panels:
            end = min(cursor+self.max_images, take); ids = list(range(cursor, end))
            out = save_dir/f"batch_epoch{int(trainer.epoch):03d}_p{pid}.jpg"; self._save_grid(trainer, batch, ids, out)
            cursor, pid = end, pid+1

# ──────────────────────────────────────────────────────────────────────────────
# 5) SupCon: inject + schedule + mirror (kept API-compatible)
# ──────────────────────────────────────────────────────────────────────────────
class InjectSupConArgsMinimal:
    def __init__(self, **cfg): self.cfg = dict(cfg)
    def _ensure_ns(self, obj, attr):
        val = getattr(obj, attr, None)
        if val is None or isinstance(val, dict): val = SimpleNamespace(**(val or {})); setattr(obj, attr, val)
        return val
    def on_pretrain_routine_start(self, trainer): trainer._supcon_cfg = SimpleNamespace(**self.cfg)
    def on_pretrain_routine_end(self, trainer):
        ma = self._ensure_ns(trainer.model, "args"); [setattr(ma, k, v) for k,v in self.cfg.items()]
        crit = getattr(trainer.model, "criterion", None)
        if crit is not None:
            hyp = self._ensure_ns(crit, "hyp"); [setattr(hyp, k, v) for k,v in self.cfg.items()]
        trainer._supcon_cfg = SimpleNamespace(**self.cfg)
    def on_train_start(self, trainer):
        cache = getattr(trainer, "_supcon_cfg", None)
        cfg = vars(cache) if isinstance(cache, SimpleNamespace) else self.cfg
        ma = self._ensure_ns(trainer.model, "args"); [setattr(ma, k, v) for k,v in cfg.items()]
        crit = getattr(trainer.model, "criterion", None)
        if crit is not None:
            hyp = self._ensure_ns(crit, "hyp"); [setattr(hyp, k, v) for k,v in cfg.items()]

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
                a,b = tok.split("-",1); a=a.strip(); b=b.strip()
                lo = 1 if a=="" else int(a); hi = 10**9 if b=="" else int(b)
            else:
                lo = hi = int(tok)
            out.append((lo,hi))
        return out
    def _on(self, e_disp: int) -> int:
        if not self.ranges: return self.default_on
        for lo,hi in self.ranges:
            if lo <= e_disp <= hi: return 1
        return 0
    def on_train_epoch_start(self, trainer):
        e = int(getattr(trainer, "epoch", 0)) + 1
        want = self._on(e)
        ma = getattr(trainer.model, "args", None)
        if ma is None or isinstance(ma, dict): ma = SimpleNamespace(**(ma or {})); trainer.model.args = ma
        setattr(ma, "supcon_on", int(want))
        if want==1 and hasattr(trainer, "loss_names"):
            if not trainer.loss_names or trainer.loss_names[-1] != "supcon_loss":
                trainer.loss_names = ("box_loss","cls_loss","dfl_loss","supcon_loss")
        LOGGER.info(f"[SupCon/schedule] e={e} -> on={want}")

class ReinforceSupConToLoss:
    def __init__(self, keys): self.keys = tuple(keys)
    def on_train_epoch_start(self, trainer):
        ma = getattr(trainer.model, "args", None); crit = getattr(trainer.model, "criterion", None)
        if ma is None or crit is None: return
        if not hasattr(crit, "hyp") or crit.hyp is None: crit.hyp = SimpleNamespace()
        for k in self.keys:
            if hasattr(ma, k): setattr(crit.hyp, k, getattr(ma, k))
        try:
            sc_on = int(getattr(ma, "supcon_on", 0) or 0)
            if sc_on==1 and hasattr(trainer, "loss_names") and trainer.loss_names[-1] != "supcon_loss":
                trainer.loss_names = ("box_loss","cls_loss","dfl_loss","supcon_loss")
        except Exception: pass

class LinkTrainerToLoss:
    def on_train_start(self, trainer):
        if getattr(trainer, "loss", None) is not None: trainer.loss._trainer = trainer

class SyncEpochToLoss:
    def on_train_epoch_start(self, trainer):
        if getattr(trainer, "loss", None) is not None: trainer.loss.epoch = int(trainer.epoch)
    def on_train_batch_start(self, trainer):
        if getattr(trainer, "loss", None) is not None: trainer.loss.epoch = int(trainer.epoch)

class TapSTNFeat:
    def __init__(self, out_idx: int|None=None, out_name: str|None=None):
        self.out_idx, self.out_name, self.h, self.latest, self.where = out_idx, (out_name or None), None, None, None
    def _cfg(self, trainer, key, default=None):
        cfg = getattr(trainer, "_supcon_cfg", None)
        if cfg is None: return default
        return (cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default))
    def _hook(self, module, inputs, output):
        if torch.is_tensor(output): self.latest = F.adaptive_avg_pool2d(output, (1,1)).flatten(1)
    def _resolve(self, model, idx, name):
        modlist = getattr(model, "model", None)
        if modlist is None: return None, None
        if isinstance(idx, int) and 0 <= idx < len(modlist): return modlist[idx], f"model.model[{idx}]"
        if name:
            low = str(name).lower()
            for i,m in enumerate(modlist):
                if low in m.__class__.__name__.lower(): return m, f"model.model[{i}]<{m.__class__.__name__}>"
        for i,m in enumerate(modlist):
            if hasattr(m, "forward"): return m, f"model.model[{i}]<{m.__class__.__name__}>"
        return None, None
    def on_train_start(self, trainer, *_, **__):
        idx = self._cfg(trainer, "supcon_out", self.out_idx); name = self._cfg(trainer, "supcon_name", self.out_name)
        m, where = self._resolve(trainer.model, idx, (name.strip() or None) if isinstance(name, str) else None)
        if m is None: LOGGER.info("[TapSTNFeat] no layer"); return
        if self.h is not None:
            try: self.h.remove()
            except Exception: pass
        self.h = m.register_forward_hook(self._hook); self.where = where
        LOGGER.info(f"[TapSTNFeat] attach at {where}")
    def on_train_end(self, *a, **k):
        if self.h is not None:
            try: self.h.remove()
            except Exception: pass
        self.h = None; self.latest = None; self.where = None
    def on_train_batch_end(self, *a, **k): self.latest = None

class SupConPercentLogger:
    def __init__(self): self.num = 0.0; self.den = 0.0
    def _read(self, t):
        n,v = getattr(t, "loss_names", None), getattr(t, "tloss", None)
        if n and v is not None:
            vals = v.detach().cpu().tolist() if torch.is_tensor(v) else list(v)
            m = {k: float(x) for k,x in zip(n, vals)}; return m.get("supcon_loss", 0.0), sum(m.values())
        return None, None
    def on_train_batch_end(self, t):
        s,d = self._read(t)
        if s is not None: self.num += s; self.den += max(d, 1e-9)
    def on_train_epoch_end(self, t):
        if self.den>0: LOGGER.info(f"[SupCon%] epoch {t.epoch+1}: ≈ {100*self.num/self.den:.1f}%")
        self.num = self.den = 0.0

_SUPCON_PROJ_GLOBAL = None

def supcon_register_projector(proj):
    global _SUPCON_PROJ_GLOBAL; _SUPCON_PROJ_GLOBAL = proj; LOGGER.info("[SupConProj] registered")

def _supcon_get_global_projector(): return _SUPCON_PROJ_GLOBAL


class AttachSupConProjToOptim:
    def __init__(self):
        self.attached = False

    def _resolve(self, t):
        cand = []
        crit = getattr(t, "criterion", None);
        loss = getattr(t, "loss", None)
        if crit is not None: cand += [getattr(crit, k, None) for k in
                                      ("_supcon_proj", "supcon_projector", "projector", "supcon_proj")]
        if loss is not None: cand += [getattr(loss, k, None) for k in
                                      ("_supcon_proj", "supcon_projector", "projector", "supcon_proj")]
        cand += [getattr(t, k, None) for k in ("_supcon_proj", "supcon_projector", "supcon_proj")]
        model = getattr(t, "model", None)
        if model is not None: cand += [getattr(model, k, None) for k in
                                       ("_supcon_proj", "supcon_projector", "supcon_proj")]
        cand.append(_supcon_get_global_projector())  # Giả định hàm _supcon_get_global_projector() tồn tại
        for c in cand:
            if c is not None: return c
        return None

    def _attach(self, t, proj, when: str):
        opt = getattr(t, "optimizer", None)
        if opt is None: LOGGER.info(f"[SupConProj] no optimizer ({when})"); return False
        if proj is None: return False
        params = [p for p in proj.parameters() if p.requires_grad]
        if not params: LOGGER.info("[SupConProj] no trainable params"); return False
        old_lrs = [g.get("lr", None) for g in opt.param_groups]
        old_inits = [g.get("initial_lr", None) for g in opt.param_groups]
        base_init = float(getattr(getattr(t, 'args', SimpleNamespace()), 'lr0', 0.001))
        base_lr, base_wd = base_init, 0.0
        target_pg = None;
        proj_ids = {id(p) for p in params}
        for g in opt.param_groups:
            if proj_ids & {id(p) for p in g.get("params", [])}: target_pg = g; break

        def fill(pg):
            d = getattr(opt, "defaults", {}) or {}
            for k, v in d.items(): pg.setdefault(k, v)
            pg.setdefault("initial_lr", pg.get("lr", base_init))
            pg.setdefault("weight_decay", base_wd)
            pg.setdefault("name", "supcon_proj");
            return pg

        if target_pg is None:
            new_pg = fill(
                {"params": list(params), "lr": float(base_lr), "weight_decay": 0.0, "initial_lr": float(base_init),
                 "name": "supcon_proj"})
            opt.add_param_group(new_pg)
            LOGGER.info(f"[SupConProj] ADDED (when={when}) | n_params={sum(p.numel() for p in params)} | lr={base_lr}")
        else:
            existed = {id(p) for p in target_pg["params"]}
            target_pg["params"].extend([p for p in params if id(p) not in existed])
            fill(target_pg)
            if float(target_pg.get("lr", 0.0)) == 0.0: target_pg["lr"] = float(base_lr)
            LOGGER.info(f"[SupConProj] REUSED (when={when}) | lr={target_pg['lr']}")
        for i, g in enumerate(opt.param_groups):
            if g.get("name", "") == "supcon_proj": continue
            if old_lrs[i] is not None: g["lr"] = old_lrs[i]
            if old_inits[i] is not None: g["initial_lr"] = old_inits[i]
        try:
            supcon_register_projector(proj)  # Giả định hàm supcon_register_projector() tồn tại
        except Exception:
            pass

        # === BẮT ĐẦU SỬA LỖI: TẠO LẠI GRADSCALER ===
        # Sau khi thay đổi optimizer (add_param_group), chúng ta PHẢI
        # tạo lại GradScaler nếu AMP đang bật, nếu không sẽ bị lỗi AssertionError.
        # 't' chính là đối tượng `trainer`.
        if getattr(t, 'amp', False):
            LOGGER.warning("[SupCon/Optim] Optimizer changed. Re-initializing GradScaler...")
            # (Đảm bảo 'import torch' đã có ở đầu file stn_utils.py)
            t.scaler = torch.cuda.amp.GradScaler(enabled=t.amp)
        # === KẾT THÚC SỬA LỖI ===

        return True

    def _try(self, t, when):
        if self.attached: return
        proj = self._resolve(t)
        if proj is None: return
        if self._attach(t, proj, when):
            self.attached = True
            # === BẮT ĐẦU SỬA LỖI: HỦY CALLBACK ===
            # Sau khi gắn thành công, hủy callback để nó không chạy nữa
            LOGGER.info(f"[SupConProj] Successfully attached. Removing callback.")
            for cb_list_name in ("on_train_start", "on_train_batch_start", "on_train_batch_end"):
                t.remove_callback(cb_list_name, getattr(self, cb_list_name, None))
            # === KẾT THÚC SỬA LỖI ===

    def on_train_start(self, t, *a, **k):
        self._try(t, "on_train_start")

    def on_train_batch_start(self, t, *a, **k):
        if not self.attached: self._try(t, "on_train_batch_start")

    def on_train_batch_end(self, t, *a, **k):
        if not self.attached: self._try(t, "on_train_batch_end")

# ──────────────────────────────────────────────────────────────────────────────
# 6) NaN guard + light batch sanity
# ──────────────────────────────────────────────────────────────────────────────
class LossNaNGuard:
    def __init__(self, stop_on_nan=True, save_bad_batch=True): self.stop_on_nan, self.save_bad_batch = stop_on_nan, save_bad_batch
    def on_train_batch_end(self, trainer, *_, **__):
        items = getattr(trainer, "loss_items", None)
        if items is None: return
        def f(v):
            if isinstance(v, float): return v
            if torch.is_tensor(v): return float(v.detach().item())
            try: return float(v)
            except Exception: return float("nan")
        vals = {k: f(v) for k,v in (items.items() if isinstance(items, dict) else zip(["box_loss","cls_loss","dfl_loss","supcon_loss"], items))}
        bad = {k:v for k,v in vals.items() if (math.isnan(v) or math.isinf(v))}
        if not bad: return
        e = getattr(trainer, "epoch", -1); step = getattr(trainer, "batch_i", getattr(trainer, "ni", -1))
        LOGGER.error(f"[NaNGuard] epoch={e} step={step} NaN/Inf in {list(bad.keys())}")
        batch = getattr(trainer, "batch", None)
        if self.save_bad_batch and isinstance(batch, dict) and ("img" in batch):
            try:
                save_dir = Path(getattr(trainer, "save_dir", Path(".")))
                fname = save_dir/f"nan_batch_e{e:03d}_i{int(step):06d}.jpg"; plot_images(images=batch["img"], batch=batch, fname=fname)
                LOGGER.error(f"[NaNGuard] saved bad batch -> {fname}")
            except Exception as ex: LOGGER.error(f"[NaNGuard] save-batch failed: {ex}")
        # enrich context trước khi dừng
        try:
            B = getattr(trainer, "batch", None)
            imgs = (B.get("img") if isinstance(B, dict) else (B[0] if B else None))

            def _mm(x):
                try:
                    return (float(x.min().item()), float(x.max().item()))
                except Exception:
                    return (None, None)

            img_minmax = _mm(imgs) if hasattr(imgs, "dtype") else (None, None)
            lr = None
            try:
                for g in trainer.optimizer.param_groups:
                    lr = g.get("lr", None);
                    break
            except Exception:
                pass
            # loss_items nếu có
            li = getattr(trainer, "loss_items", None)
            li = [float(v) for v in li] if li is not None else None
            LOGGER.error(f"[NaNGuard/ctx] img_minmax={img_minmax} lr={lr} loss_items={li}")
        except Exception:
            pass

        if self.stop_on_nan:
            raise RuntimeError("[NaNGuard] Stop due to NaN")

class BatchSanityFilter:
    def __init__(self, eps: float = 1e-6): self.eps = float(eps)
    def _extract(self, trainer, *args, **kwargs):
        batch = kwargs.get("batch", None) or (args[0] if len(args)>=1 else getattr(trainer, "batch", None))
        return batch
    def on_train_batch_start(self, trainer, *args, **kwargs):
        batch = self._extract(trainer, *args, **kwargs)
        if batch is None: return
        for k in ("img","bboxes","cls"):
            v = batch.get(k, None) if isinstance(batch, dict) else None
            if torch.is_tensor(v): batch[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if isinstance(batch, dict) and torch.is_tensor(batch.get("bboxes", None)):
            bb = batch["bboxes"]
            if torch.all(bb.abs() <= 2.0): batch["bboxes"] = torch.clamp(bb, 0.0 - self.eps, 1.0 + self.eps)

# ──────────────────────────────────────────────────────────────────────────────
# 7) BG pairing preview (compact)
# ──────────────────────────────────────────────────────────────────────────────
class DebugBgPairROIs:
    def __init__(self, epochs=(0,1,2,5,10), max_pairs=6): self.epochs, self.max_pairs = set(map(int,epochs)), int(max_pairs)
    def _label_path(self, p: str) -> str|None:
        q = Path(p); lbl = Path(str(q).replace(os.sep+"images"+os.sep, os.sep+"labels"+os.sep)).with_suffix(".txt")
        return str(lbl) if lbl.exists() else None
    def _read_yolo_xyxy(self, lbl: str, W: int, H: int):
        out = []
        if not lbl or not os.path.exists(lbl): return out
        for line in Path(lbl).read_text(encoding="utf-8").splitlines():
            s = line.strip().split()
            if len(s) >= 5:
                c = int(float(s[0])); cx,cy,bw,bh = map(float, s[1:5])
                x1,y1 = int(max(0,(cx-bw/2)*W)), int(max(0,(cy-bh/2)*H))
                x2,y2 = int(min(W-1,(cx+bw/2)*W)), int(min(H-1,(cy+bh/2)*H))
                out.append((x1,y1,x2,y2,c))
        return out
    def _draw(self, img, boxes, color=(0,0,255), thick=2, tag=None, names=None, per_cls=False):
        out = img.copy()
        for (x1,y1,x2,y2,ci) in boxes:
            col = (CLASS_COLORS[int(ci)%len(CLASS_COLORS)] if per_cls else color)
            cv2.rectangle(out,(x1,y1),(x2,y2),col,thick)
            name = (names.get(int(ci)) if names and int(ci) in names else CLASS_LABELS.get(int(ci), str(int(ci))))
            cv2.putText(out, name, (x1+4, max(18,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        if tag: cv2.putText(out, tag, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2, cv2.LINE_AA)
        return out
    def on_train_epoch_end(self, trainer):
        e = int(getattr(trainer, "epoch", 0))
        if e not in self.epochs: return
        try:
            batch = next(iter(trainer.train_loader))
        except Exception as ex:
            LOGGER.warning(f"[DebugBgPairROIs] cannot fetch batch: {ex}"); return
        pair_idx, abn_mask, im_files = batch.get("pair_idx"), batch.get("abn_mask"), batch.get("im_files")
        if pair_idx is None or abn_mask is None or im_files is None:
            LOGGER.warning("[DebugBgPairROIs] missing pair_idx/abn_mask/im_files"); return
        save_dir = Path(str(trainer.save_dir))/"bgpair_preview"; save_dir.mkdir(parents=True, exist_ok=True)
        rows, take = [], 0
        for (i,j) in pair_idx.tolist():
            if take >= self.max_pairs: break
            fg_i, bg_j = (i,j) if bool(abn_mask[i]) and not bool(abn_mask[j]) else ((j,i) if bool(abn_mask[j]) and not bool(abn_mask[i]) else (None,None))
            if fg_i is None: continue
            Lp, Rp = im_files[fg_i], im_files[bg_j]
            if not (Lp and Rp and os.path.exists(Lp) and os.path.exists(Rp)): continue
            L, R = cv2.imread(Lp, cv2.IMREAD_COLOR), cv2.imread(Rp, cv2.IMREAD_COLOR)
            if L is None or R is None: continue
            H,W = L.shape[:2]; fg = self._read_yolo_xyxy(self._label_path(Lp), W, H)
            H2,W2 = R.shape[:2]; bg = []
            for (x1,y1,x2,y2,ci) in fg:
                x1,y1,x2,y2 = max(0,min(W2-1,x1)), max(0,min(H2-1,y1)), max(1,min(W2,x2)), max(1,min(H2,y2))
                if x2>x1 and y2>y1: bg.append((x1,y1,x2,y2,ci))
            try:
                names = getattr(getattr(trainer, "model", None), "names", None) or getattr(getattr(trainer, "validator", None), "names", None)
            except Exception: names = None
            Ld = self._draw(L, fg, color=(0,0,255), tag="FG", names=names, per_cls=True)
            Rd = self._draw(R, bg, color=(0,255,255), tag="BG (pseudo bbox)", names=names, per_cls=False)
            h = max(Ld.shape[0], Rd.shape[0]); w = Ld.shape[1] + Rd.shape[1]
            row = np.zeros((h,w,3), np.uint8); row[:Ld.shape[0], :Ld.shape[1]] = Ld; row[:Rd.shape[0], Ld.shape[1]:] = Rd
            rows.append(row); take += 1
        if not rows: return
        w = max(r.shape[1] for r in rows); out_h = sum(r.shape[0] for r in rows)
        out = np.zeros((out_h,w,3), np.uint8); y = 0
        for r in rows: out[y:y+r.shape[0], :r.shape[1]] = r; y += r.shape[0]
        fp = save_dir/f"epoch_{e:03d}.jpg"; cv2.imwrite(str(fp), out); LOGGER.info(f"[DebugBgPairROIs] saved {fp} ({len(rows)} pairs)")

def attach_stn_loss_hooks(model, loss_obj):
    """
    Gắn hook cho mô hình để STN gửi theta cho loss và lấy feature map sau STN.
    - loss_obj: instance của v8DetectionLoss (hoặc class loss tương ứng) đã được khởi tạo.
    """
    stn_seen = False
    for m in model.modules():
        if isinstance(m, STN):
            # Gắn hàm record_theta để STN gọi, truyền theta về loss
            m.record_theta = loss_obj.set_theta
            stn_seen = True
            continue
        if stn_seen:
            # Gắn forward hook cho lớp ngay sau STN để lấy output feature map (cho SupCon)
            m.register_forward_hook(loss_obj._after_hook)
            # chỉ gắn hook đầu tiên sau STN
            break

def debug_val_sample(images, pred_scores, pred_bboxes, save_dir="runs_stn/val_debug", conf_thres=0.3):
    """
    Lưu ảnh val kèm box dự đoán (debug prediction).

    Args:
        images (Tensor): [B, 3, H, W]
        pred_scores (Tensor): [B, N, C]
        pred_bboxes (Tensor): [B, N, 4] - xyxy
        save_dir (str): Thư mục để lưu ảnh
        conf_thres (float): Ngưỡng confidence để lọc box
    """
    os.makedirs(save_dir, exist_ok=True)
    scores = pred_scores.sigmoid()
    mask = scores.max(2).values > conf_thres

    for i in range(images.shape[0]):
        img = images[i].detach().cpu()
        img_np = (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
        annotator = Annotator(img_np)
        for j in range(pred_bboxes.shape[1]):
            if not mask[i, j]:
                continue
            xyxy = pred_bboxes[i, j].detach().cpu().numpy().tolist()
            label = scores[i, j].argmax().item()
            annotator.box_label(xyxy, str(label))
        annotated_img = annotator.result()
        save_image(torch.from_numpy(annotated_img).permute(2, 0, 1).float() / 255, f"{save_dir}/val_dbg_{i}.jpg")

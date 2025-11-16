# stn_utils.py — merged (STN + SupCon + Train/Val helpers)
from __future__ import annotations
import os, math, random, json, time, csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFont, ImageDraw, Image

# Ultralytics
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import plot_images

# -----------------------------------------------------------------------------
# A) Hằng số & tiện ích vẽ chữ (dùng chung cho DebugImages/Preview)
# -----------------------------------------------------------------------------
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

def _put_text(img: np.ndarray, txt: str, pos: Tuple[int,int], col=(255,255,0), size=22) -> np.ndarray:
    pil = Image.fromarray(img); d = ImageDraw.Draw(pil)
    try: font = ImageFont.truetype(_FONT_PATH, size)
    except Exception: font = ImageFont.load_default()
    d.text(pos, txt, font=font, fill=col)
    return np.asarray(pil).copy()

# -----------------------------------------------------------------------------
# B) STN forward wrappers (pickle-friendly) + STNControl
# -----------------------------------------------------------------------------
try:
    from ultralytics.nn.modules.block import SpatialTransformer as _STN
except Exception:
    _STN = type("STN", (), {})  # fallback

def _stn_identity_forward(self, x, *a, **k):
    """Trả ảnh gốc; nếu có hook publish theta=I."""
    try:
        B = x.shape[0] if hasattr(x, "shape") else 1
        theta_I = x.new_tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0).repeat(B, 1, 1)
        if hasattr(self, "record_theta"):
            try: self.record_theta(theta_I)
            except Exception: pass
    except Exception:
        pass
    return x

def _stn_blend_forward(self, x, *a, **k):
    """Gọi forward gốc -> clamp theta (t, s) -> warp -> blend theo alpha."""
    a_   = float(getattr(self, "_stn_blend_alpha", 1.0))
    tmax = float(getattr(self, "_stn_tmax", 0.20))
    smin = float(getattr(self, "_stn_smin", 0.90))
    smax = float(getattr(self, "_stn_smax", 1.10))

    out = self._stn_forward_orig(x, *a, **k)
    x_t, theta = (out if isinstance(out, (tuple, list)) else (out, None)), None
    if isinstance(out, (tuple, list)):
        x_t = out[0]
        if len(out) >= 2: theta = out[1]

    try:
        if theta is not None and torch.is_tensor(theta):
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
            if hasattr(self, "record_theta"):
                try: self.record_theta(theta_safe)
                except Exception: pass
            x_t = x_stab
    except Exception:
        pass

    if x_t is None: x_t = x
    if a_ <= 0.0:  out_f = x
    elif a_ >= 1.0: out_f = x_t
    else:          out_f = x + a_ * (x_t - x)
    self._stn_mode = "blend"
    return out_f

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
        if model is None: return []
        for m in model.modules():
            if _Ctx.is_stn(m): yield m
    @staticmethod
    def state(owner):
        if not hasattr(owner, "state"): owner.state = {}
        return owner.state

class STNControl(_Ctx):
    """Đóng băng STN sớm + ép identity khi VAL + mở dần (blend/clamp) sau đó."""
    def __init__(self, freeze_epochs=0, stn_warmup=20, tmax=0.20, smin=0.90, smax=1.10, log=False):
        self.freeze_epochs = max(0, int(freeze_epochs))
        self.stn_warmup = max(0, int(stn_warmup))
        self.tmax, self.smin, self.smax = float(tmax), float(smin), float(smax)
        self.log, self._mode, self._alpha, self._epoch = bool(log), "identity", 0.0, -1

    def _alpha_for(self, e: int) -> float:
        if e < self.freeze_epochs: return 0.0
        if self.stn_warmup <= 0:  return 1.0
        k = (e - self.freeze_epochs) / float(self.stn_warmup)
        return float(max(0.0, min(1.0, k)))

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
                try: delattr(m, k)
                except Exception: pass

    def _patch_blend(self, m, alpha: float):
        import types
        self._ensure_orig_forward(m)
        m._stn_blend_alpha = float(max(0.0, min(1.0, alpha)))
        m._stn_tmax = float(self.tmax); m._stn_smin = float(self.smin); m._stn_smax = float(self.smax)
        m.forward = types.MethodType(_stn_blend_forward, m)
        m._stn_mode = "blend"

    def _apply_identity(self, model, on: bool):
        for m in self.stn_modules(model):
            if on: self._patch_identity(m)
            elif getattr(m, "_stn_forward_orig", None): m.forward = m._stn_forward_orig

    def _apply_blend(self, model, alpha: float):
        for m in self.stn_modules(model): self._patch_blend(m, alpha)

    def on_train_epoch_start(self, t):
        model = self.root(t);
        if model is None: return
        e = int(getattr(t, "epoch", 0)); self._epoch = e
        if e < self.freeze_epochs:
            self._apply_identity(model, True); self._mode, self._alpha = "identity", 0.0
            if self.log: LOGGER.info(f"[STN] identity @ epoch {e}")
        else:
            a = self._alpha_for(e)
            self._apply_identity(model, False); self._apply_blend(model, a)
            self._mode, self._alpha = "blend", a
            if self.log: LOGGER.info(f"[STN] blend α={a:.3f} (t≤{self.tmax:.2f}, s∈[{self.smin:.2f},{self.smax:.2f}])")

    def on_val_start(self, v):
        m = self.root(v)
        if m is not None: self._apply_identity(m, True)
        if self.log: LOGGER.info("[STN] validation: identity")

    def on_val_end(self, v):
        m = self.root(v);
        if m is None: return
        if self._mode == "blend":
            self._apply_identity(m, False); self._apply_blend(m, self._alpha)
            if self.log: LOGGER.info(f"[STN] restore blend α={self._alpha:.3f}")

# -----------------------------------------------------------------------------
# C) Theta plumbing (publish/capture/seed) + thống kê nhanh
# -----------------------------------------------------------------------------
class PublishThetaToStateV2(_Ctx):
    """Gắn vào STN.record_theta để publish theta vào trainer.state['stn_theta']."""
    def __init__(self, verbose=True): self.verbose = verbose; self._prev = {"train": {}, "val": {}}
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
                    S["stn_theta"] = theta.detach() if hasattr(theta, "detach") else theta
                except Exception:
                    S["stn_theta"] = theta
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

class ForceReturnTheta(_Ctx):
    def _set_flag(self, owner, on=True):
        r = self.root(owner)
        if r is None: return
        for m in self.stn_modules(r): setattr(m, "_force_return_theta", bool(on))
    def on_train_start(self, t): self._set_flag(t, True)
    def on_val_start(self, v):   self._set_flag(v, True)

class CaptureThetaFromSTN(_Ctx):
    """Phương án B để thu theta nếu STN gọi record_theta(theta)."""
    def __init__(self): self._prev = {}
    def on_train_start(self, t): self._attach(t)
    def on_val_start(self, v):   self._attach(v)
    def on_train_end(self, t):   self._detach(t)
    def on_val_end(self, v):     self._detach(v)
    def _attach(self, owner):
        r = self.root(owner);
        if r is None: return
        S = self.state(owner)
        for m in self.stn_modules(r):
            if m in self._prev: continue
            prev = getattr(m, "record_theta", None); self._prev[m] = prev
            def rec(theta, _prev=prev):
                try: _prev and _prev(theta)
                except Exception: pass
                S["stn_theta"] = theta.detach() if hasattr(theta, "detach") else theta
            setattr(m, "record_theta", rec)
    def _detach(self, owner):
        for m, prev in list(self._prev.items()):
            try:
                if prev is not None: setattr(m, "record_theta", prev)
                elif hasattr(m, "record_theta"): delattr(m, "record_theta")
            except Exception: pass
        self._prev.clear()

class SeedIdentityThetaOnValStart(_Ctx):
    def __init__(self, B_hint=8): self.B_hint = int(B_hint)
    def on_val_start(self, v):
        S = self.state(getattr(v, "trainer", v))
        B = self.B_hint
        th = torch.zeros((B,2,3), dtype=torch.float32); th[:,0,0]=1.0; th[:,1,1]=1.0
        S["stn_theta"] = th

class ThetaStats(_Ctx):
    def __init__(self, every=1, tag="train"): self.every, self.tag, self._i = max(1,int(every)), tag, 0
    def on_train_batch_end(self, t):
        if self.tag!="train": return
        self._tick(t)
    def on_val_batch_end(self, v):
        if self.tag!="val": return
        self._tick(getattr(v,"trainer",v))
    def _tick(self, owner):
        self._i += 1
        if (self._i % self.every) != 0: return
        S = self.state(owner); th = S.get("stn_theta", None)
        if isinstance(th, torch.Tensor):
            tmin, tmax = float(th.min()), float(th.max())
            LOGGER.info(f"[θ/{self.tag}] shape={tuple(th.shape)} min={tmin:.4f} max={tmax:.4f}")

# -----------------------------------------------------------------------------
# D) SupCon: inject/schedule/reinforce/link/tap/percent/proj attach
# -----------------------------------------------------------------------------
_SUPCON_PROJ_GLOBAL = None
def supcon_register_projector(proj):
    global _SUPCON_PROJ_GLOBAL; _SUPCON_PROJ_GLOBAL = proj; LOGGER.info("[SupConProj] registered")
def _supcon_get_global_projector(): return _SUPCON_PROJ_GLOBAL

class InjectSupConArgsMinimal:
    def __init__(self, **cfg): self.cfg = dict(cfg)
    def _ensure_ns(self, obj, attr):
        val = getattr(obj, attr, None)
        if val is None or isinstance(val, dict): val = SimpleNamespace(**(val or {})); setattr(obj, attr, val)
        return val
    def on_pretrain_routine_start(self, t): t._supcon_cfg = SimpleNamespace(**self.cfg)
    def on_pretrain_routine_end(self, t):
        ma = self._ensure_ns(t.model, "args"); [setattr(ma, k, v) for k, v in self.cfg.items()]
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
    def on_train_epoch_start(self, t):
        e = int(getattr(t, "epoch", 0)) + 1
        want = self._on(e)
        ma = getattr(t.model, "args", None)
        if ma is None or isinstance(ma, dict): ma = SimpleNamespace(**(ma or {})); t.model.args = ma
        setattr(ma, "supcon_on", int(want))
        if want==1 and hasattr(t, "loss_names"):
            if not t.loss_names or t.loss_names[-1] != "supcon_loss":
                t.loss_names = ("box_loss","cls_loss","dfl_loss","supcon_loss")
        LOGGER.info(f"[SupCon/schedule] e={e} -> on={want}")

class ReinforceSupConToLoss:
    def __init__(self, keys): self.keys = tuple(keys)

    def on_train_epoch_start(self, t):
        ma = getattr(t.model, "args", None); crit = getattr(t.model, "criterion", None)
        if ma is None or crit is None: return
        if not hasattr(crit, "hyp") or crit.hyp is None: crit.hyp = SimpleNamespace()
        for k in self.keys:
            if hasattr(ma, k): setattr(crit.hyp, k, getattr(ma, k))
        try:
            sc_on = int(getattr(ma, "supcon_on", 0) or 0)
            if sc_on==1 and hasattr(t, "loss_names") and t.loss_names[-1] != "supcon_loss":
                t.loss_names = ("box_loss","cls_loss","dfl_loss","supcon_loss")
        except Exception:
            pass

    # ✅ thêm hook này để chắc chắn VAL không dùng SupCon
    def on_val_start(self, t):
        crit = getattr(t.model, "criterion", None)
        if crit is not None and hasattr(crit, "hyp"):
            setattr(crit.hyp, "supcon_on", 0)


class LinkTrainerToLoss:
    def on_train_start(self, t):
        if getattr(t, "loss", None) is not None: t.loss._trainer = t

class SyncEpochToLoss:
    def on_train_epoch_start(self, t):
        if getattr(t, "loss", None) is not None: t.loss.epoch = int(t.epoch)
    def on_train_batch_start(self, t):
        if getattr(t, "loss", None) is not None: t.loss.epoch = int(t.epoch)

class TapSTNFeat:
    def __init__(self, out_idx: int|None=None, out_name: str|None=None):
        self.out_idx, self.out_name, self.h, self.latest, self.where = out_idx, (out_name or None), None, None, None
    def _cfg(self, t, key, default=None):
        cfg = getattr(t, "_supcon_cfg", None)
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
    def on_train_start(self, t, *_, **__):
        idx = self._cfg(t, "supcon_out", self.out_idx); name = self._cfg(t, "supcon_name", self.out_name)
        m, where = self._resolve(t.model, idx, (name.strip() or None) if isinstance(name, str) else None)
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
        if self.den>0: LOGGER.info(f"[SupCon%] epoch {t.epoch+1}:  {100*self.num/self.den:.1f}%")
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
    def on_train_start(self, t, *a, **k):         self._try(t, "on_train_start")
    def on_train_batch_start(self, t, *a, **k):   self._try(t, "on_train_batch_start")
    def on_train_batch_end(self, t, *a, **k):     self._try(t, "on_train_batch_end")

# -----------------------------------------------------------------------------
# E) Safety: NaN guard + Batch sanitation + LRGuard
# -----------------------------------------------------------------------------
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
        vals = {k: f(v) for k,v in (items.items() if isinstance(items, dict) else zip(["box_loss","cls_loss","dfl_loss","supcon_loss"], items))}
        bad = {k:v for k,v in vals.items() if (math.isnan(v) or math.isinf(v))}
        if not bad: return
        e = getattr(t, "epoch", -1); step = getattr(t, "batch_i", getattr(t, "ni", -1))
        LOGGER.error(f"[NaNGuard] epoch={e} step={step} NaN/Inf in {list(bad.keys())}")
        batch = getattr(t, "batch", None)
        if self.save_bad_batch and isinstance(batch, dict) and ("img" in batch):
            try:
                save_dir = Path(getattr(t, "save_dir", Path(".")))
                fname = save_dir/f"nan_batch_e{e:03d}_i{int(step):06d}.jpg"; plot_images(images=batch["img"], batch=batch, fname=fname)
                LOGGER.error(f"[NaNGuard] saved bad batch -> {fname}")
            except Exception as ex: LOGGER.error(f"[NaNGuard] save-batch failed: {ex}")
        try:
            B = getattr(t, "batch", None)
            imgs = (B.get("img") if isinstance(B, dict) else (B[0] if B else None))
            def _mm(x):
                try: return (float(x.min().item()), float(x.max().item()))
                except Exception: return (None, None)
            img_minmax = _mm(imgs) if hasattr(imgs, "dtype") else (None, None)
            lr = None
            try:
                for g in t.optimizer.param_groups: lr = g.get("lr", None); break
            except Exception: pass
            li = getattr(t, "loss_items", None)
            li = [float(v) for v in li] if li is not None else None
            LOGGER.error(f"[NaNGuard/ctx] img_minmax={img_minmax} lr={lr} loss_items={li}")
        except Exception: pass
        if self.stop_on_nan: raise RuntimeError("[NaNGuard] Stop due to NaN")

class BatchSanityFilter:
    def __init__(self, eps: float = 1e-6): self.eps = float(eps)
    def on_train_batch_start(self, t, *args, **kwargs):
        batch = getattr(t, "batch", None)
        if not isinstance(batch, dict): return
        for k in ("img","bboxes","cls"):
            v = batch.get(k, None)
            if torch.is_tensor(v): batch[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.is_tensor(batch.get("bboxes", None)):
            bb = batch["bboxes"]
            if torch.all(bb.abs() <= 2.0): batch["bboxes"] = torch.clamp(bb, 0.0 - self.eps, 1.0 + self.eps)

class LRGuard:
    def on_train_start(self, t):
        args = getattr(t, "args", getattr(t.model, "args", None))
        lr0 = float(getattr(args, "lr0", 1e-3)); lrf = float(getattr(args, "lrf", lr0))
        if lr0 <= 0: LOGGER.warning("[LRGuard] lr0<=0?"); return
        if lr0 > 1e-2: LOGGER.warning(f"[LRGuard] lr0={lr0} có thể hơi cao (>=1e-2)")
    def on_train_epoch_start(self, t): pass

# -----------------------------------------------------------------------------
# F) Debug ảnh STN + preview FG/BG pairing
# -----------------------------------------------------------------------------
class DebugImages(_Ctx):
    """Render cặp (Original vs STN) + overlay θ."""
    def __init__(self, epochs=(0,5,10,15,20), max_images=5):
        self.epochs, self.max_images, self.samples, self.dbg_dir = set(map(int, epochs)), int(max_images), [], None

    @staticmethod
    def _grab_loader(t):
        return getattr(getattr(t, "validator", None), "dataloader", None) or t.train_loader

    def _cache_samples(self, t):
        for batch in self._grab_loader(t):
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

    def on_train_epoch_end(self, t):
        ep = int(getattr(t, "epoch", 0))
        if ep not in self.epochs: return
        if self.dbg_dir is None:
            self.dbg_dir = os.path.join(str(t.save_dir), "stn_dbg"); os.makedirs(self.dbg_dir, exist_ok=True)
        if not self.samples: self._cache_samples(t)

        model = t.model; device = next(model.parameters()).device
        was_train = model.training; model.eval()

        stn_control = None
        for cb in t.callbacks.get("on_train_epoch_start", []):
            if isinstance(cb, STNControl): stn_control = cb; break
        if stn_control:
            if stn_control._mode == "blend":
                stn_control._apply_identity(model, False)
                stn_control._apply_blend(model, stn_control._alpha)
                LOGGER.info(f"[DebugImages] Applied STN blend patch (α={stn_control._alpha:.3f}) for visualization.")
            else:
                stn_control._apply_identity(model, True)
                LOGGER.info(f"[DebugImages] Applied STN identity patch for visualization.")

        H_panel = 672
        try:
            for i,(img0,bxywh,bcls,path0) in enumerate(self.samples):
                x = img0.clone().to(device).unsqueeze(0).float(); x = x/255.0 if x.max()>1 else x
                if hasattr(t, "state"): t.state.pop("stn_theta", None); t.state.pop("stn_out", None)
                _ = model(x); theta = getattr(t, "state", {}).get("stn_theta", None); stn_img = getattr(t, "state", {}).get("stn_out", None)

                ori = img0.detach().cpu().numpy()
                if ori.ndim==3 and ori.shape[0] in (1,3): ori = np.transpose(ori, (1,2,0))
                if ori.max() <= 1.5: ori = (ori*255.0).round()
                ori = np.clip(ori,0,255).astype(np.uint8)
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
                        out = _put_text(out, CLASS_LABELS.get(int(c), str(int(c))), (x1+4, max(0,y1-22)), col, 28)
                    return out

                Ld,Rd = draw(L, xyxy_left), draw(R, xyxy_right)
                Ld = _put_text(Ld, "ORIGINAL", (10,10), (255,255,0), 42)
                Rd = _put_text(Rd, "STN",      (10,10), (255,255,0), 42)

                ttheta = (theta[0] if (isinstance(theta, torch.Tensor) and theta.dim()==3) else theta)
                if not isinstance(ttheta, torch.Tensor) or ttheta.numel()==0:
                    ttheta = torch.tensor([[1,0,0],[0,1,0]], dtype=torch.float32)
                T = ttheta.detach().cpu().view(2,3).numpy()
                Rd = _put_text(Rd, f"θ0: {T[0,0]:+0.5f} {T[0,1]:+0.5f} {T[0,2]:+0.5f}", (10, 60), (120,255,120), 28)
                Rd = _put_text(Rd, f"θ1: {T[1,0]:+0.5f} {T[1,1]:+0.5f} {T[1,2]:+0.5f}", (10, 94), (120,255,120), 28)
                dx, dy = float(T[0,2])*(Ws/2.0), float(T[1,2])*(Hs/2.0)
                Rd = _put_text(Rd, f"Δt ≈ ({dx:+.2f}px, {dy:+.2f}px)", (10, 128), (180,220,255), 26)

                both = np.concatenate([Ld,Rd],1); pad=16
                Hh,Ww = both.shape[:2]; canvas = np.full((Hh+2*pad, Ww+2*pad, 3), (40,40,40), np.uint8)
                canvas[pad:pad+Hh, pad:pad+Ww] = both
                base = os.path.splitext(os.path.basename(str(path0) or f"sample{i:02d}"))[0]
                out = os.path.join(self.dbg_dir, f"{base}_epoch_{ep:03d}_{i:02d}.png"); cv2.imwrite(out, canvas)
        finally:
            model.train(was_train)
        LOGGER.info(f"[DebugImages] saved for epoch {ep}")

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
    def on_train_epoch_end(self, t):
        e = int(getattr(t, "epoch", 0))
        if e not in self.epochs: return
        try:
            batch = next(iter(t.train_loader))
        except Exception as ex:
            LOGGER.warning(f"[DebugBgPairROIs] cannot fetch batch: {ex}"); return
        pair_idx, abn_mask, im_files = batch.get("pair_idx"), batch.get("abn_mask"), batch.get("im_files")
        if pair_idx is None or abn_mask is None or im_files is None:
            LOGGER.warning("[DebugBgPairROIs] missing pair_idx/abn_mask/im_files"); return
        save_dir = Path(str(t.save_dir))/"bgpair_preview"; save_dir.mkdir(parents=True, exist_ok=True)
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
                names = getattr(getattr(t, "model", None), "names", None) or getattr(getattr(t, "validator", None), "names", None)
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

# -----------------------------------------------------------------------------
# G) Val debug nhỏ: vẽ nhanh bbox dự đoán sau NMS (1-2 dòng log/ảnh)
# -----------------------------------------------------------------------------
def debug_val_sample(images: torch.Tensor, pred_scores: torch.Tensor, pred_bboxes: torch.Tensor,
                     save_dir="runs_stn/val_debug", conf_thres=0.30):
    os.makedirs(save_dir, exist_ok=True)
    scores = pred_scores.sigmoid()
    keep = scores.max(2).values > float(conf_thres)
    B = images.shape[0]
    for i in range(B):
        img = images[i].detach().cpu()
        if img.max() <= 1.5: img = img * 255.0
        img_np = img.permute(1, 2, 0).clamp(0, 255).numpy().astype("uint8").copy()
        for j in torch.nonzero(keep[i], as_tuple=False).view(-1).tolist():
            xyxy = pred_bboxes[i, j].detach().cpu().numpy().tolist()
            label = int(scores[i, j].argmax().item())
            col = CLASS_COLORS[label % len(CLASS_COLORS)]
            x1,y1,x2,y2 = map(int, xyxy)
            cv2.rectangle(img_np, (x1,y1), (x2,y2), col, 2)
            cv2.putText(img_np, f"{label}", (x1+4, max(18,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        outp = os.path.join(save_dir, f"val_dbg_{i}.jpg")
        cv2.imwrite(outp, img_np)

def debug_pred_boxes_on_val(validator):
    batch = getattr(validator, "batch", None)
    preds  = getattr(validator, "pred", None)
    if batch is None or preds is None: return
    if not isinstance(preds, (list, tuple)) or len(preds) < 3: return
    boxes, scores = preds[0], preds[1]
    if not (torch.is_tensor(boxes) and torch.is_tensor(scores)): return
    images = batch["img"]
    try:
        debug_val_sample(images, scores, boxes, save_dir=str(Path(getattr(validator, "save_dir", "."))/ "val_dbg"))
    except Exception as e:
        LOGGER.debug(f"[val_dbg] skip: {e}")

class ValPredStats:
    def __init__(self, every=1): self.every = max(1, int(every)); self.i = 0
    def on_val_batch_end(self, v):
        self.i += 1
        if (self.i % self.every) != 0: return
        preds = getattr(v, "pred", None)
        if preds is None or not isinstance(preds, (list,tuple)) or len(preds) < 1: return
        boxes = preds[0]
        if torch.is_tensor(boxes):
            LOGGER.info(f"[ValPredStats] batch{self.i}: boxes={int((boxes[...,0]>=0).sum().item())}")

# -----------------------------------------------------------------------------
# H) VAL overrides & probe + results.csv guard
# -----------------------------------------------------------------------------
class ForceValArgs:
    """Ép một số tham số validator cho debug (conf/iou/max_det/agnostic)."""
    def __init__(self, conf=0.001, iou=0.50, max_det=2000, agnostic=False):
        self.conf, self.iou, self.max_det, self.agnostic = float(conf), float(iou), int(max_det), bool(agnostic)
    def on_val_start(self, v):
        for k in ("conf", "iou", "max_det", "agnostic"):
            if hasattr(v.args, k):
                setattr(v.args, k, getattr(self, k))

def apply_val_debug_overrides(
    yolo,
    conf: float = 0.001,
    iou: float = 0.5,
    *,
    epochs: set|list|tuple|None = None,
    every_n: int|None = None
):
    """
    Chỉ bật lưu ảnh/plots khi epoch thỏa điều kiện:
    - `epochs` chứa epoch (0-based), hoặc
    - `every_n` chia hết (vd every_n=5 => e%5==0).
    Nếu cả hai đều None => hành vi cũ: bật mọi epoch.
    """
    epochs = (set(epochs) if epochs is not None else None)
    every_n = int(every_n) if (every_n is not None and every_n > 0) else None

    def _on_val_start(v):
        e = int(getattr(getattr(v, "trainer", v), "epoch", 0))
        active = True
        if epochs is not None:
            active = (e in epochs)
        if active and every_n is not None:
            active = (e % every_n == 0)

        # luôn hạ ngưỡng conf/iou để soi dự đoán
        v.args.conf = float(conf); v.args.iou = float(iou)

        if active:
            v.args.save_txt = True; v.args.save_conf = True; v.args.save = True; v.args.plots = True
            LOGGER.info(f"[VAL/Debug] epoch {e}: SAVE=ON (conf={conf}, iou={iou})")
        else:
            # Tắt toàn bộ lưu để tránh spam
            v.args.save_txt = False; v.args.save_conf = False; v.args.save = False; v.args.plots = False
            LOGGER.info(f"[VAL/Debug] epoch {e}: SAVE=OFF (conf={conf}, iou={iou})")

    yolo.add_callback("on_val_start", _on_val_start)

def register_val_probe(yolo, gt_labels_dir: str|None = None):
    if gt_labels_dir:
        LOGGER.info(f"[VAL/Probe] GT labels dir: {gt_labels_dir}")
    else:
        LOGGER.info("[VAL/Probe] no GT dir provided (skipping)")

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

# -----------------------------------------------------------------------------
# I) Logging filter + seed + TSNE
# -----------------------------------------------------------------------------
def apply_logging_filter(): pass

def set_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class SafeTSNE:
    def __init__(self, every=0, max_samples=2000, per_class_cap=300, seed=0):
        self.every, self.max_samples, self.per_class_cap, self.seed = int(every), int(max_samples), int(per_class_cap), int(seed)
        self.cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
    def on_train_start(self, t): self.cache.clear()
    def on_train_batch_end(self, t): pass
    def on_train_epoch_end(self, t):
        if self.every <= 0: return
        e = int(getattr(t, "epoch", 0)) + 1
        if (e % self.every) != 0: return
        out = Path(str(t.save_dir)) / f"tsne_epoch{e:03d}.npy"
        try:
            np.save(str(out), np.zeros((1, 2), dtype=np.float32))
            LOGGER.info(f"[TSNE] dumped placeholder -> {out}")
        except Exception as ex:
            LOGGER.warning(f"[TSNE] save failed: {ex}")
    def on_train_end(self, t): self.cache.clear()

# -----------------------------------------------------------------------------
# J) Làm dịu log STN khi visualize/val
# -----------------------------------------------------------------------------
class QuietSTNLogs:
    def on_train_epoch_start(self, tr, ctrl: STNControl): pass
    def on_val_start(self, tr, ctrl: STNControl): pass
def setup_stn_quiet_logs(): return QuietSTNLogs()

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
        # Nếu Ultralytics truyền path (WindowsPath/str) thì thay bằng model đang train
        if isinstance(m, (str, bytes, Path)):
            tr = getattr(v, "trainer", None)
            mdl = getattr(tr, "model", None) if tr is not None else None
            if mdl is not None:
                v.model = mdl
                from ultralytics.utils import LOGGER
                LOGGER.warning("[FinalEvalFix] Validator.model is a Path -> using in-memory trainer.model instead.")
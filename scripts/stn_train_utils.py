# stn_train_utils.py
from __future__ import annotations
import os, glob, csv, logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import LOGGER
from ultralytics.utils.stn_utils import debug_val_sample
from pathlib import Path
from ultralytics.utils import LOGGER
import re
# ========== CSV GUARD (REPLACE TOÀN BỘ BLOCK NÀY) ==========
from pathlib import Path
from ultralytics.utils import LOGGER

class ResultsCsvGuard:
    """
    Giữ results.csv luôn đồng nhất số cột.
    - Khi phát hiện hàng dữ liệu nhiều cột hơn header, tự mở rộng header
      (mặc định thêm 'train/supcon_loss') và đệm 0 cho các dòng cũ.
    """
    def __init__(self, extra_cols=("train/supcon_loss",)):
        self.extra_cols = tuple(extra_cols)

    def _fix(self, trainer):
        csv_path = Path(getattr(trainer, "save_dir", "")) / "results.csv"
        if not csv_path.exists():
            return
        try:
            raw = csv_path.read_text(encoding="utf-8")
            lines = [ln.rstrip("\n") for ln in raw.splitlines()]
            if not lines:
                return

            header = lines[0]
            H = len(header.split(","))
            # Tìm số cột tối đa thực tế
            M = H
            for ln in lines[1:]:
                if ln.strip():
                    M = max(M, len(ln.split(",")))
            if M <= H:
                return  # đã ổn, không cần sửa

            add = M - H
            add_names = list(self.extra_cols)[:add] or [f"extra_{i}" for i in range(add)]
            new_header = header + "".join("," + n for n in add_names)

            new_lines = [new_header]
            for ln in lines[1:]:
                parts = ln.split(",") if ln.strip() else []
                if len(parts) < M:
                    parts += ["0"] * (M - len(parts))  # đệm 0 cho dòng cũ
                elif len(parts) > M:
                    parts = parts[:M]  # cắt bớt nếu thừa (phòng hờ)
                new_lines.append(",".join(parts))

            tmp = csv_path.with_suffix(".csv.tmp")
            tmp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            tmp.replace(csv_path)
            LOGGER.warning(f"[CSVFix] Expanded results.csv by {add} col(s): {add_names}")
        except Exception as e:
            LOGGER.warning(f"[CSVFix] skip ({e})")

    # gọi sau mỗi epoch và sau VAL để cố gắng giữ đồng nhất sớm nhất có thể
    def on_train_epoch_end(self, trainer, *a, **k):  self._fix(trainer)
    def on_val_end(self, trainer, *a, **k):          self._fix(trainer)
    def on_fit_epoch_end(self, trainer, *a, **k):    self._fix(trainer)


class _PatchResultsReader:
    """
    Monkey-patch trainer.read_results_csv để:
    - Nếu pandas đọc lỗi -> gọi guard._fix() rồi đọc lại.
    - Dùng engine='python' và on_bad_lines='skip' làm phương án dự phòng.
    """
    def __init__(self, guard: ResultsCsvGuard):
        self.guard = guard

    def on_train_start(self, trainer, *a, **k):
        from types import MethodType
        import pandas as pd

        def _patched_read_results_csv(_self):
            try:
                return pd.read_csv(_self.csv).to_dict(orient="list")
            except Exception as e:
                try:
                    self.guard._fix(_self)
                except Exception:
                    pass
                # đọc lại với engine='python'; nếu pandas mới, dùng on_bad_lines='skip'
                try:
                    return pd.read_csv(_self.csv, engine="python", on_bad_lines="skip").to_dict(orient="list")
                except TypeError:
                    # pandas cũ không có on_bad_lines
                    return pd.read_csv(_self.csv, engine="python").to_dict(orient="list")

        try:
            trainer.read_results_csv = MethodType(_patched_read_results_csv, trainer)
            LOGGER.warning("[CSVFix] Patched trainer.read_results_csv with guard")
        except Exception as e:
            LOGGER.warning(f"[CSVFix] patch failed: {e}")


def register_results_csv_guard(yolo):
    guard = ResultsCsvGuard()
    yolo.add_callback("on_train_epoch_end", guard.on_train_epoch_end)
    yolo.add_callback("on_val_end",         guard.on_val_end)
    yolo.add_callback("on_fit_epoch_end",   guard.on_fit_epoch_end)   # thêm mốc này cho chắc

    # Monkey-patch ngay khi train bắt đầu
    patcher = _PatchResultsReader(guard)
    yolo.add_callback("on_train_start", patcher.on_train_start)

# ==================== Logging filter: ẩn dòng [SupConDbg] ====================
class _DenySupConDbg(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return "[SupConDbg]" not in record.getMessage()
        except Exception:
            return True

def apply_logging_filter():
    """Ẩn log [SupConDbg] để log sạch hơn (an toàn nếu gọi nhiều lần)."""
    try:
        flt = _DenySupConDbg()

        # Root logger
        root = logging.getLogger()
        if not any(isinstance(f, _DenySupConDbg) for f in getattr(root, "filters", [])):
            root.addFilter(flt)

        # Logger "ultralytics"
        ul = logging.getLogger("ultralytics")
        if not any(isinstance(f, _DenySupConDbg) for f in getattr(ul, "filters", [])):
            ul.addFilter(flt)

        # Ultralytics utils LOGGER (đã import ở đầu file)
        handlers = getattr(LOGGER, "handlers", None)
        if handlers:
            for h in list(handlers):
                if not any(isinstance(f, _DenySupConDbg) for f in getattr(h, "filters", [])):
                    h.addFilter(flt)
    except Exception as e:
        # Đừng crash chỉ vì filter; ghi debug nhẹ nhàng
        logging.getLogger(__name__).debug(f"apply_logging_filter: {e}")


# ==================== Seed ====================
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== LRGuard (fix initial_lr) ====================
class LRGuard:
    """Ensure every optimizer param_group has 'initial_lr' (avoids scheduler KeyError)."""
    def _fix(self, trainer):
        opt = getattr(trainer, "optimizer", None)
        if opt is None:
            return
        fixed = 0
        for g in opt.param_groups:
            if "initial_lr" not in g:
                g["initial_lr"] = g.get("lr", 0.0)
                fixed += 1
        if fixed:
            LOGGER.info(f"[LRGuard] injected 'initial_lr' into {fixed} optimizer param groups")
    def on_train_start(self, trainer, *_, **__): self._fix(trainer)
    def on_train_epoch_start(self, trainer, *_, **__): self._fix(trainer)

# ==================== SafeTSNE (nhẹ, tự vẽ png) ====================
class SafeTSNE:
    """Collect GAP vectors (CPU) via a forward hook and export t-SNE every N epochs."""
    def __init__(self, every=10, max_samples=4000, per_class_cap=300, seed=0):
        self.every = int(every)
        self.max_samples = int(max_samples)
        self.per_class_cap = int(per_class_cap)
        self.seed = int(seed)
        self.hook = None
        self._feat = None
        self._xs, self._ys = [], []

    def _cfg(self, trainer, key, default=None):
        cfg = getattr(trainer, "_supcon_cfg", None)
        if cfg is None:
            return default
        return (cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default))

    @torch.no_grad()
    def _hook_fn(self, module, _in, out):
        if torch.is_tensor(out):
            x = F.adaptive_avg_pool2d(out, (1,1)).flatten(1) if out.dim()==4 else out
            if x.dim()==2:
                self._feat = x.detach().to("cpu", non_blocking=True).contiguous()

    def _resolve_module(self, model, idx, name):
        modlist = getattr(model, "model", None)
        if modlist is None:
            return None, None
        if isinstance(idx, int) and 0 <= idx < len(modlist):
            return modlist[idx], f"model.model[{idx}]"
        if name:
            low = str(name).lower()
            for i, m in enumerate(modlist):
                if low in m.__class__.__name__.lower():
                    return m, f"model.model[{i}]<{m.__class__.__name__}>"
        for i, m in enumerate(modlist):
            if hasattr(m, "forward"):
                return m, f"model.model[{i}]<{m.__class__.__name__}>"
        return None, None

    def on_train_start(self, trainer, *a, **k):
        idx = self._cfg(trainer, "supcon_out", None); name = self._cfg(trainer, "supcon_name", None)
        m, where = self._resolve_module(trainer.model, idx, name)
        if m is None:
            LOGGER.info("[SafeTSNE] could not attach hook -> skip")
            return
        if self.hook is not None:
            try: self.hook.remove()
            except Exception: pass
        self.hook = m.register_forward_hook(self._hook_fn)
        self._feat = None; self._xs.clear(); self._ys.clear()
        LOGGER.info(f"[SafeTSNE] hook at {where}; every={self.every}, max={self.max_samples}")

    @torch.no_grad()
    def on_train_batch_end(self, trainer, *a, **k):
        if self._feat is None:
            return
        x = self._feat; self._feat = None
        y = None
        b = getattr(trainer, "batch", None)
        if isinstance(b, dict) and torch.is_tensor(b.get("cls", None)):
            y = b["cls"].view(-1).to("cpu", non_blocking=True).long()
            if y.numel() != x.size(0):
                y = None
        if y is None:
            y = torch.full((x.size(0),), -1, dtype=torch.long)
        if self.per_class_cap > 0:
            from collections import Counter
            cnt = Counter(self._ys)
            for xi, yi in zip(x, y):
                yi = int(yi.item())
                if cnt[yi] < self.per_class_cap:
                    self._xs.append(xi.numpy()); self._ys.append(yi); cnt[yi] += 1
                if len(self._xs) >= self.max_samples:
                    break
        else:
            for xi, yi in zip(x, y):
                self._xs.append(xi.numpy()); self._ys.append(int(yi.item()))
                if len(self._xs) >= self.max_samples:
                    break
        if len(self._xs) > self.max_samples:
            self._xs = self._xs[:self.max_samples]; self._ys = self._ys[:self.max_samples]

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, *a, **k):
        ep = int(getattr(trainer, "epoch", 0))
        if self.every <= 0 or (ep % self.every) != 0:
            return
        if len(self._xs) < 50:
            LOGGER.info(f"[SafeTSNE] skip epoch {ep}: few samples ({len(self._xs)})")
            return
        xs = np.asarray(self._xs, dtype=np.float32)
        ys = np.asarray(self._ys, dtype=np.int64) if len(self._ys)==len(self._xs) else None
        try:
            from sklearn.manifold import TSNE
            emb = TSNE(
                n_components=2,
                perplexity=min(30, max(5, xs.shape[0]//100)),
                init="pca",
                learning_rate="auto",
                random_state=self.seed,
                n_iter=1000,
                verbose=0
            ).fit_transform(xs)
        except Exception as e:
            LOGGER.warning(f"[SafeTSNE] TSNE failed: {e}")
            return
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6,5), dpi=160)
            if ys is None or (ys < 0).all():
                plt.scatter(emb[:,0], emb[:,1], s=3)
            else:
                num_cls = int(np.clip(len(np.unique(ys[ys>=0])), 1, 20))
                plt.scatter(emb[:,0], emb[:,1], c=np.clip(ys, 0, num_cls-1), s=3, cmap="tab20")
            plt.title(f"t-SNE epoch {ep} (n={xs.shape[0]})"); plt.tight_layout()
            out_dir = os.path.join(str(getattr(trainer, "save_dir", "runs")), "embeddings"); os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"tsne_epoch_{ep:03d}.png"); fig.savefig(out_path); plt.close(fig)
            LOGGER.info(f"[SafeTSNE] saved {out_path} (n={xs.shape[0]})")
        except Exception as e:
            LOGGER.warning(f"[SafeTSNE] plot failed: {e}")

    def on_train_end(self, trainer, *a, **k):
        if self.hook is not None:
            try: self.hook.remove()
            except Exception: pass
        self.hook = None; self._feat = None; self._xs.clear(); self._ys.clear()

# ==================== Debug overlay dự đoán ở VAL ====================
def debug_pred_boxes_on_val(trainer, *a, **k):
    """Lưu nhanh overlay box dự đoán ở VAL (sample) để soi mắt thường."""
    if not hasattr(trainer, "batch") or not hasattr(trainer, "pred"):
        return
    batch = trainer.batch
    pred = trainer.pred
    if not isinstance(pred, (list, tuple)) or len(pred) < 2:
        return
    pred_scores, pred_bboxes = pred[0], pred[1]
    images = batch.get("img", None) if isinstance(batch, dict) else None
    if images is None:
        return
    save_dir = f"{trainer.save_dir}/val_pred"
    try:
        debug_val_sample(images, pred_scores, pred_bboxes, save_dir=save_dir)
    except Exception as e:
        LOGGER.warning(f"[debug_pred_boxes_on_val] failed: {e}")

# ==================== Chuẩn hóa pred cho Validator YOLOv8 ====================
def fix_val_prediction_format(trainer, *a, **k):
    """
    Biến trainer.pred (scores[B,N,C], boxes[B,N,4]) -> list[B] of [N,6] (xyxy, conf, cls)
    để Validator YOLOv8 hiểu đúng format.
    """
    pred = getattr(trainer, "pred", None)
    if not isinstance(pred, (list, tuple)) or len(pred) < 2:
        return
    scores, boxes = pred[0], pred[1]  # [B, N, C], [B, N, 4]
    if not (torch.is_tensor(scores) and torch.is_tensor(boxes)):
        return
    preds = []
    probs = scores.sigmoid()
    for i in range(scores.shape[0]):
        conf, cls = probs[i].max(dim=1)  # [N]
        mask = conf > 0.001
        if mask.sum() == 0:
            preds.append(torch.zeros((0, 6), device=scores.device))
            continue
        filtered_boxes = boxes[i][mask]
        filtered_conf = conf[mask].unsqueeze(1)
        filtered_cls = cls[mask].float().unsqueeze(1)
        preds.append(torch.cat([filtered_boxes, filtered_conf, filtered_cls], dim=1))  # [N, 6]
    trainer.pred = preds

# ==================== VAL overrides: ép lưu dự đoán ====================
def apply_val_debug_overrides(yolo, conf: float = 0.001, iou: float = 0.50):
    """Ép validator lưu txt + conf + overlay + plots mỗi epoch."""
    val_overrides = dict(
        save_txt=True,
        save_conf=True,
        save=True,
        plots=True,
        conf=float(conf),
        iou=float(iou),
    )
    if hasattr(yolo, "overrides") and isinstance(yolo.overrides, dict):
        yolo.overrides.update(val_overrides)
    else:
        yolo.overrides = val_overrides
    LOGGER.info(f"[VAL/Debug] overrides applied: {val_overrides}")

# ==================== Bypass SDTN inverse chỉ khi VAL ====================
class _BypassSDTN:
    """
    Bypass inverse SDTN chỉ khi VAL.
    - Chịu cả 2 ngữ cảnh: Trainer.validate() và Validator.__call__().
    - An toàn khi model chưa sẵn: retry ở on_val_batch_start.
    """
    def __init__(self, attr: str = "bypass_sdt_invert", verbose: bool = True):
        self.attr = attr
        self.verbose = verbose
        self._targets = []  # list các module head tìm được

    def _get_model(self, obj):
        # 1) trực tiếp
        m = getattr(obj, "model", None)
        if m is not None:
            return m
        # 2) qua trainer
        trainer = getattr(obj, "trainer", None)
        if trainer is not None:
            m = getattr(trainer, "model", None)
            if m is not None:
                return m
        # 3) một số bản có session
        session = getattr(obj, "session", None)
        if session is not None:
            m = getattr(session, "model", None)
            if m is not None:
                return m
        return None

    def _collect_heads_from(self, model):
        self._targets.clear()
        if model is None:
            return
        # Nếu model dạng YOLO(...).model = Sequential/list
        modlist = getattr(model, "model", None)
        if modlist is not None:
            for m in modlist:
                name = m.__class__.__name__.lower()
                if ("detectsdt" in name) or ("detectsdtn" in name) or hasattr(m, self.attr) or hasattr(m, "inverse_affine"):
                    self._targets.append(m)
            return
        # Fallback: duyệt modules()
        try:
            for m in model.modules():
                name = m.__class__.__name__.lower()
                if ("detectsdt" in name) or ("detectsdtn" in name) or hasattr(m, self.attr) or hasattr(m, "inverse_affine"):
                    self._targets.append(m)
        except Exception:
            pass

    def _set_flag(self, value: int):
        ok = 0
        for m in self._targets:
            try:
                setattr(m, self.attr, int(value))
                ok += 1
            except Exception:
                pass
        if self.verbose:
            LOGGER.info(f"[VAL/BypassSDTN] set {self.attr}={value} for {ok} head(s)")

    def on_val_start(self, obj, *a, **k):
        model = self._get_model(obj)
        if model is None:
            LOGGER.warning("[VAL/BypassSDTN] model is None at on_val_start; will retry on first val batch.")
            return
        self._collect_heads_from(model)
        if not self._targets:
            LOGGER.warning("[VAL/BypassSDTN] no SDTN-like head found at on_val_start; will retry on first val batch.")
            return
        self._set_flag(1)

    def on_val_batch_start(self, obj, *a, **k):
        # Nếu lúc on_val_start chưa tìm được model/head, thử lại ở batch đầu
        if self._targets:
            return
        model = self._get_model(obj)
        if model is None:
            return
        self._collect_heads_from(model)
        if self._targets:
            self._set_flag(1)

    def on_val_end(self, obj, *a, **k):
        if not self._targets:
            return
        self._set_flag(0)
        self._targets.clear()

def register_bypass_sdt_callbacks(yolo):
    bypass = _BypassSDTN()
    # Đăng ký ở cả 2 mốc: đầu VAL và đầu mỗi batch VAL (để retry khi model chưa sẵn)
    yolo.add_callback("on_val_start",       bypass.on_val_start)
    yolo.add_callback("on_val_batch_start", bypass.on_val_batch_start)
    yolo.add_callback("on_val_end",         bypass.on_val_end)
# ==== REPLACE đến đây ====


def register_bypass_sdt_callbacks(yolo):
    bypass = _BypassSDTN()
    yolo.add_callback("on_val_start", bypass.on_val_start)
    yolo.add_callback("on_val_end",   bypass.on_val_end)

# ==================== Probe IoU GT↔Pred sau VAL ====================
def _xywhn_to_xyxy(xywhn):
    x, y, w, h = xywhn
    return [x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0]

def _iou(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    sa = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    sb = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    return inter / (sa + sb - inter + 1e-12)

def _read_label_file(path: Path, is_pred: bool):
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            toks = ln.strip().split()
            if len(toks) < 5:
                continue
            cls = int(float(toks[0]))
            x, y, w, h = map(float, toks[1:5])
            # pred.txt có thể có thêm conf → bỏ qua
            items.append((cls, _xywhn_to_xyxy([x,y,w,h])))
    return items

def _infer_gt_labels_dir(trainer) -> Optional[Path]:
    # Thử lấy từ trainer.data['val'] (đường ảnh) rồi chuyển sang labels/
    d = getattr(trainer, "data", None)
    valp = None
    if isinstance(d, dict):
        valp = d.get("val", None) or d.get("validation", None)
    if isinstance(valp, (str, Path)):
        p = Path(valp)
        if p.is_dir():
            # thay 'images'->'labels' nếu có
            parts = list(p.parts)
            for i, s in enumerate(parts):
                if s.lower() == "images":
                    parts[i] = "labels"
                    break
            guess = Path(*parts)
            if guess.exists():
                return guess
            # nếu thư mục cha có 'labels'
            cand = p.parent / "labels"
            if cand.exists():
                return cand
    # Fallback: scan thư mục 'valid/labels' dưới gốc data nếu có
    save_dir = Path(getattr(trainer, "save_dir", "runs"))
    root = save_dir.parent if save_dir else Path(".")
    for cand in [
        root / "datasets" / "valid" / "labels",
        root / "dataset"  / "valid" / "labels",
    ]:
        if cand.exists():
            return cand
    return None

class ProbeValTxt:
    """Sau VAL: so sánh IoU GT↔Pred ở không gian normalized, ghi CSV."""
    def __init__(self, gt_labels_dir: Optional[Path] = None):
        self.gt_labels_dir = Path(gt_labels_dir) if gt_labels_dir else None

    def on_val_end(self, trainer, *a, **k):
        pred_labels = Path(trainer.save_dir) / "val" / "labels"
        if not pred_labels.exists():
            LOGGER.warning("[Probe] pred labels folder not found; skip")
            return
        gt_dir = self.gt_labels_dir or _infer_gt_labels_dir(trainer)
        if gt_dir is None or not gt_dir.exists():
            LOGGER.warning("[Probe] GT labels folder not found; skip")
            return

        stems = {p.stem for p in pred_labels.glob("*.txt")}
        if not stems:
            LOGGER.warning("[Probe] no pred txt found; skip")
            return

        rows = [["stem","n_gt","n_pred","mean_best_iou","gt_>0.1","gt_>0.3","gt_>0.5"]]
        for s in sorted(stems):
            gts  = _read_label_file(gt_dir / f"{s}.txt", is_pred=False)
            prs  = _read_label_file(pred_labels / f"{s}.txt", is_pred=True)
            if len(gts) == 0:
                rows.append([s,0,len(prs),0.0,0.0,0.0,0.0]); continue
            bests = []
            for _, g in gts:
                best = 0.0
                for _, p in prs:
                    best = max(best, _iou(g,p))
                bests.append(best)
            mean_iou = float(sum(bests)/len(bests)) if bests else 0.0
            gt10 = sum(i>0.1 for i in bests)/max(1,len(bests))
            gt30 = sum(i>0.3 for i in bests)/max(1,len(bests))
            gt50 = sum(i>0.5 for i in bests)/max(1,len(bests))
            rows.append([s, len(gts), len(prs), round(mean_iou,4), round(gt10,3), round(gt30,3), round(gt50,3)])

        out_csv = pred_labels.parent / "val_probe_epoch.csv"
        try:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
            LOGGER.info(f"[Probe] wrote {out_csv}")
        except Exception as e:
            LOGGER.warning(f"[Probe] write CSV failed: {e}")

def register_val_probe(yolo, gt_labels_dir: Optional[str] = None):
    probe = ProbeValTxt(gt_labels_dir=Path(gt_labels_dir) if gt_labels_dir else None)
    yolo.add_callback("on_val_end", probe.on_val_end)

class ForceReturnTheta:
    """Bật các cờ để STN trả về theta nếu module hỗ trợ."""
    def _set(self, trainer):
        model = getattr(trainer, "model", None)
        if model is None:
            return
        n = 0
        for m in model.modules():
            name = m.__class__.__name__.lower()
            if "spatialtransformer" in name:
                for flag in ("return_theta", "theta_out", "with_theta"):
                    if hasattr(m, flag) and not bool(getattr(m, flag)):
                        try:
                            setattr(m, flag, True)
                            n += 1
                        except Exception:
                            pass
        if n:
            LOGGER.info(f"[STN/ThetaFlag] enabled return_theta on {n} STN module(s)")
    def on_train_start(self, trainer, *a, **k): self._set(trainer)
    def on_val_start(self, trainer, *a, **k):   self._set(trainer)

class CaptureThetaFromSTN:
    """Hook vào SpatialTransformer để hốt theta và publish vào trainer.state['theta_raw']"""
    def __init__(self):
        self._handles = []
        self._trainer = None
        self._did_warn = False

    def _hook(self, mod, _in, out):
        import torch
        theta = None
        # 1) Ưu tiên lấy từ output tuple: (y, theta) hoặc (y, ..., theta)
        if isinstance(out, (tuple, list)):
            for t in reversed(out):
                if torch.is_tensor(t) and t.dim() >= 2:
                    if (t.shape[-2:] == (2, 3)) or (t.shape[-1] in (6, 9)):
                        theta = t
                        break
        # 2) Không có -> thử lấy từ thuộc tính module: last_theta/theta/theta_raw
        if theta is None:
            for attr in ("last_theta", "theta", "theta_raw"):
                t = getattr(mod, attr, None)
                if torch.is_tensor(t):
                    theta = t
                    break
        # 3) Publish sang trainer.state
        if theta is not None and self._trainer is not None:
            try:
                st = getattr(self._trainer, "state", None) or {}
                st["theta_raw"] = theta.detach()
                self._trainer.state = st
            except Exception:
                pass
        elif not self._did_warn:
            LOGGER.info("[STN/ThetaHook] no theta found in STN outputs/attrs; will keep trying…")
            self._did_warn = True
        return out

    def _attach(self, trainer):
        # gỡ cũ
        for h in self._handles:
            try: h.remove()
            except Exception: pass
        self._handles.clear()
        self._trainer = trainer

        model = getattr(trainer, "model", None)
        if model is None:
            return
        cnt = 0
        for m in model.modules():
            name = m.__class__.__name__.lower()
            if "spatialtransformer" in name:
                try:
                    self._handles.append(m.register_forward_hook(self._hook))
                    cnt += 1
                except Exception:
                    pass
        LOGGER.info(f"[STN/ThetaHook] attached to {cnt} SpatialTransformer module(s)")

    def _detach(self, *_):
        for h in self._handles:
            try: h.remove()
            except Exception: pass
        self._handles.clear()

    # Đăng ký với lifecycle
    def on_train_start(self, trainer, *a, **k): self._attach(trainer)
    def on_val_start(self, trainer, *a, **k):   self._attach(trainer)
    def on_train_end(self, trainer, *a, **k):   self._detach()
    def on_val_end(self, trainer, *a, **k):     self._detach()

class QuietSTNLogs:
    """
    Giảm spam log của STN/CTN/WireTheta:
      - Chỉ cho phép mỗi tag hiển thị 1 lần/epoch (train và val).
      - Tag nhận diện: [STNDbg], '[STN] STN forward: no theta output...' và [WireTheta].
    """
    def __init__(self, extra_patterns=None):
        base_patterns = [
            r'^\[STNDbg\]\s*theta_raw not available',
            r'^\[STN\]\s*STN forward: no theta output, using identity transform\.',
            r'^\[WireTheta\]\s*',  # mọi dòng bắt đầu bằng [WireTheta]
        ]
        if extra_patterns:
            base_patterns.extend(extra_patterns)

        self._regex = [re.compile(p) for p in base_patterns]
        self.cur_epoch = -999
        self.supcon_on = None
        self._shown = set()    # {(epoch, pattern_str)}
        self._attached = False

        # logging.Filter dùng được cho cả LOGGER và root logger
        parent = self
        class _Gate(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                msg = str(record.getMessage())
                # Không đụng vào log khác
                matched = None
                for rx in parent._regex:
                    if rx.search(msg):
                        matched = rx.pattern
                        break
                if matched is None:
                    return True

                key = (parent.cur_epoch, matched)
                # nếu đã hiển thị trong epoch hiện tại -> chặn
                if key in parent._shown:
                    return False

                # lần đầu trong epoch -> cho qua và ghi dấu
                parent._shown.add(key)
                return True

        self._filter = _Gate()

    def attach(self):
        if self._attached:
            return self
        # Gắn filter vào cả root logger lẫn LOGGER riêng của Ultralytics
        logging.getLogger().addFilter(self._filter)
        try:
            LOGGER.addFilter(self._filter)
        except Exception:
            pass
        self._attached = True
        return self

    # ——— Callbacks để reset theo epoch/phase ———
    def on_train_epoch_start(self, trainer, *a, **k):
        # reset bộ đếm mỗi epoch
        self.cur_epoch = int(getattr(trainer, "epoch", 0))
        self._shown = set()
        # cố gắng đọc supcon_on/use_supcon (nếu cần dùng sau này)
        supcon = None
        crit = getattr(trainer, "criterion", None)
        for name in ("use_supcon", "supcon_on", "_use_supcon"):
            if crit is not None and hasattr(crit, name):
                v = getattr(crit, name)
                if isinstance(v, (bool, int)):
                    supcon = bool(v)
                    break
        if supcon is None:
            args = getattr(getattr(trainer, "model", None), "args", None)
            if args is not None and hasattr(args, "supcon_on"):
                supcon = bool(getattr(args, "supcon_on"))
        self.supcon_on = supcon

    def on_val_start(self, *a, **k):
        # val cũng reset, coi như epoch tách biệt
        self._shown = set()
        self.supcon_on = True  # để có 1 dòng đầu tiên nếu phát sinh

def setup_stn_quiet_logs(**kwargs) -> QuietSTNLogs:
    """Tạo và tự động attach bộ lọc log chống spam."""
    q = QuietSTNLogs()
    q.attach()
    return q

import torch

class SeedIdentityThetaOnValStart:
    """Đặt theta_raw = I ngay khi validator bắt đầu, đảm bảo batch-đầu có θ hợp lệ."""
    def __init__(self, B_hint: int = 16):
        self.B_hint = B_hint  # dùng khi chưa biết batch size

    @staticmethod
    def _theta_I(B, device, dtype):
        return torch.tensor([[[1, 0, 0], [0, 1, 0]]], device=device, dtype=dtype).expand(B, -1, -1).contiguous()

    def on_val_start(self, validator, *a, **k):
        # đoán kích thước batch từ dataloader nếu có
        try:
            B = getattr(getattr(validator, "dataloader", None), "batch_size", None)
            if B is None:
                B = self.B_hint
        except Exception:
            B = self.B_hint

        dev = getattr(validator, "device", None)
        if dev is None:
            # cố gắng lấy từ model
            model = getattr(validator, "model", None)
            dev = next(model.parameters()).device if model else torch.device("cpu")

        dtype = torch.float32
        theta_I = self._theta_I(B, dev, dtype)

        # publish vào state cho head SDTN
        try:
            st = getattr(validator, "state", None)
            if st is None and hasattr(validator, "trainer"):
                st = getattr(validator.trainer, "state", None)
            if isinstance(st, dict):
                st["theta_raw"] = theta_I
        except Exception:
            pass
# ──────────────────────────────────────────────────────────────────────────────
# ThetaStats: tóm tắt θ mỗi N batch (dịch, scale, det) để phát hiện STN "giật"
# ──────────────────────────────────────────────────────────────────────────────
class ThetaStats:
    def __init__(self, every=200, tag="train"):
        self.every = int(every)
        self.tag = str(tag)
        self._step = 0

    @staticmethod
    def _summary(theta):
        try:
            import torch
            if theta is None or not hasattr(theta, "shape") or theta.ndim != 3:
                return None
            # theta: (B, 2, 3)
            A = theta[..., :2, :2]
            t = theta[..., :2, 2]
            det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
            smean = (A[..., 0, 0].abs() + A[..., 1, 1].abs()) * 0.5
            return dict(
                t_abs=t.abs().mean().item(),
                t_max=t.abs().max().item(),
                det_mean=det.mean().item(),
                s_mean=smean.mean().item(),
            )
        except Exception:
            return None

    def _log(self, owner):
        from ultralytics.utils import LOGGER
        S = getattr(owner, "state", None) or getattr(getattr(owner, "trainer", None), "state", None)
        if isinstance(S, dict):
            theta = S.get("stn_theta", None)
            summ = self._summary(theta)
            if summ:
                LOGGER.info(f"[ThetaStats/{self.tag}] t_abs={summ['t_abs']:.4f} "
                            f"t_max={summ['t_max']:.4f} det={summ['det_mean']:.4f} "
                            f"s={summ['s_mean']:.4f}")

    # hooks:
    def on_train_batch_end(self, trainer, *a, **k):
        if self.tag != "train": return
        self._step += 1
        if self._step % self.every == 0:
            self._log(trainer)

    def on_val_batch_end(self, validator, *a, **k):
        if self.tag != "val": return
        self._step += 1
        # val thường ít batch hơn: log mỗi batch
        self._log(validator)
# ──────────────────────────────────────────────────────────────────────────────
# ValPredStats: đếm số ảnh không có pred và min/median/max bbox per image
# ──────────────────────────────────────────────────────────────────────────────
class ValPredStats:
    def __init__(self, every=1):
        self.every = int(every)
        self._step = 0

    def on_val_batch_end(self, validator, *a, **k):
        try:
            self._step += 1
            if self._step % self.every != 0:
                return
            from ultralytics.utils import LOGGER
            preds = getattr(validator, "pred", None) or getattr(validator, "outputs", None)
            # Ultralytics thường là list[tensor[N,6]] sau NMS
            if isinstance(preds, list):
                sizes = [ (0 if p is None else (p.shape[0] if hasattr(p, "shape") else len(p))) for p in preds ]
                if sizes:
                    empty = sum(1 for s in sizes if s == 0)
                    sizes_sorted = sorted(sizes)
                    med = sizes_sorted[len(sizes_sorted)//2]
                    LOGGER.info(f"[Val/PredStats] batch={getattr(validator,'batch_i',-1)} "
                                f"empty={empty}/{len(sizes)} min={min(sizes)} "
                                f"median={med} max={max(sizes)}")
        except Exception:
            pass

class ForceValArgs:
    def __init__(self, conf=0.001, iou=0.50, max_det=2000, agnostic=False):
        self.conf, self.iou, self.max_det, self.agnostic = conf, iou, max_det, agnostic

    def on_val_start(self, validator, *a, **k):
        args = getattr(validator, "args", None)
        if args is None:
            return
        args.conf = self.conf
        args.iou = self.iou
        args.max_det = self.max_det
        # một số bản Ultralytics tách agnostic_nms/nms:
        setattr(args, "agnostic_nms", bool(self.agnostic))
        setattr(args, "nms", True)  # bắt buộc NMS trong val

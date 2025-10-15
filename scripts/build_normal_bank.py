# build_normal_bank.py
# Offline: extract global embeddings AFTER STN for normal/abnormal and make quick visuals
import os, json, math, random, argparse, shutil, stat, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO  # dùng bản YOLO + STN của bạn
import yaml
import time

class _Prog:
    def __init__(self, total, tag):
        self.t0 = time.time()
        self.total = total
        self.tag = tag
    def tick(self, i):
        dt = time.time() - self.t0
        done = i
        rate = done / max(dt, 1e-9)
        remain = self.total - done
        eta = remain / max(rate, 1e-9)
        return rate, eta

# ===================== CONFIG (chỉnh ở đây) =====================
# DÙNG raw string r"..." ĐỂ TRÁNH LỖI escape trên Windows
MODEL_WEIGHTS = r"C:\OneDrive\Study\AI\STN_Final_Term\models\yolo11m_stn.pt"
DATA_YAML     = r"C:\OneDrive\Study\AI\STN_Final_Term\dataset.yaml"

IMG_SIZE      = 640         # đúng với training
MIN_CHANNELS  = 1054           # chọn fmap sâu (>= như TSNE bạn dùng)
MIN_DOWNS     = 8             # yêu cầu downsample >= 4x
MAX_IMG       = 10**9          # giới hạn tối đa số ảnh duyệt
SEED          = 42
N_VIS_ABN     = 20             # số abnormal để vẽ panel kiểm chứng
TOPK_VIS      = 3              # số normal hiển thị kèm
OUT_DIR       = "bank_out"     # GIỮ TÊN CỐ ĐỊNH
VIS_DIR       = "bank_vis"     # GIỮ TÊN CỐ ĐỊNH
PRINT_EVERY   = 50             # log mỗi 50 ảnh
USE_REGIONAL_POOL = True
REG_GRID = (1, 2, 4)  # 1x1 + 2x2 + 4x4 -> concat

# ===============================================================

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def safe_rmtree(path):
    """Xóa cưỡng bức thư mục trên Windows (gỡ readonly/lock nếu có)."""
    if not os.path.isdir(path):
        return
    def _on_rm_error(func, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass
    shutil.rmtree(path, onerror=_on_rm_error)

def load_data_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    for k in ("train", "val", "test"):
        if k in y and isinstance(y[k], list):
            y[k] = [str(p) for p in y[k]]
    return y

def list_images_and_labels(data_yaml):
    """
    Đọc phần 'train' trong yaml (string dir hoặc list dir/filelist).
    Tạo list [(img_path, lbl_path)] theo YOLO format.
    """
    def _collect_from_dir(imgdir):
        exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".pnm",".pgm")
        imgs = []
        for root, _, files in os.walk(imgdir):
            for fn in files:
                if fn.lower().endswith(exts):
                    imgs.append(os.path.join(root, fn))
        return imgs

    def _label_path(img_path):
        # images/.../abc.jpg -> labels/.../abc.txt
        p = Path(img_path)
        for tok in ("images", "Images"):
            if tok in p.parts:
                idx = p.parts.index(tok)
                rel = Path(*p.parts[idx+1:]).with_suffix(".txt")
                lbl = Path(*p.parts[:idx], "labels", rel)
                return str(lbl)
        # fallback
        return str(p.parent.parent / "labels" / (p.stem + ".txt"))

    y = load_data_yaml(data_yaml)
    trains = y["train"] if isinstance(y["train"], list) else [y["train"]]
    img_paths = []
    for item in trains:
        item = str(item)
        if os.path.isdir(item):
            img_paths.extend(_collect_from_dir(item))
        elif os.path.isfile(item):
            with open(item, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip()
                    if p:
                        img_paths.append(p)
        else:
            print(f"[WARN] train entry not found: {item}")
    img_paths = sorted(img_paths)
    pairs = [(p, _label_path(p)) for p in img_paths]
    if MAX_IMG < len(pairs):
        pairs = pairs[:MAX_IMG]
    return pairs, y

def is_abnormal(lbl_path):
    try:
        if not os.path.exists(lbl_path):
            return False
        with open(lbl_path, "r", encoding="utf-8") as f:
            s = f.read().strip()
        return len(s) > 0
    except:
        return False

def load_image_resized(path, size):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    h, w = im.shape[:2]
    scale = min(size/float(h), size/float(w))
    nh, nw = int(round(h*scale)), int(round(w*scale))
    imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size), dtype=np.uint8)
    top = (size - nh)//2; left = (size - nw)//2
    canvas[top:top+nh, left:left+nw] = imr
    # ---> replicate to 3 channels (CHW)
    chw = np.stack([canvas, canvas, canvas], axis=0).astype(np.float32) / 255.0  # [3,H,W]
    return chw, dict(orig_hw=(h,w), resized_hw=(nh,nw), pad=(top,left), scale=scale)


def choose_after_stn_module(core: nn.Module, sample_tensor: torch.Tensor):
    """Chọn module đầu tiên SAU STN cho ra out 4D, C>=MIN_CHANNELS, downsample >= MIN_DOWNS."""
    from ultralytics.nn.modules.block import SpatialTransformer as STN
    stn_seen = False
    hooks, outs = [], []
    modules = list(core.modules())

    def mk_hook(m):
        def _h(_m, _i, o):
            if torch.is_tensor(o):
                outs.append((m, o))
            elif isinstance(o, (list, tuple)):
                for t in o:
                    if torch.is_tensor(t):
                        outs.append((m, t))
        return _h

    for m in modules:
        if isinstance(m, STN):
            stn_seen = True
            continue
        if not stn_seen:
            continue
        if hasattr(m, "forward"):
            hooks.append(m.register_forward_hook(mk_hook(m)))

    was = core.training
    core.eval()
    with torch.no_grad():
        _ = core(sample_tensor)
    for h in hooks:
        h.remove()
    core.train(was)

    H, W = sample_tensor.shape[-2], sample_tensor.shape[-1]
    cands = []
    for m, o in outs:
        if not (torch.is_tensor(o) and o.dim() == 4):
            continue
        _, C, Hf, Wf = o.shape
        deep = (C >= MIN_CHANNELS) and (Hf <= H//MIN_DOWNS) and (Wf <= W//MIN_DOWNS)
        if deep:
            cands.append((m, (C, Hf, Wf)))
    if not cands:
        for m, o in reversed(outs):
            if torch.is_tensor(o) and o.dim() == 4:
                C, Hf, Wf = o.shape[1:]
                cands.append((m, (C, Hf, Wf)))
                break
    if not cands:
        raise RuntimeError("Không tìm được feature 4D sau STN. Kiểm tra model/forward.")
    print(f"[Probe] chọn feature sau STN: {cands[0][0].__class__.__name__} with shape {cands[0][1]}")
    return cands[0][0]

def get_stn_out_and_feat(core: nn.Module, feat_mod: nn.Module, x: torch.Tensor):
    """Trả về (stn_out_img, feat_map, theta(2x3))."""
    from ultralytics.nn.modules.block import SpatialTransformer as STN
    stn = None
    for m in core.modules():
        if isinstance(m, STN):
            stn = m; break
    if stn is None:
        raise RuntimeError("STN not found in model.")
    stn_out = {"img": None, "theta": None}
    feat_out = {"fmap": None}

    def stn_hook(mod, inp, out):
        stn_out["img"] = out.detach()
        th = None
        for name in ("theta", "_theta", "last_theta"):
            val = getattr(mod, name, None)
            if isinstance(val, torch.Tensor):
                th = val.detach(); break
        if th is None:
            th = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=out.device)
        stn_out["theta"] = th

    def feat_hook(mod, inp, out):
        if torch.is_tensor(out):
            feat_out["fmap"] = out.detach()
        elif isinstance(out, (list, tuple)):
            for t in out:
                if torch.is_tensor(t):
                    feat_out["fmap"] = t.detach(); break

    h1 = stn.register_forward_hook(stn_hook)
    h2 = feat_mod.register_forward_hook(feat_hook)

    was = core.training
    core.eval()
    with torch.no_grad():
        _ = core(x)
    h1.remove(); h2.remove()
    core.train(was)

    if stn_out["img"] is None or feat_out["fmap"] is None:
        raise RuntimeError("Hook không lấy được STN out hoặc feature.")
    return stn_out["img"], feat_out["fmap"], stn_out["theta"]

def global_avg_pool(fmap: torch.Tensor) -> torch.Tensor:
    return fmap.mean(dim=(2,3))  # [B,C]

def regional_pool(fmap: torch.Tensor, grid=(1,2,4)) -> torch.Tensor:
    # fmap: [B,C,H,W] -> concat mean của các ô lưới
    B, C, H, W = fmap.shape
    outs = []
    for g in grid:
        ph = H // g; pw = W // g
        for i in range(g):
            for j in range(g):
                ys, ye = i*ph, (i+1)*ph
                xs, xe = j*pw, (j+1)*pw
                cell = fmap[:, :, ys:ye, xs:xe]
                outs.append(cell.mean(dim=(2,3)))
    return torch.cat(outs, dim=1) if outs else fmap.mean(dim=(2,3))

def make_descriptor(fmap: torch.Tensor) -> torch.Tensor:
    return regional_pool(fmap) if USE_REGIONAL_POOL else fmap.mean(dim=(2,3))

def cosine_topk(query, keys, k):
    q = query / (np.linalg.norm(query, axis=1, keepdims=True)+1e-9)
    K = keys / (np.linalg.norm(keys, axis=1, keepdims=True)+1e-9)
    sims = q @ K.T
    idx = np.argsort(-sims, axis=1)[:, :k]
    return sims, idx

def save_panel(stn_abn, stn_norm_list, sims, out_path):
    """Trái: abnormal sau STN; Phải: top-k normal sau STN + cosine."""
    def to_rgb(t):
        a = t.squeeze(0).detach().cpu().numpy()  # [C,H,W]
        if a.ndim == 3 and a.shape[0] in (1,3):
            a = np.transpose(a, (1,2,0))
        if a.ndim == 2:
            a = np.repeat(a[...,None], 3, axis=2)
        a = (a*255.0).clip(0,255).astype(np.uint8)
        return a
    L = to_rgb(stn_abn)
    Rs = [to_rgb(t) for t in stn_norm_list]

    H = 512
    def rez(im):
        h, w = im.shape[:2]
        nw = int(round(w * H / float(h)))
        return cv2.resize(im, (nw,H), interpolation=cv2.INTER_LINEAR)

    Lr = rez(L)
    Rrs = [rez(x) for x in Rs]
    for i, (im, s) in enumerate(zip(Rrs, sims)):
        txt = f"sim={s:.3f}"
        cv2.putText(im, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)
    gap = 10
    pad = 16
    Wtot = Lr.shape[1] + sum(x.shape[1] for x in Rrs) + gap*(len(Rrs))
    canvas = np.full((H+2*pad, Wtot+2*pad, 3), 40, dtype=np.uint8)
    x = pad
    canvas[pad:pad+H, x:x+Lr.shape[1]] = Lr; x += Lr.shape[1] + gap
    for im in Rrs:
        canvas[pad:pad+H, x:x+im.shape[1]] = im; x += im.shape[1] + gap
    cv2.putText(canvas, "ABNORMAL (STN)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)

def main():
    set_seed()

    # Dọn thư mục cố định bằng xóa cưỡng bức (2B)
    safe_rmtree(VIS_DIR); os.makedirs(VIS_DIR, exist_ok=True)
    safe_rmtree(OUT_DIR); os.makedirs(OUT_DIR, exist_ok=True)

    print("[Load] model:", MODEL_WEIGHTS)
    ymodel = YOLO(MODEL_WEIGHTS)
    core = ymodel.model  # nn.Module
    device = next(core.parameters()).device

    # Chọn module feature sau STN bằng 1 ảnh probe
    probe = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32, device=device)
    feat_mod = choose_after_stn_module(core, probe)

    # Duyệt train set
    pairs, data_y = list_images_and_labels(DATA_YAML)
    print(f"[Data] total train images: {len(pairs)}")

    normals, abns = [], []
    for img_path, lbl_path in pairs:
        (abns if is_abnormal(lbl_path) else normals).append((img_path, lbl_path))
    print(f"[Split] abnormal={len(abns)} | normal={len(normals)}")

    # ---- Extract NORMAL ----
    N = len(normals)
    feats_N, meta_N = [], []
    progN = _Prog(N, "NORMAL")
    print(f"[NORMAL] start: {N} images", flush=True)
    for i, (ip, lp) in enumerate(normals, 1):
        chw, info = load_image_resized(ip, IMG_SIZE)
        x = torch.from_numpy(chw).unsqueeze(0).to(device)
        with torch.inference_mode():
            stn_img, fmap, _theta = get_stn_out_and_feat(core, feat_mod, x)
            g = make_descriptor(fmap).squeeze(0).cpu().numpy()
        feats_N.append(g)
        meta_N.append({"im": ip, "lbl": lp, "hw": info["orig_hw"]})
        if (i % PRINT_EVERY) == 0 or i == N:
            rate, eta = progN.tick(i)
            print(f"[NORMAL] {i}/{N} | {rate:.2f} img/s | ETA ~ {eta / 60:.1f} min", flush=True)

    feats_N = np.stack(feats_N, 0) if feats_N else np.zeros((0,1), np.float32)
    np.save(os.path.join(OUT_DIR, "normal_feats.npy"), feats_N)
    with open(os.path.join(OUT_DIR, "normal_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_N, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] normal feats: {feats_N.shape}")

    # ---- Extract ABNORMAL ----
    M = len(abns)
    feats_A, meta_A = [], []
    progA = _Prog(M, "ABNORMAL")
    print(f"[ABNORMAL] start: {M} images", flush=True)
    for i, (ip, lp) in enumerate(abns, 1):
        chw, info = load_image_resized(ip, IMG_SIZE)
        x = torch.from_numpy(chw).unsqueeze(0).to(device)
        with torch.inference_mode():
            stn_img, fmap, _theta = get_stn_out_and_feat(core, feat_mod, x)
            g = make_descriptor(fmap).squeeze(0).cpu().numpy()
        feats_A.append(g)
        meta_A.append({"im": ip, "lbl": lp, "hw": info["orig_hw"]})
        if (i % PRINT_EVERY) == 0 or i == M:
            rate, eta = progA.tick(i)
            print(f"[ABNORMAL] {i}/{M} | {rate:.2f} img/s | ETA ~ {eta / 60:.1f} min", flush=True)

    feats_A = np.stack(feats_A, 0) if feats_A else np.zeros((0,1), np.float32)
    np.save(os.path.join(OUT_DIR, "abn_feats.npy"), feats_A)
    with open(os.path.join(OUT_DIR, "abn_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_A, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] abnormal feats: {feats_A.shape}")

    # ---- Quick visual sanity: 12 panels ----
    if len(feats_A) and len(feats_N):
        K = min(TOPK_VIS, len(feats_N))
        rng = np.random.RandomState(SEED)
        idxs = rng.choice(len(feats_A), size=min(N_VIS_ABN, len(feats_A)), replace=False)
        sims, topk = cosine_topk(feats_A[idxs], feats_N, k=K)

        for ri, a_idx in enumerate(idxs):
            a_path = meta_A[a_idx]["im"]
            chw_a, _ = load_image_resized(a_path, IMG_SIZE)
            x_a = torch.from_numpy(chw_a).unsqueeze(0).to(device)
            stn_a, _, _ = get_stn_out_and_feat(core, feat_mod, x_a)

            norm_ids = topk[ri].tolist()
            stn_norms, scs = [], []
            for j in norm_ids:
                n_path = meta_N[j]["im"]
                chw_n, _ = load_image_resized(n_path, IMG_SIZE)
                x_n = torch.from_numpy(chw_n).unsqueeze(0).to(device)
                stn_n, _, _ = get_stn_out_and_feat(core, feat_mod, x_n)
                stn_norms.append(stn_n)
                scs.append(float(sims[ri, norm_ids[len(stn_norms)-1]]))

            out_path = os.path.join(VIS_DIR, f"pair_panel_{ri:02d}.jpg")
            save_panel(stn_a, stn_norms, scs, out_path)
        print(f"[VIS] saved panels to {VIS_DIR}/")
    else:
        print("[VIS] skip (no feats)")

if __name__ == "__main__":
    main()

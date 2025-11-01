# pairing_build.py
# Step 1: build feature bank (after STN) for normal/abnormal
# Step 2: pair abnormal → top-k normal with AR/size filtering + cosine
# Designed to run in: C:\OneDrive\Study\AI\STN_Final_Term\pairing

from __future__ import annotations
import os, json, math, random, argparse, shutil, stat, time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import cv2
import yaml

# =========== PATH ROOTS ===========
SCRIPT_DIR = Path(__file__).resolve().parent                  # ...\STN_Final_Term\pairing
PROJECT_ROOT = SCRIPT_DIR.parent                              # ...\STN_Final_Term

# =========== CONFIG (bạn có thể chỉnh) ===========
# (1) BANK (trích xuất descriptor sau STN)
MODEL_WEIGHTS = PROJECT_ROOT / "models" / "yolo11m_stn.pt"
DATA_YAML     = PROJECT_ROOT / "dataset.yaml"
IMG_SIZE      = 640
MIN_CHANNELS  = 1024       # chọn fmap sâu (C lớn)
MIN_DOWNS     = 8          # tối thiểu downsample >= 8x
MAX_IMG       = 10**9
SEED          = 42
PRINT_EVERY   = 50
USE_REGIONAL_POOL = True
REG_GRID          = (1, 2, 4)  # 1x1 + 2x2 + 4x4

OUT_DIR_BANK  = SCRIPT_DIR / "bank_out"
VIS_DIR_BANK  = SCRIPT_DIR / "bank_vis"
N_VIS_ABN     = 20     # số abnormal để vẽ
TOPK_VIS      = 3

# (2) PAIR (ghép cặp)
BANK_DIR      = OUT_DIR_BANK
OUT_JSON      = SCRIPT_DIR / "bgpair_map.json"
OUT_STATS     = SCRIPT_DIR / "bgpair_stats.json"
VIS_DIR_PAIR  = SCRIPT_DIR / "bgpair_vis"
TOPK_PER_ABN  = 3
TAU_AR        = 0.00   # |log(H/W)_A - log(H/W)_N| <= TAU_AR
TAU_SIZE      = 0.01   # max(|Hn-Ha|/Ha, |Wn-Wa|/Wa) <= TAU_SIZE
TAU_SIM_MIN   = 0.90   # cosine tối thiểu

# =========== TIỆN ÍCH ===========
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def safe_rmtree(path: Path | str):
    path_str = str(path)
    if not os.path.isdir(path_str):
        return
    def _on_rm_error(func, p, _):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass
    shutil.rmtree(path_str, onerror=_on_rm_error)

def load_data_yaml(path: Path | str) -> dict:
    with open(str(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def list_images_and_labels(data_yaml: Path | str) -> List[Tuple[str, str]]:
    """
    Đọc 'train' trong dataset.yaml. Trả [(img_path, lbl_path)].
    """
    def _collect_from_dir(imgdir: str) -> List[str]:
        exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".pnm",".pgm")
        imgs = []
        for root, _, files in os.walk(imgdir):
            for fn in files:
                if fn.lower().endswith(exts):
                    imgs.append(os.path.join(root, fn))
        return imgs

    def _label_path(img_path: str) -> str:
        p = Path(img_path)
        for tok in ("images", "Images"):
            if tok in p.parts:
                idx = p.parts.index(tok)
                rel = Path(*p.parts[idx+1:]).with_suffix(".txt")
                return str(Path(*p.parts[:idx], "labels", rel))
        return str(p.parent.parent / "labels" / (p.stem + ".txt"))

    y = load_data_yaml(data_yaml)
    trains = y.get("train", [])
    if not isinstance(trains, list):
        trains = [trains]

    img_paths = []
    for item in trains:
        if not item:
            continue
        p = str(item)
        if os.path.isdir(p):
            img_paths += _collect_from_dir(p)
        elif os.path.isfile(p):
            # file list (.txt) hay 1 ảnh
            if p.lower().endswith(".txt"):
                with open(p, "r", encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if ln:
                            img_paths.append(ln)
            else:
                img_paths.append(p)

    pairs = []
    for ip in img_paths:
        pairs.append((ip, _label_path(ip)))
    return pairs

def read_yolo_labels(lbl_path: str) -> List[List[float]]:
    if not os.path.exists(lbl_path): return []
    rows = []
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().split()
            if len(t) >= 5:
                cls = int(float(t[0])); cx, cy, w, h = map(float, t[1:5])
                rows.append([cls, cx, cy, w, h])
    return rows

def is_abnormal(lbl_path: str) -> bool:
    rows = read_yolo_labels(lbl_path)
    return len(rows) > 0

def load_image_resized(img_path: str, imgsz: int = IMG_SIZE) -> Tuple[np.ndarray, dict]:
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(img_path)
    H0, W0 = im.shape[:2]
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, 2)
    im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    chw = np.transpose(im, (2, 0, 1)).astype(np.float32) / 255.0
    return chw, {"orig_hw": (H0, W0)}

# =========== HOOKS: bắt STN output và 1 feature deep ===========
def choose_after_stn_module(core: nn.Module, sample_x: torch.Tensor) -> nn.Module:
    """
    Chọn module đầu tiên SAU STN cho out 4D, C >= MIN_CHANNELS, downsample >= MIN_DOWNS.
    """
    try:
        from ultralytics.nn.modules.block import SpatialTransformer as STN
    except Exception:
        STN = None

    stn_seen = False
    outs = []
    hooks = []

    # chỉ hook backbone/neck
    modules = list(core.model.modules()) if hasattr(core, "model") else list(core.modules())

    def mk_hook(m):
        def _h(_m, _in, o):
            if torch.is_tensor(o):
                outs.append((m, o))
            elif isinstance(o, (list, tuple)) and o:
                # ghi nhận tensor đầu tiên
                for item in o:
                    if torch.is_tensor(item):
                        outs.append((m, item)); break
        return _h

    for m in modules:
        name = m.__class__.__name__.lower()
        if "detect" in name:         # bỏ head
            continue
        hooks.append(m.register_forward_hook(mk_hook(m)))

    # chạy 1 lượt để thu out
    core.eval()
    with torch.inference_mode():
        _ = core(sample_x)

    for h in hooks:
        try: h.remove()
        except Exception: pass

    # duyệt theo thứ tự xuất hiện, tìm “sau STN”
    picked = None
    seen = False
    for m, o in outs:
        if (hasattr(m, "__class__") and (m.__class__.__name__ in {"SpatialTransformer","STN","SpatialTransformer2D"})) or (STN and isinstance(m, STN)):
            seen = True
            continue
        if not seen:
            continue
        if torch.is_tensor(o) and o.dim() == 4:
            B, C, H, W = o.shape
            down_ok = min(IMG_SIZE / max(H,1), IMG_SIZE / max(W,1)) >= MIN_DOWNS
            if C >= MIN_CHANNELS or down_ok:
                picked = m
                break

    # fallback: chọn tensor 4D có C lớn nhất “sau STN”
    if picked is None:
        best = (-1, None)
        seen = False
        for m, o in outs:
            if (hasattr(m, "__class__") and (m.__class__.__name__ in {"SpatialTransformer","STN","SpatialTransformer2D"})) or (STN and isinstance(m, STN)):
                seen = True; continue
            if not seen: continue
            if torch.is_tensor(o) and o.dim()==4:
                C = o.shape[1]
                if C > best[0]:
                    best = (C, m)
        picked = best[1] if best[1] is not None else core

    return picked

def get_stn_out_and_feat(core: nn.Module, feat_mod: nn.Module, x: torch.Tensor):
    """
    Trả:
      - stn_img (B,H,W,3) uint8 (sau STN)
      - fmap     (B,C,h,w) tensor (từ feat_mod)
      - theta    (B,2,3)   nếu bắt được; ngược lại None
    """
    stn_img = None
    theta   = None
    fmap    = None

    def stn_hook(m, _in, out):
        nonlocal stn_img, theta
        # out có thể là tensor hoặc (x_t, theta)
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            x_t = out[0]
            if len(out) >= 2:
                theta = out[1]
        else:
            x_t = out
        if torch.is_tensor(x_t):
            vis = x_t.detach().float().clamp(0,1).cpu().numpy()
            if vis.ndim == 4:  # B,C,H,W
                vis = np.transpose(vis, (0,2,3,1))
            stn_img = (vis * 255.0).round().astype(np.uint8)
        # một số STN lưu theta vào thuộc tính
        if theta is None and hasattr(m, "theta"):
            t = getattr(m, "theta")
            if torch.is_tensor(t):
                theta = t.detach()

    def feat_hook(m, _in, out):
        nonlocal fmap
        if torch.is_tensor(out):
            fmap = out.detach()
        elif isinstance(out, (list, tuple)) and out and torch.is_tensor(out[0]):
            fmap = out[0].detach()

    # gắn hook
    stn_modules = []
    for m in core.modules():
        if m.__class__.__name__ in {"SpatialTransformer","STN","SpatialTransformer2D","SpatialTransformerBlock"}:
            stn_modules.append(m)
    stn_handles = [m.register_forward_hook(stn_hook) for m in stn_modules]
    feat_handle = feat_mod.register_forward_hook(feat_hook)

    # forward
    core.eval()
    with torch.inference_mode():
        _ = core(x)

    # tháo hook
    for h in stn_handles:
        try: h.remove()
        except Exception: pass
    try: feat_handle.remove()
    except Exception: pass

    return stn_img, fmap, theta

# =========== POOLING & DESCRIPTOR ===========
def global_avg_pool(fmap: torch.Tensor) -> torch.Tensor:
    return torch.mean(fmap, dim=(2,3), keepdim=False)

def regional_pool(fmap: torch.Tensor, grids=(1,2,4)) -> torch.Tensor:
    """
    SPoC-ish: chia lưới (1,2,4), avg-pool theo từng ô rồi concat.
    """
    B, C, H, W = fmap.shape
    outs = []
    for g in grids:
        gh, gw = g, g
        hs, ws = H//gh, W//gw
        for iy in range(gh):
            for ix in range(gw):
                ys, xs = iy*hs, ix*ws
                ye, xe = (H if iy==gh-1 else (iy+1)*hs), (W if ix==gw-1 else (ix+1)*ws)
                patch = fmap[:, :, ys:ye, xs:xe]
                outs.append(patch.mean(dim=(2,3)))
    return torch.cat(outs, dim=1) if outs else fmap.mean(dim=(2,3))

def make_descriptor(fmap: torch.Tensor) -> torch.Tensor:
    x = fmap
    if USE_REGIONAL_POOL:
        x = regional_pool(x, REG_GRID)
    else:
        x = global_avg_pool(x)
    x = torch.nn.functional.normalize(x, dim=1)
    return x

def cosine_topk(Q: np.ndarray, K: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Q: [M,D], K: [N,D] -> cosine similarity topk index cho mỗi query
    """
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)
    sims = Qn @ Kn.T  # [M,N]
    topk = np.argsort(-sims, axis=1)[:, :k]
    return sims, topk

# =========== PANEL VIZ ===========
def to_rgb(im):
    if im.ndim==2: im = np.repeat(im[...,None], 3, 2)
    if im.shape[2]==1: im = np.repeat(im, 3, 2)
    return im

def rez(im, H=448):
    h,w = im.shape[:2]
    if h == 0: return np.zeros((H,H,3), np.uint8)
    nw = int(round(w*H/float(h)))
    return cv2.resize(im, (nw,H), interpolation=cv2.INTER_LINEAR)

def save_panel(stn_abn: np.ndarray, stn_norms: List[np.ndarray], scores: List[float], out_path: str):
    TILE_H = 448
    rows = [rez(to_rgb(stn_abn[0]))]  # stn_abn: (B,H,W,3)
    for i, norm_img in enumerate(stn_norms):
        rows.append(rez(to_rgb(norm_img[0])))
    # nối ngang, thêm text
    cats = rows
    pad = 8
    Wsum = sum(r.shape[1] for r in cats) + pad*(len(cats)-1)
    panel = np.ones((TILE_H, Wsum, 3), np.uint8) * 255
    x = 0
    for i, r in enumerate(cats):
        panel[:, x:x+r.shape[1]] = r
        # text
        if i == 0:
            cv2.putText(panel, "ABN", (x+10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        else:
            sc = scores[i-1] if i-1 < len(scores) else 0.0
            cv2.putText(panel, f"#{i}  sim={sc:.3f}", (x+10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (34,139,34), 2, cv2.LINE_AA)
        x += r.shape[1] + pad
    cv2.imwrite(str(out_path), panel)

# =========== WARP BOX XYWH → XYXY SAU STN (để vẽ) ===========
def pix2norm(p,L): return (p/max(L-1,1)*2.0)-1.0
def norm2pix(pn,L): return (pn+1.0)*0.5*(L-1)

def warp_xywh_to_stn_xyxy(bxywh_px: np.ndarray, W: int, H: int, theta_2x3: Optional[torch.Tensor]):
    if theta_2x3 is None or (torch.is_tensor(theta_2x3) and theta_2x3.numel()==0):
        x,y,w,h = [bxywh_px[:,i] for i in range(4)]
        x1,y1 = x-w/2, y-h/2
        x2,y2 = x+w/2, y+h/2
        return np.stack([x1,y1,x2,y2],1)

    tb = torch.as_tensor(bxywh_px, dtype=torch.float32)
    x,y,w,h = tb.unbind(-1)
    x1,y1 = x-w/2, y-h/2; x2,y2 = x+w/2, y+h/2
    Xs = torch.stack([x1,x2,x2,x1],1); Ys = torch.stack([y1,y1,y2,y2],1)
    Xn,Yn = pix2norm(Xs,W), pix2norm(Ys,H)

    th = torch.as_tensor(theta_2x3, dtype=torch.float32)
    if th.dim()==3: th = th[0]
    A = torch.tensor([[th[0,0],th[0,1],th[0,2]],
                      [th[1,0],th[1,1],th[1,2]],
                      [0.,0.,1.]], dtype=torch.float32, device=th.device)
    try:
        Ainv = torch.linalg.inv(A)
    except torch.linalg.LinAlgError:
        Ainv = torch.eye(3, dtype=torch.float32, device=th.device)

    ones = torch.ones_like(Xn)
    Pin = torch.stack([Xn,Yn,ones],1)     # [N,3,4]
    Pout = torch.einsum("ij,njk->nik", Ainv, Pin)  # [N,3,4]
    Pout = Pout/(Pout[:,2:3,:]+1e-9)
    Xo,Yo = Pout[:,0,:], Pout[:,1,:]
    Xp, Yp = norm2pix(Xo,W), norm2pix(Yo,H)
    x1p, x2p = Xp.min(1).values, Xp.max(1).values
    y1p, y2p = Yp.min(1).values, Yp.max(1).values
    x1p = x1p.clamp(0,W-1); y1p = y1p.clamp(0,H-1)
    x2p = x2p.clamp(1,W);   y2p = y2p.clamp(1,H)
    return torch.stack([x1p,y1p,x2p,y2p],1).cpu().numpy()

# =========== CLASS NAMES ===========
def get_class_names(dataset_yaml: Path | str) -> List[str]:
    with open(str(dataset_yaml), "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    names = y.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return names if isinstance(names, list) else []

def label_path_from_image(img_path: str) -> str:
    p = Path(img_path)
    for tok in ("images","Images"):
        if tok in p.parts:
            idx = p.parts.index(tok)
            rel = Path(*p.parts[idx+1:]).with_suffix(".txt")
            return str(Path(*p.parts[:idx], "labels", rel))
    return str(p.parent.parent / "labels" / (p.stem + ".txt"))

# =========== BƯỚC 1: BUILD BANK ===========
def step_build_bank(weights: Path | str = MODEL_WEIGHTS, data_yaml: Path | str = DATA_YAML):
    from ultralytics import YOLO
    set_seed(SEED)

    pairs = list_images_and_labels(data_yaml)
    if len(pairs) == 0:
        raise SystemExit("Không tìm thấy ảnh trong 'train' của dataset.yaml")

    # tách normal/abnormal
    normals, abns = [], []
    for ip, lp in pairs[:MAX_IMG]:
        (abns if is_abnormal(lp) else normals).append((ip, lp))
    print(f"[SPLIT] normal={len(normals)} | abnormal={len(abns)}")

    # load model core
    yolo = YOLO(str(weights))
    core: nn.Module = yolo.model
    device = next(core.parameters()).device
    core.to(device).eval()

    # chọn 1 module feature sau STN bằng dynamic hook
    # dùng 1 ảnh normal làm sample
    sample_chw, _ = load_image_resized(normals[0][0] if normals else abns[0][0], IMG_SIZE)
    sample_x = torch.from_numpy(sample_chw).unsqueeze(0).to(device)
    feat_mod = choose_after_stn_module(core, sample_x)
    print(f"[HOOK] feature module = {feat_mod.__class__.__name__}")

    # chuẩn bị output
    safe_rmtree(OUT_DIR_BANK); os.makedirs(str(OUT_DIR_BANK), exist_ok=True)
    safe_rmtree(VIS_DIR_BANK); os.makedirs(str(VIS_DIR_BANK), exist_ok=True)

    # ===== NORMAL =====
    feats_N, meta_N = [], []
    t0 = time.time()
    for i, (ip, lp) in enumerate(normals, 1):
        try:
            chw, info = load_image_resized(ip, IMG_SIZE)
            x = torch.from_numpy(chw).unsqueeze(0).to(device)
            with torch.inference_mode():
                stn_img, fmap, _theta = get_stn_out_and_feat(core, feat_mod, x)
                g = make_descriptor(fmap).squeeze(0).cpu().numpy()
            feats_N.append(g)
            meta_N.append({"im": ip, "lbl": lp, "hw": info["orig_hw"]})
        except Exception as e:
            print(f"[WARN] normal error: {ip} -> {e}")
        if (i % PRINT_EVERY)==0 or i==len(normals):
            rate = i/max(time.time()-t0, 1e-9)
            print(f"[NORMAL] {i}/{len(normals)} | {rate:.2f} img/s")

    feats_N = np.stack(feats_N, 0) if feats_N else np.zeros((0,1), np.float32)
    np.save(os.path.join(str(OUT_DIR_BANK), "normal_feats.npy"), feats_N)
    with open(os.path.join(str(OUT_DIR_BANK), "normal_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_N, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] normal feats: {feats_N.shape}")

    # ===== ABNORMAL =====
    feats_A, meta_A = [], []
    t0 = time.time()
    for i, (ip, lp) in enumerate(abns, 1):
        try:
            chw, info = load_image_resized(ip, IMG_SIZE)
            x = torch.from_numpy(chw).unsqueeze(0).to(device)
            with torch.inference_mode():
                stn_img, fmap, _theta = get_stn_out_and_feat(core, feat_mod, x)
                g = make_descriptor(fmap).squeeze(0).cpu().numpy()
            feats_A.append(g)
            meta_A.append({"im": ip, "lbl": lp, "hw": info["orig_hw"]})
        except Exception as e:
            print(f"[WARN] abnormal error: {ip} -> {e}")
        if (i % PRINT_EVERY)==0 or i==len(abns):
            rate = i/max(time.time()-t0, 1e-9)
            print(f"[ABN] {i}/{len(abns)} | {rate:.2f} img/s")

    feats_A = np.stack(feats_A, 0) if feats_A else np.zeros((0,1), np.float32)
    np.save(os.path.join(str(OUT_DIR_BANK), "abn_feats.npy"), feats_A)
    with open(os.path.join(str(OUT_DIR_BANK), "abn_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_A, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] abnormal feats: {feats_A.shape}")

    # ===== VIS quick sanity (N_VIS_ABN panels) =====
    if len(feats_A)>0 and len(feats_N)>0:
        K = min(TOPK_VIS, len(feats_N))
        rng = np.random.RandomState(SEED)
        idxs = rng.choice(len(feats_A), size=min(N_VIS_ABN, len(feats_A)), replace=False)
        sims, topk = cosine_topk(feats_A[idxs], feats_N, k=K)

        for ridx, a_idx in enumerate(idxs):
            try:
                a_path = meta_A[a_idx]["im"]
                chw_a, _ = load_image_resized(a_path, IMG_SIZE)
                x_a = torch.from_numpy(chw_a).unsqueeze(0).to(device)
                stn_a, _, _ = get_stn_out_and_feat(core, feat_mod, x_a)

                stn_norms, scs = [], []
                for j in topk[ridx]:
                    n_path = meta_N[j]["im"]
                    chw_n, _ = load_image_resized(n_path, IMG_SIZE)
                    x_n = torch.from_numpy(chw_n).unsqueeze(0).to(device)
                    stn_n, _, _ = get_stn_out_and_feat(core, feat_mod, x_n)
                    stn_norms.append(stn_n)
                    scs.append(float((feats_A[a_idx]/(np.linalg.norm(feats_A[a_idx])+1e-9)) @ (feats_N[j]/(np.linalg.norm(feats_N[j])+1e-9))))
                out_path = os.path.join(str(VIS_DIR_BANK), f"bank_panel_{ridx:02d}.jpg")
                save_panel(stn_a, stn_norms, scs, out_path)
            except Exception as e:
                print(f"[WARN] panel error: {e}")
        print(f"[VIS] saved panels to {VIS_DIR_BANK}/")
    else:
        print("[VIS] skip bank panels (no feats)")

# =========== BƯỚC 2: GHÉP CẶP ===========
def read_bank(bank_dir: Path | str):
    bd = str(bank_dir)
    nf = np.load(os.path.join(bd, "normal_feats.npy"))
    af = np.load(os.path.join(bd, "abn_feats.npy"))
    with open(os.path.join(bd, "normal_meta.json"), "r", encoding="utf-8") as f:
        nmeta = json.load(f)
    with open(os.path.join(bd, "abn_meta.json"), "r", encoding="utf-8") as f:
        ameta = json.load(f)
    return nf, nmeta, af, ameta

def make_mask_ar_size(nmeta, ameta, tau_ar, tau_sz):
    A, N = len(ameta), len(nmeta)
    mask = np.zeros((A,N), dtype=bool)
    arN, sizeN = [], []
    for m in nmeta:
        H,W = m["hw"]; arN.append(math.log((H+1e-6)/(W+1e-6))); sizeN.append((H,W))
    arN = np.asarray(arN, np.float32); sizeN = np.asarray(sizeN, np.float32)
    for i, ma in enumerate(ameta):
        Ha,Wa = ma["hw"]; arA = math.log((Ha+1e-6)/(Wa+1e-6))
        cond_ar = np.abs(arN - arA) <= tau_ar
        dH = np.abs(sizeN[:,0]-Ha)/(Ha+1e-9)
        dW = np.abs(sizeN[:,1]-Wa)/(Wa+1e-9)
        cond_sz = np.maximum(dH,dW) <= tau_sz
        mask[i] = cond_ar & cond_sz
    return mask

def draw_pair_panels_for_review(weights: Path | str, pairs: List[Tuple[str,List[str],List[float]]], out_dir: Path | str, data_yaml: Path | str):
    from ultralytics import YOLO
    safe_rmtree(out_dir); os.makedirs(str(out_dir), exist_ok=True)
    yolo = YOLO(str(weights)); core = yolo.model
    device = next(core.parameters()).device; core.to(device).eval()

    # chọn module feature (1 ảnh bất kỳ)
    any_img = pairs[0][0] if pairs else None
    if any_img is None:
        return
    chw, _ = load_image_resized(any_img, IMG_SIZE)
    sample_x = torch.from_numpy(chw).unsqueeze(0).to(device)
    feat_mod = choose_after_stn_module(core, sample_x)

    for ridx, (abn_path, norm_paths, sims) in enumerate(pairs):
        try:
            chw_a, _ = load_image_resized(abn_path, IMG_SIZE)
            x_a = torch.from_numpy(chw_a).unsqueeze(0).to(device)
            stn_a, _, _ = get_stn_out_and_feat(core, feat_mod, x_a)

            stn_norms, scs = [], []
            for j, npth in enumerate(norm_paths):
                chw_n, _ = load_image_resized(npth, IMG_SIZE)
                x_n = torch.from_numpy(chw_n).unsqueeze(0).to(device)
                stn_n, _, _ = get_stn_out_and_feat(core, feat_mod, x_n)
                stn_norms.append(stn_n)
                scs.append(float(sims[j] if j < len(sims) else 0.0))
            out_path = os.path.join(str(out_dir), f"pair_{ridx:04d}.jpg")
            save_panel(stn_a, stn_norms, scs, out_path)
        except Exception as e:
            print(f"[WARN] draw pair panel error: {e}")

def step_pair_and_export(weights: Path | str = MODEL_WEIGHTS, data_yaml: Path | str = DATA_YAML):
    nf, nmeta, af, ameta = read_bank(BANK_DIR)
    A = af.shape[0] if af.ndim==2 else 0
    N = nf.shape[0] if nf.ndim==2 else 0
    print(f"[BANK] abnormal={A} | normal={N}")
    if A==0 or N==0:
        raise SystemExit("Bank rỗng. Hãy chạy --mode bank trước.")

    mask = make_mask_ar_size(nmeta, ameta, TAU_AR, TAU_SIZE)  # [A,N]

    # cosine
    Qn = af/(np.linalg.norm(af, axis=1, keepdims=True)+1e-9)
    Kn = nf/(np.linalg.norm(nf, axis=1, keepdims=True)+1e-9)
    sims = Qn @ Kn.T  # [A,N]

    bgpair = {}
    sims_kept = []
    cover = 0
    for i in range(A):
        abn_path = ameta[i]["im"]
        cand = np.where(mask[i])[0]
        if cand.size == 0:
            bgpair[abn_path] = []
            continue
        cand_sims = sims[i, cand]
        order = np.argsort(-cand_sims)
        kept_idx = []
        for j in order:
            if cand_sims[j] >= TAU_SIM_MIN:
                kept_idx.append(cand[j])
            if len(kept_idx) >= TOPK_PER_ABN:
                break
        if kept_idx:
            cover += 1
            sims_kept.append(float(cand_sims[order[0]]))
            bgpair[abn_path] = [nmeta[j]["im"] for j in kept_idx]
        else:
            bgpair[abn_path] = []

    with open(str(OUT_JSON), "w", encoding="utf-8") as f:
        json.dump(bgpair, f, ensure_ascii=False, indent=2)

    hist_bins = [0,0.5,0.6,0.7,0.8,0.9,1.0]
    hist = np.histogram(np.asarray(sims_kept) if sims_kept else np.asarray([0.0]), bins=hist_bins)[0].tolist()
    stats = {
        "abnormal_total": A,
        "normal_total": N,
        "coverage": cover,
        "coverage_rate": (0.0 if A==0 else round(cover/float(A), 4)),
        "filters": {"tau_ar": TAU_AR, "size": TAU_SIZE, "sim": TAU_SIM_MIN},
        "topk": TOPK_PER_ABN,
        "max_sim_hist": hist
    }
    with open(str(OUT_STATS), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {OUT_JSON} | {OUT_STATS}")
    print(f"[COVER] {cover}/{A} abnormal có ≥1 normal với sim ≥ {TAU_SIM_MIN}")

    # Visual hoá cặp (mỗi abnormal một hàng)
    vis_pairs = []
    for i in range(A):
        abn_path = ameta[i]["im"]
        jlist = []
        sims_row = []
        for npth in bgpair.get(abn_path, []):
            try:
                j = next(k for k, mm in enumerate(nmeta) if mm["im"] == npth)
                jlist.append(npth)
                sims_row.append(float(sims[i, j]))
            except StopIteration:
                pass
        vis_pairs.append((abn_path, jlist, sims_row))

    safe_rmtree(VIS_DIR_PAIR); os.makedirs(str(VIS_DIR_PAIR), exist_ok=True)
    draw_pair_panels_for_review(weights, vis_pairs, VIS_DIR_PAIR, data_yaml)
    print(f"[VIS] saved to {VIS_DIR_PAIR}/")

# =========== CLI ===========
def parse_args():
    p = argparse.ArgumentParser("Build bank + Pairing (abn→normal) in one script")
    p.add_argument("--mode", choices=["bank","pair","all"], default="all",
                   help="bank: build features; pair: pairing; all: both")
    p.add_argument("--weights", default=str(MODEL_WEIGHTS))
    p.add_argument("--data",    default=str(DATA_YAML))
    return p.parse_args()

def main():
    args = parse_args()
    # override nếu người dùng truyền đường dẫn khác
    global MODEL_WEIGHTS, DATA_YAML
    MODEL_WEIGHTS = Path(args.weights)
    DATA_YAML     = Path(args.data)

    if args.mode in ("bank","all"):
        step_build_bank(MODEL_WEIGHTS, DATA_YAML)
    if args.mode in ("pair","all"):
        step_pair_and_export(MODEL_WEIGHTS, DATA_YAML)

if __name__ == "__main__":
    main()

# build_bgpair_map.py
# Pair abnormal↔normal by AR/size filters + cosine on STN embeddings (offline)
import os, json, math, shutil, stat
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO
import yaml

# CHW float(0..1) -> HWC uint8 contiguous (+ writeable) cho OpenCV
def to_hwc_uint8_contig(t):
    if hasattr(t, "detach"):
        a = t.detach().cpu().numpy()
    else:
        a = t
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    elif a.ndim == 2:
        a = a[..., None]
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    a = (a * 255.0).clip(0, 255).astype(np.uint8)
    a = np.ascontiguousarray(a)
    a.setflags(write=True)
    return a

# bảng màu (BGR) — đủ khác biệt cho 15 lớp, lặp lại nếu >15
PALETTE = [
    (36,255,12), (0,215,255), (255,0,0), (0,114,255), (255,178,29),
    (87,87,255), (255,51,255), (0,255,0), (255,0,127), (0,140,255),
    (204,72,63), (0,255,255), (255,255,0), (180,130,70), (153,51,255)
]

def get_class_names(data_yaml):
    # đọc 'names' từ data.yaml (list hoặc dict)
    import yaml
    with open(data_yaml, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names", [])
    if isinstance(names, dict):
        # đảm bảo sắp xếp theo khóa số
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return names

def draw_label(img, x1, y1, text, color):
    pad = 4
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICK)
    x2, y2 = x1 + tw + 2*pad, y1 - th - 2*pad
    y2 = max(y2, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.putText(img, text, (x1 + pad, y1 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,0,0), FONT_THICK, cv2.LINE_AA)

# ===================== CONFIG (chỉnh ở đây) =====================
MODEL_WEIGHTS = r"C:\OneDrive\Study\AI\STN_Final_Term\models\yolo11m_stn.pt"
DATA_YAML     = r"C:\OneDrive\Study\AI\STN_Final_Term\dataset.yaml"
IMG_SIZE      = 1024
BANK_DIR      = "bank_out"
OUT_JSON      = "bgpair_map.json"
OUT_STATS     = "bgpair_stats.json"
VIS_DIR       = "bgpair_vis"
TOPK_PER_ABN  = 3

# --- Feature pick for STN-visual (đồng bộ với Step 1) ---
MIN_CHANNELS = 512
MIN_DOWNS    = 8       # yêu cầu Hf,Wf <= H//8
FEATURE_PICK = "deepest"  # deepest | first | index
FEATURE_INDEX = None      # dùng khi FEATURE_PICK="index"

# Xiết ngưỡng để ghép cặp giống hơn
TAU_AR        = 0.00   # |log(H/W)_A - log(H/W)_N| ≤ TAU_AR (chặt hơn)
TAU_SIZE      = 0.01   # max(|Hn-Ha|/Ha, |Wn-Wa|/Wa) ≤ TAU_SIZE
TAU_SIM_MIN   = 0.9  # cosine tối thiểu (chặt hơn)
N_VIS         = 50
# Vẽ
TILE_H        = 720    # chiều cao mỗi tile (to hơn)
THICK         = 3      # độ dày bbox
FONT_SCALE    = 0.9
FONT_THICK    = 2

# ===============================================================

def safe_rmtree(path):
    if not os.path.isdir(path): return
    def _on_rm_error(func, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception: pass
    shutil.rmtree(path, onerror=_on_rm_error)

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_bank(bank_dir):
    nf = np.load(os.path.join(bank_dir, "normal_feats.npy"))
    af = np.load(os.path.join(bank_dir, "abn_feats.npy"))
    with open(os.path.join(bank_dir, "normal_meta.json"), "r", encoding="utf-8") as f:
        nmeta = json.load(f)
    with open(os.path.join(bank_dir, "abn_meta.json"), "r", encoding="utf-8") as f:
        ameta = json.load(f)
    return nf, nmeta, af, ameta

def normalize_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / n

def cosine_topk_query_to_keys(q, K, k, mask_idx=None):
    qn = q / (np.linalg.norm(q) + 1e-9)
    Kn = normalize_rows(K)
    sims = Kn @ qn
    if mask_idx is not None:
        sims = np.where(mask_idx, sims, -1.0)
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def load_img_resize_letterbox_3c(path, size):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None: raise FileNotFoundError(path)
    h, w = im.shape[:2]
    scale = min(size/float(h), size/float(w))
    nh, nw = int(round(h*scale)), int(round(w*scale))
    imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size), dtype=np.uint8)
    top = (size - nh)//2; left = (size - nw)//2
    canvas[top:top+nh, left:left+nw] = imr
    chw = np.stack([canvas, canvas, canvas], axis=0).astype(np.float32) / 255.0  # [3,H,W]
    return chw

def choose_after_stn_module(core: nn.Module, sample_tensor: torch.Tensor,
                            minC=MIN_CHANNELS, minDown=MIN_DOWNS):
    from ultralytics.nn.modules.block import SpatialTransformer as STN
    hooks, outs, stn_seen = [], [], False

    def mk_hook(m):
        def _h(_m, _i, o):
            if torch.is_tensor(o): outs.append((m, o))
            elif isinstance(o, (list, tuple)):
                for t in o:
                    if torch.is_tensor(t): outs.append((m, t))
        return _h

    for m in list(core.modules()):
        if isinstance(m, STN):
            stn_seen = True
            continue
        if not stn_seen:  # chỉ hook các layer SAU STN
            continue
        if hasattr(m, "forward"):
            hooks.append(m.register_forward_hook(mk_hook(m)))

    was = core.training
    core.eval()
    with torch.no_grad(): _ = core(sample_tensor)
    for h in hooks: h.remove()
    core.train(was)

    H, W = sample_tensor.shape[-2], sample_tensor.shape[-1]

    # thu hết các out 4D sau STN
    cands = []
    for m, o in outs:
        if torch.is_tensor(o) and o.dim() == 4:
            _, C, Hf, Wf = o.shape
            down_ok = (Hf <= H // minDown) and (Wf <= W // minDown)
            if C >= minC and down_ok:
                cands.append((m, (C, Hf, Wf)))

    # nếu không có ứng viên đủ tiêu chí, hạ tiêu chí nhưng vẫn chọn layer SÂU NHẤT
    if not cands:
        for m, o in outs:
            if torch.is_tensor(o) and o.dim() == 4:
                C, Hf, Wf = o.shape[1:]
                cands.append((m, (C, Hf, Wf)))

    if not cands:
        raise RuntimeError("No 4D feature after STN.")

    # chọn theo chiến lược
    if FEATURE_PICK == "index" and FEATURE_INDEX is not None:
        k = max(0, min(FEATURE_INDEX, len(cands)-1))
        chosen = cands[k]
    elif FEATURE_PICK == "first":
        chosen = cands[0]
    else:  # deepest: ưu tiên downsample lớn nhất (Hf*Wf nhỏ), rồi C lớn
        chosen = sorted(cands, key=lambda it: (it[1][1]*it[1][2], -it[1][0]))[0]

    print(f"[Probe] after-STN feature: {chosen[0].__class__.__name__} shape={chosen[1]}")
    return chosen[0]


def get_stn_img_and_theta(core: nn.Module, x: torch.Tensor):
    from ultralytics.nn.modules.block import SpatialTransformer as STN
    stn=None
    for m in core.modules():
        if isinstance(m, STN): stn=m; break
    if stn is None: raise RuntimeError("STN not found.")
    out_pack={"img":None,"theta":None}
    def stn_hook(mod, inp, out):
        out_pack["img"]=out.detach()
        th=None
        for name in ("theta","_theta","last_theta"):
            val=getattr(mod,name,None)
            if isinstance(val, torch.Tensor): th=val.detach(); break
        if th is None:
            th = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=out.device)
        out_pack["theta"]=th
    h = stn.register_forward_hook(stn_hook)
    was = core.training; core.eval()
    with torch.no_grad():
        _ = core(x)
    h.remove(); core.train(was)
    return out_pack["img"], out_pack["theta"]

def pix2norm(px, L):  return (px / max(L - 1, 1) * 2.0) - 1.0
def norm2pix(pn, L):  return (pn + 1.0) * 0.5 * (L - 1)

def warp_xywh_to_stn_xyxy(bxywh_px, W, H, theta_2x3):
    if theta_2x3 is None or (torch.is_tensor(theta_2x3) and theta_2x3.numel()==0):
        x,y,w,h = [bxywh_px[:,i] for i in range(4)]
        x1,y1 = x - w/2, y - h/2
        x2,y2 = x + w/2, y + h/2
        return np.stack([x1,y1,x2,y2], axis=1)
    tb = torch.as_tensor(bxywh_px, dtype=torch.float32)
    x,y,w,h = tb.unbind(-1)
    x1,y1 = x - w/2, y - h/2
    x2,y2 = x + w/2, y + h/2
    Xs = torch.stack([x1,x2,x2,x1], dim=1)
    Ys = torch.stack([y1,y1,y2,y2], dim=1)
    Xn, Yn = pix2norm(Xs, W), pix2norm(Ys, H)
    th = torch.as_tensor(theta_2x3, dtype=torch.float32)
    if th.dim()==3: th = th[0]
    A  = torch.tensor([[th[0,0], th[0,1], th[0,2]],
                       [th[1,0], th[1,1], th[1,2]],
                       [0.,0.,1.]], dtype=torch.float32)
    Ainv = torch.linalg.inv(A)
    ones = torch.ones_like(Xn)
    Pin  = torch.stack([Xn,Yn,ones], dim=1)
    Pout = torch.einsum("ij,njk->nik", Ainv, Pin)
    Xo,Yo = Pout[:,0,:], Pout[:,1,:]
    Xp, Yp = norm2pix(Xo, W), norm2pix(Yo, H)
    x1p, x2p = Xp.min(dim=1).values, Xp.max(dim=1).values
    y1p, y2p = Yp.min(dim=1).values, Yp.max(dim=1).values
    x1p = x1p.clamp(0, W-1); y1p = y1p.clamp(0, H-1)
    x2p = x2p.clamp(1, W);   y2p = y2p.clamp(1, H)
    return torch.stack([x1p,y1p,x2p,y2p], dim=1).cpu().numpy()

def read_yolo_labels(lbl_path):
    if not os.path.exists(lbl_path): return []
    rows=[]
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            t=line.strip().split()
            if len(t)<5: continue
            cls=int(float(t[0])); cx,cy,w,h = map(float, t[1:5])
            rows.append([cls, cx, cy, w, h])
    return rows

def label_path_from_image(img_path: str):
    p = Path(img_path)
    for tok in ("images","Images"):
        if tok in p.parts:
            idx=p.parts.index(tok)
            rel = Path(*p.parts[idx+1:]).with_suffix(".txt")
            return str(Path(*p.parts[:idx], "labels", rel))
    return str(p.parent.parent / "labels" / (p.stem + ".txt"))

def make_mask_ar_size(nmeta, ameta, tau_ar, tau_sz):
    A, N = len(ameta), len(nmeta)
    mask = np.zeros((A,N), dtype=bool)
    arN=[]; sizeN=[]
    for m in nmeta:
        H,W = m["hw"]
        arN.append(math.log((H+1e-6)/(W+1e-6)))
        sizeN.append((H,W))
    arN=np.array(arN, np.float32); sizeN=np.array(sizeN, np.float32)
    for i, ma in enumerate(ameta):
        Ha, Wa = ma["hw"]
        arA = math.log((Ha+1e-6)/(Wa+1e-6))
        cond_ar = np.abs(arN - arA) <= tau_ar
        dH = np.abs(sizeN[:,0]-Ha)/(Ha+1e-9)
        dW = np.abs(sizeN[:,1]-Wa)/(Wa+1e-9)
        cond_sz = np.maximum(dH,dW) <= tau_sz
        mask[i] = cond_ar & cond_sz
    return mask

def draw_panel(core, feat_mod, pairs, out_dir, data_yaml):
    safe_rmtree(out_dir); os.makedirs(out_dir, exist_ok=True)
    device = next(core.parameters()).device
    names = get_class_names(data_yaml)

    def rez(img):
        h, w = img.shape[:2]
        nw = int(round(w * TILE_H / float(h)))
        return cv2.resize(img, (nw, TILE_H), interpolation=cv2.INTER_LINEAR)

    for idx, (abn_path, norm_paths, sims) in enumerate(pairs):
        try:
            # ----- ABNORMAL tile -----
            chw_a = load_img_resize_letterbox_3c(abn_path, IMG_SIZE)
            x_a = torch.from_numpy(chw_a).unsqueeze(0).to(device)
            stn_a, theta_a = get_stn_img_and_theta(core, x_a)

            Aimg = to_hwc_uint8_contig(stn_a[0])
            H, W = Aimg.shape[:2]

            rows = read_yolo_labels(label_path_from_image(abn_path))
            # convert YOLO norm -> px (theo kích thước STN output) rồi warp sang STN
            if rows:
                bxywh = [[cx*W, cy*H, w*W, h*H] for cls, cx, cy, w, h in rows]
                abn_xyxy = warp_xywh_to_stn_xyxy(np.array(bxywh, np.float32), W, H, theta_a)
                # vẽ bbox xanh-theo-lớp + label lớp
                for (box, r) in zip(abn_xyxy.astype(int), rows):
                    cls = int(r[0])
                    cname = names[cls] if 0 <= cls < len(names) else f"id{cls}"
                    color = PALETTE[cls % len(PALETTE)]
                    x1,y1,x2,y2 = box
                    x1 = int(np.clip(x1, 0, W-1)); y1 = int(np.clip(y1, 0, H-1))
                    x2 = int(np.clip(x2, 1, W));   y2 = int(np.clip(y2, 1, H))
                    cv2.rectangle(Aimg, (x1,y1), (x2,y2), color, THICK)
                    draw_label(Aimg, x1, y1, cname, color)
            else:
                abn_xyxy = np.zeros((0,4), np.int32)

            A_tile = rez(Aimg)
            cv2.putText(A_tile, "ABN STN (green=GT warped)",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,255,200), 2, cv2.LINE_AA)

            # ----- NORMAL tiles (up to 3) -----
            norm_tiles = []
            for npath, s in list(zip(norm_paths, sims))[:3]:
                chw_n = load_img_resize_letterbox_3c(npath, IMG_SIZE)
                x_n = torch.from_numpy(chw_n).unsqueeze(0).to(device)
                stn_n, _ = get_stn_img_and_theta(core, x_n)

                Nim = to_hwc_uint8_contig(stn_n[0])
                Hn, Wn = Nim.shape[:2]

                # vẽ cùng bbox (dùng màu theo class, nếu có)
                for bi, box in enumerate(abn_xyxy.astype(int)):
                    x1,y1,x2,y2 = box
                    x1c = int(np.clip(x1, 0, Wn-1)); y1c = int(np.clip(y1, 0, Hn-1))
                    x2c = int(np.clip(x2, 1, Wn));   y2c = int(np.clip(y2, 1, Hn))
                    color = (255,200,0)
                    if rows:
                        cls = int(rows[bi][0])
                        color = PALETTE[cls % len(PALETTE)]
                    cv2.rectangle(Nim, (x1c,y1c), (x2c,y2c), color, THICK)

                Nim = rez(Nim)
                cv2.putText(Nim, f"cos={float(s):.3f}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
                norm_tiles.append(Nim)

            # nếu <3 ứng viên thì chèn ô trống
            while len(norm_tiles) < 3:
                blank = np.full((TILE_H, TILE_H, 3), 30, np.uint8)
                cv2.putText(blank, "no candidate", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180,180,180), 2, cv2.LINE_AA)
                norm_tiles.append(blank)

            # ----- compose 2x2 grid -----
            t00 = A_tile
            t01 = norm_tiles[0]
            t10 = norm_tiles[1]
            t11 = norm_tiles[2]

            # chuẩn chiều rộng: pad theo chiều cao TILE_H
            def pad_to_h(img, H=TILE_H):
                h, w = img.shape[:2]
                if h == H: return img
                return cv2.resize(img, (int(round(w*H/h)), H))

            t00, t01, t10, t11 = map(pad_to_h, (t00, t01, t10, t11))
            gap, pad = 12, 16
            row1 = np.concatenate([t00, np.full((TILE_H, gap, 3), 25, np.uint8), t01], axis=1)
            row2 = np.concatenate([t10, np.full((TILE_H, gap, 3), 25, np.uint8), t11], axis=1)
            Wmax = max(row1.shape[1], row2.shape[1])
            # align width
            def align_w(img, Wt):
                h, w = img.shape[:2]
                if w == Wt: return img
                pad_r = Wt - w
                return np.pad(img, ((0,0),(0,pad_r),(0,0)), mode="constant", constant_values=25)
            row1, row2 = align_w(row1, Wmax), align_w(row2, Wmax)
            canvas = np.full((TILE_H*2 + gap + 2*pad, Wmax + 2*pad, 3), 25, np.uint8)
            canvas[pad:pad+TILE_H, pad:pad+Wmax] = row1
            canvas[pad+TILE_H+gap:pad+2*TILE_H+gap, pad:pad+Wmax] = row2

            cv2.imwrite(os.path.join(out_dir, f"pair_panel_{idx:02d}.jpg"), canvas)

        except Exception as e:
            print(f"[VIS] fail {idx}: {e}")

def main():
    # load bank
    nf, nmeta, af, ameta = read_bank(BANK_DIR)
    A = af.shape[0] if af.ndim==2 else 0
    N = nf.shape[0] if nf.ndim==2 else 0
    print(f"[BANK] abnormal: {A} | normal: {N}")
    if A==0 or N==0:
        raise SystemExit("Bank rỗng. Chạy Step 1 trước.")

    # AR/size mask
    mask = make_mask_ar_size(nmeta, ameta, TAU_AR, TAU_SIZE)  # [A,N]

    # pair
    bgpair = {}
    sims_all = []
    cover = 0
    for i in range(A):
        abn_path = ameta[i]["im"]
        idx_mask = mask[i]
        if not idx_mask.any():
            bgpair[abn_path] = []
            continue
        idxs, sims = cosine_topk_query_to_keys(af[i], nf, max(TOPK_PER_ABN*5, TOPK_PER_ABN), mask_idx=idx_mask)
        kept = [(j, float(s)) for j,s in zip(idxs, sims) if s >= TAU_SIM_MIN]
        kept = kept[:TOPK_PER_ABN]
        if kept:
            cover += 1
            sims_all.append(kept[0][1])
            bgpair[abn_path] = [nmeta[j]["im"] for j,_ in kept]
        else:
            bgpair[abn_path] = []

    # save json + stats
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(bgpair, f, ensure_ascii=False, indent=2)
    hist_bins = [0,0.5,0.6,0.7,0.8,0.9,1.0]
    hist = np.histogram(np.array(sims_all) if sims_all else np.array([0.0]), bins=hist_bins)[0].tolist()
    stats = {
        "abnormal_total": A,
        "normal_total": N,
        "coverage": cover,
        "coverage_rate": round(cover/max(A,1), 4),
        "tau": {"ar": TAU_AR, "size": TAU_SIZE, "sim": TAU_SIM_MIN},
        "topk": TOPK_PER_ABN,
        "max_sim_hist": hist
    }
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {OUT_JSON} | {OUT_STATS}")
    print(f"[COVER] {cover}/{A} abn have ≥1 normal with sim≥{TAU_SIM_MIN}")

    # visuals for sanity
    print(f"[VIS] making panels…")
    ymodel = YOLO(MODEL_WEIGHTS); core = ymodel.model
    device = next(core.parameters()).device
    probe = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32, device=device)
    feat_mod = choose_after_stn_module(core, probe)  # dùng MIN_CHANNELS/MIN_DOWNS

    # pick first N_VIS abn that have pairs
    have = [k for k,v in bgpair.items() if len(v)>0]
    vis_pairs=[]
    Kn = normalize_rows(nf)
    for abn_path in have[:N_VIS]:
        i = next(idx for idx,m in enumerate(ameta) if m["im"]==abn_path)
        cand_paths = bgpair[abn_path]
        j_idx = [next(j for j,nm in enumerate(nmeta) if nm["im"]==p) for p in cand_paths]
        sims = (Kn[j_idx] @ (af[i]/(np.linalg.norm(af[i])+1e-9))).tolist()
        vis_pairs.append((abn_path, cand_paths, sims))

    safe_rmtree(VIS_DIR); os.makedirs(VIS_DIR, exist_ok=True)
    draw_panel(core, feat_mod, vis_pairs, VIS_DIR, DATA_YAML)
    print(f"[VIS] saved to {VIS_DIR}/")

if __name__ == "__main__":
    main()

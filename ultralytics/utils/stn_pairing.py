# ultralytics/utils/stn_pairing.py
import os, json, random

from collections import defaultdict
import torch
from torch.utils.data import BatchSampler, DataLoader
from pathlib import Path
from typing import Optional, Union
# ---------- helpers: đọc dataset.yaml + phân abnormal/normal ----------
def _load_lines_or_dirlist(data_yaml):
    """Trả về list đường dẫn ảnh train theo dataset.yaml (hỗ trợ thư mục hoặc txt)."""
    import yaml
    with open(data_yaml, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    trains = y["train"] if isinstance(y["train"], list) else [y["train"]]
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pgm", ".pnm")
    imgs = []
    for item in trains:
        p = str(item)
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    if fn.lower().endswith(exts):
                        imgs.append(os.path.join(root, fn))
        elif os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as ff:
                for line in ff:
                    q = line.strip()
                    if q:
                        imgs.append(q)
        else:
            pass
    return sorted(imgs), y

def _label_path_from_image(p):
    P = Path(p)
    for tok in ("images", "Images"):
        if tok in P.parts:
            idx = P.parts.index(tok)
            rel = Path(*P.parts[idx+1:]).with_suffix(".txt")
            return str(Path(*P.parts[:idx], "labels", rel))
    return str(P.parent.parent / "labels" / (P.stem + ".txt"))

def _is_abnormal(lbl_path):
    try:
        if not os.path.exists(lbl_path):
            return False
        with open(lbl_path, "r", encoding="utf-8") as f:
            s = f.read().strip()
        return len(s) > 0
    except Exception:
        return False

# ---------------------- index bank & map ----------------------
class PairIndexBank:
    """
    - Xây chỉ số: dataset index <-> path
    - is_abn mask (có nhãn -> abnormal)
    - Load bgpair_map.json: abnormal path -> list normal paths (ứng viên)
    - Cấp phát ứng viên normal theo round-robin cho mỗi abnormal
    """
    def __init__(self, data_yaml, bgpair_json, seed=0):
        self.seed = int(seed)
        self.imgs, _ = _load_lines_or_dirlist(data_yaml)
        self.path_to_idx = {p: i for i, p in enumerate(self.imgs)}
        self.is_abn = [_is_abnormal(_label_path_from_image(p)) for p in self.imgs]
        self.abn_idx = [i for i, v in enumerate(self.is_abn) if v]
        self.norm_idx = [i for i, v in enumerate(self.is_abn) if not v]

        with open(bgpair_json, "r", encoding="utf-8") as f:
            mp = json.load(f)

        self.map_idx = {}
        for apath, nlst in mp.items():
            if apath not in self.path_to_idx:
                continue
            ai = self.path_to_idx[apath]
            cand = [self.path_to_idx[p] for p in nlst if p in self.path_to_idx]
            if not cand:
                cand = list(self.norm_idx)  # fallback an toàn
            self.map_idx[ai] = cand

        self.cursor = defaultdict(int)  # round-robin mỗi abnormal

    def next_norm_for(self, abn_i):
        cands = self.map_idx.get(abn_i, self.norm_idx)
        if not cands:
            return random.choice(self.norm_idx)
        k = self.cursor[abn_i] % len(cands)
        self.cursor[abn_i] += 1
        return cands[k]

# --------------------- batch sampler ---------------------
class PairedBatchSampler(BatchSampler):
    """
    Mỗi batch: [abn0, norm0, abn1, norm1, ...]  (batch_size PHẢI chẵn).
    Hỗ trợ DDP qua rank/world_size (chia abnormal theo shard).
    """
    def __init__(self, bank: PairIndexBank, batch_size: int, drop_last=False,
                 seed=0, rank=0, world_size=1):
        assert batch_size % 2 == 0, "batch_size phải chẵn để ghép cặp."
        self.bank = bank
        self.bs = int(batch_size)
        self.drop_last = drop_last
        self._epoch = 0
        self._rng = random.Random(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))

    def __iter__(self):
        abn = self.bank.abn_idx[self.rank::self.world_size]  # shard theo rank
        self._rng.seed(self._epoch + self.bank.seed + self.rank * 1337)
        abn = abn.copy()
        self._rng.shuffle(abn)

        half = self.bs // 2
        for i in range(0, len(abn), half):
            chunk = abn[i:i+half]
            if len(chunk) < half and self.drop_last:
                break
            norms = [self.bank.next_norm_for(ai) for ai in chunk]
            z = []
            for a, n in zip(chunk, norms):
                z.extend([a, n])
            yield z

    def __len__(self):
        half = self.bs // 2
        n = len(self.bank.abn_idx[self.rank::self.world_size]) // half
        if not self.drop_last and (len(self.bank.abn_idx[self.rank::self.world_size]) % half):
            n += 1
        return n

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

# --------------------- collate wrapper (picklable) ---------------------
def _label_path_from_image(img_path: Union[str, Path]) -> Optional[str]:
    """
    Suy ra đường dẫn nhãn YOLO từ đường dẫn ảnh:
    .../images/.../xxx.jpg -> .../labels/.../xxx.txt
    Hỗ trợ cả 'images' và 'Images'. Nếu không tìm thấy, fallback về parent.parent/labels.
    """
    if not img_path:
        return None
    p = Path(img_path)

    # Tìm segment 'images' (không phân biệt hoa/thường)
    parts = list(p.parts)
    idx = -1
    for i, seg in enumerate(parts):
        if seg.lower() == "images":
            idx = i
            break

    if idx >= 0:
        # rel = phần sau 'images' (giữ nguyên cấu trúc con)
        rel = Path(*parts[idx + 1:]).with_suffix(".txt")
        cand = Path(*parts[:idx], "labels", rel)
        if cand.exists():
            return str(cand)

    # Fallback: .../parent/parent/labels/xxx.txt
    # (áp dụng cho layout datasets/<split>/{images,labels}/)
    try:
        alt = (p.parent.parent / "labels" / (p.stem + ".txt"))
        if alt.exists():
            return str(alt)
    except Exception:
        pass

    # Cuối cùng: thử ngay cạnh ảnh (ít gặp nhưng an toàn)
    sibling = (p.parent / (p.stem + ".txt"))
    return str(sibling) if sibling.exists() else None


def _is_abnormal(lbl_path: Optional[Union[str, Path]]) -> bool:
    """
    Ảnh 'abnormal' nếu file nhãn tồn tại và có >= 1 dòng (YOLO: class x y w h).
    """
    if not lbl_path:
        return False
    try:
        txt = Path(lbl_path)
        if not txt.exists() or not txt.is_file():
            return False
        # Nhãn rỗng (0 byte) hoặc toàn khoảng trắng => normal
        content = txt.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            return False
        # Có ít nhất 1 dòng non-empty
        for line in content.splitlines():
            if line.strip():
                return True
        return False
    except Exception:
        return False

class PairedCollate:
    """
    Wrap collate gốc và bổ sung:
      - 'pair_idx': [[0,1],[2,3],...]
      - 'abn_mask': Bool[B] (ảnh abnormal theo NHÃN GỐC TRÊN DISK, không phụ thuộc augment)
      - 'im_files': list[str] (đường dẫn ảnh)
    Dùng class top-level để picklable trên Windows.
    """
    def __init__(self, base_collate=None):
        # Nên truyền vào hàm collate top-level của Ultralytics (nếu có)
        self.base_collate = base_collate

    def __call__(self, batch):
        import torch

        # Gọi collate gốc (nếu có), sau đó mình nhồi thêm field vào dict kết quả.
        if self.base_collate is not None:
            base_out = self.base_collate(batch)
            out = base_out if isinstance(base_out, dict) else {"samples": base_out}
        else:
            out = {"samples": batch}

        # Tạo cặp (0,1), (2,3), ...
        B = len(batch)
        pair = []
        for i in range(0, B, 2):
            if i + 1 < B:
                pair.append([i, i + 1])
        out["pair_idx"] = torch.as_tensor(pair, dtype=torch.long)

        # Thu thập im_files & tính abn_mask từ NHÃN GỐC (ổn định, không bị mosaic/crop)
        im_files, abn_mask = [], []
        for s in batch:
            if isinstance(s, dict):
                # Ultralytics thường gán 'im_file'; fallback 'path' nếu cần
                p = s.get("im_file", s.get("path", None))
                im_files.append(p)
                if p:
                    lbl = _label_path_from_image(p)
                    abn_mask.append(_is_abnormal(lbl))
                else:
                    abn_mask.append(False)
            else:
                # Trường hợp hiếm: phần tử batch không phải dict (đảm bảo vẫn hoạt động)
                im_files.append(None)
                abn_mask.append(False)

        out["im_files"] = im_files
        out["abn_mask"] = torch.tensor(abn_mask, dtype=torch.bool)
        return out

def paired_collate_fn(base_collate=None):
    """Giữ API cũ nhưng trả về một callable picklable."""
    return PairedCollate(base_collate)

# === Mount paired DataLoader via callback (minimal wiring) ===
class UsePairedLoader:
    """
    Callback: thay trainer.train_loader bằng DataLoader dùng PairedBatchSampler.
    Đăng ký (cùng 1 instance):
        pl = UsePairedLoader(data_yaml=..., bgpair_map=..., batch_size=...)
        yolo.add_callback("on_train_start", pl.on_train_start)
        yolo.add_callback("on_fit_epoch_start", pl.on_fit_epoch_start)
    """
    def __init__(self, data_yaml=None, bgpair_map="bgpair_map.json",
                 batch_size=8, seed=0):
        self.data_yaml = data_yaml
        self.bgpair_map = bgpair_map
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._sampler = None
        self._trainer = None

    # đảm bảo train_loader có .reset() để YOLO gọi mà không văng khi close_mosaic
    def _attach_reset_noop(self, loader):
        if not hasattr(loader, "reset"):
            def _reset():
                ds = getattr(loader, "dataset", None)
                # YOLO đã tự gạt mosaic=False trước khi gọi reset; mình chỉ tắt chắc chắn.
                if hasattr(ds, "mosaic"):
                    try:
                        ds.mosaic = False
                    except Exception:
                        pass
                return None
            setattr(loader, "reset", _reset)

    def on_train_start(self, trainer):
        ds = trainer.train_loader.dataset
        base_collate = getattr(ds, "collate_fn", None)
        workers = getattr(trainer.args, "workers", 4)

        data_yaml = self.data_yaml or getattr(trainer.args, "data", None)
        assert data_yaml, "[UsePairedLoader] data_yaml is required."

        bank = PairIndexBank(data_yaml, self.bgpair_map, seed=self.seed)

        self._sampler = PairedBatchSampler(
            bank, batch_size=self.batch_size, drop_last=False,
            seed=self.seed,
            rank=getattr(trainer, "rank", 0),
            world_size=getattr(trainer, "world_size", 1)
        )

        loader = DataLoader(
            ds,
            batch_sampler=self._sampler,
            num_workers=workers,
            pin_memory=True,
            collate_fn=paired_collate_fn(base_collate)
        )

        trainer.train_loader = loader
        try:
            trainer.nb = len(loader)  # một số version dùng nb để tính bước/epoch
        except Exception:
            pass

        self._attach_reset_noop(trainer.train_loader)
        self._trainer = trainer  # giữ tham chiếu để dùng sau

    # set epoch cho sampler mỗi epoch (giống DistributedSampler)
    def on_fit_epoch_start(self, trainer):
        if self._sampler is not None:
            self._sampler.set_epoch(int(getattr(trainer, "epoch", 0)))

# ultralytics/utils/stn_pairing.py
import os, json, random

from collections import defaultdict
import torch
from torch.utils.data import BatchSampler, DataLoader
from pathlib import Path
from typing import Optional, Union
from ultralytics.utils import LOGGER
# Helper: đọc danh sách ảnh train từ dataset.yaml (hỗ trợ đường dẫn thư mục hoặc file txt)
def _load_lines_or_dirlist(data_yaml):
    import yaml
    with open(data_yaml, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    train_paths = y["train"] if isinstance(y["train"], list) else [y["train"]]
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pgm", ".pnm")
    imgs = []
    for item in train_paths:
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
    return sorted(imgs), y

# Helper: suy ra đường dẫn file label từ đường dẫn ảnh (theo cấu trúc YOLO: .../images/... -> .../labels/...)
def _label_path_from_image(p):
    from pathlib import Path
    P = Path(p)
    for tok in ("images", "Images"):
        if tok in P.parts:
            idx = P.parts.index(tok)
            rel = Path(*P.parts[idx+1:]).with_suffix(".txt")
            return str(Path(*P.parts[:idx], "labels", rel))
    return str(P.parent.parent / "labels" / (P.stem + ".txt"))

# Helper: kiểm tra ảnh có bất thường (abnormal) hay không dựa trên file label (có label => abnormal)
def _is_abnormal(lbl_path):
    try:
        if not os.path.exists(lbl_path):
            return False
        with open(lbl_path, "r", encoding="utf-8") as f:
            s = f.read().strip()
        return len(s) > 0
    except Exception:
        return False

# ---------------------- Index Bank & Mapping ----------------------
class PairIndexBank:
    """
    Tạo ngân hàng index cho dataset, phân nhóm ảnh abnormal (có vật thể) và normal (không vật thể).
    Sử dụng ds.im_files nếu có để đảm bảo thứ tự index khớp dataset thực tế.
    Yêu cầu: cung cấp data_yaml và file map JSON (bgpair_map) chứa mapping abnormal->danh sách normal liên quan.
    """
    def __init__(self, data_yaml=None, bgpair_json="bgpair_map.json", seed=0, im_files=None):
        self.seed = int(seed)
        # Chuẩn hoá đường dẫn để so khớp
        def _norm(p):
            return os.path.normpath(str(p)).replace("\\", "/").lower()
        # Lấy danh sách ảnh từ ds.im_files (nếu có) hoặc từ data_yaml
        if im_files is not None:
            self.imgs = [_norm(p) for p in im_files]
        else:
            assert data_yaml is not None, "PairIndexBank: cần truyền data_yaml hoặc im_files."
            _imgs, _ = _load_lines_or_dirlist(data_yaml)
            self.imgs = [_norm(p) for p in _imgs]
        # Map path->index
        self.path_to_idx = {p: i for i, p in enumerate(self.imgs)}
        # Xác định ảnh abnormal vs normal
        self.is_abn = []
        for p in self.imgs:
            lbl = _label_path_from_image(p)
            self.is_abn.append(_is_abnormal(lbl))
        self.abn_idx = [i for i, v in enumerate(self.is_abn) if v]       # list index abnormal
        self.norm_idx = [i for i, v in enumerate(self.is_abn) if not v]  # list index normal
        # Đọc file map abnormal->list normal (bgpair_map.json)
        with open(bgpair_json, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        # Chuẩn hoá key và value trong map theo _norm
        self.map_idx = {}
        have = 0
        for abn_path, norm_list in raw_map.items():
            ai = self.path_to_idx.get(_norm(abn_path), None)
            if ai is None:
                continue
            candidates = []
            for q in norm_list:
                j = self.path_to_idx.get(_norm(q), None)
                if j is not None:
                    candidates.append(j)
            if not candidates:
                # Nếu không có ảnh normal nào khớp, fallback dùng toàn bộ self.norm_idx
                candidates = list(self.norm_idx)
            self.map_idx[ai] = candidates
            have += 1
        self.cursor = defaultdict(int)  # con trỏ cho mỗi ảnh abnormal để xoay vòng chọn ảnh normal
        self.coverage = have / max(1, len(self.abn_idx))

    def next_norm_for(self, abn_i):
        """Lấy index ảnh normal tiếp theo ứng với ảnh abnormal index=abn_i (xoay vòng)."""
        cands = self.map_idx.get(abn_i, self.norm_idx)
        if not cands:  # nếu không có ứng viên, chọn ngẫu nhiên normal
            return random.choice(self.norm_idx)
        k = self.cursor[abn_i] % len(cands)
        self.cursor[abn_i] += 1
        return cands[k]

    def summary(self):
        return {
            "n_total": len(self.imgs),
            "n_abn": len(self.abn_idx),
            "n_norm": len(self.norm_idx),
            "coverage": self.coverage,  # tỷ lệ abnormal có ít nhất 1 mapping
        }

# --------------------- Paired Batch Sampler ---------------------
class PairedBatchSampler(BatchSampler):
    """
    BatchSampler tạo batch ghép cặp: [abn0, norm0, abn1, norm1, ...].
    Yêu cầu batch_size chẵn (mỗi cặp chiếm 2 slot).
    Hỗ trợ DistributedDataParallel (chia chỉ số abnormal theo rank).
    """
    def __init__(self, bank: PairIndexBank, batch_size: int, drop_last=False, seed=0, rank=0, world_size=1):
        assert batch_size % 2 == 0, "batch_size phải chẵn để ghép cặp."
        self.bank = bank
        self.bs = int(batch_size)
        self.drop_last = drop_last
        self._epoch = 0
        self._rng = random.Random(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))

    def __iter__(self):
        # Lấy danh sách index abnormal cho shard hiện tại (nếu DDP)
        abn_indices = self.bank.abn_idx[self.rank::self.world_size]
        # Trộn ngẫu nhiên danh sách abnormal cho epoch này
        self._rng.seed(self._epoch + self.bank.seed + self.rank * 1337)
        abn = abn_indices.copy()
        self._rng.shuffle(abn)
        half = self.bs // 2
        # Tạo batch cặp: mỗi batch half abnormal + half normal
        for i in range(0, len(abn), half):
            chunk = abn[i:i + half]
            if len(chunk) < half:
                if self.drop_last:
                    break  # bỏ batch cuối nếu thiếu
            # Lấy ảnh normal tương ứng cho mỗi abnormal trong chunk
            norms = [self.bank.next_norm_for(ai) for ai in chunk]
            # Tạo list index xen kẽ [a0, n0, a1, n1, ...]
            batch_indices = []
            for a, n in zip(chunk, norms):
                batch_indices.extend([a, n])
            yield batch_indices

    def __len__(self):
        half = self.bs // 2
        n_abn = len(self.bank.abn_idx[self.rank::self.world_size])
        total_batches = n_abn // half
        if not self.drop_last and n_abn % half != 0:
            total_batches += 1
        return total_batches

# --------------------- Paired Collate Function ---------------------
class PairedCollate:
    """
    Bọc collate gốc và bổ sung thông tin:
      - 'pair_idx': danh sách index các cặp [[0,1],[2,3],...]
      - 'abn_mask': tensor Bool [B] đánh dấu ảnh abnormal (theo nhãn gốc)
      - 'im_files': danh sách đường dẫn ảnh trong batch
    """
    def __init__(self, base_collate=None):
        self.base_collate = base_collate  # collate function ban đầu (nếu dataset có cung cấp)

    def __call__(self, batch):
        # Gọi collate gốc trước (nếu có)
        if self.base_collate is not None:
            base_out = self.base_collate(batch)
            out = base_out if isinstance(base_out, dict) else {"samples": base_out}
        else:
            out = {"samples": batch}
        # Tạo cặp chỉ số (0,1), (2,3), ... cho batch hiện tại
        B = len(batch)
        pair = [[i, i+1] for i in range(0, B, 2) if i+1 < B]
        out["pair_idx"] = torch.as_tensor(pair, dtype=torch.long)
        # Lấy danh sách file ảnh và mask abnormal cho từng ảnh
        im_files, abn_flags = [], []
        for sample in batch:
            # Giả sử mỗi sample là dict chứa 'im_file' hoặc 'path' và nhãn
            if isinstance(sample, dict):
                p = sample.get("im_file", sample.get("path", None))
                im_files.append(p)
                if p:
                    abn_flags.append(_is_abnormal(_label_path_from_image(p)))
                else:
                    abn_flags.append(False)
            else:
                # trường hợp sample không phải dict (ít xảy ra)
                im_files.append(None)
                abn_flags.append(False)
        out["im_files"] = im_files
        out["abn_mask"] = torch.tensor(abn_flags, dtype=torch.bool)
        return out

def paired_collate_fn(base_collate=None):
    """Trả về hàm collate để dùng trong DataLoader (khắc phục lambda pickling)."""
    return PairedCollate(base_collate)

# --------------------- UsePairedLoader Callback ---------------------
class UsePairedLoader:
    """
    Callback: Thay trainer.train_loader bằng DataLoader mới sử dụng PairedBatchSampler.
    Cách dùng:
        pl = UsePairedLoader(data_yaml=..., bgpair_map=..., batch_size=...)
        yolo.add_callback("on_train_start", pl.on_train_start)
        yolo.add_callback("on_fit_epoch_start", pl.on_fit_epoch_start)
    """
    def __init__(self, data_yaml=None, bgpair_map="bgpair_map.json", batch_size=8, seed=0):
        self.data_yaml = data_yaml
        self.bgpair_map = bgpair_map
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._sampler = None
        self._trainer = None

    def _attach_reset_noop(self, loader):
        # Đảm bảo DataLoader mới có hàm reset() (YOLO Trainer có thể gọi loader.reset())
        if not hasattr(loader, "reset"):
            def _reset():
                ds = getattr(loader, "dataset", None)
                # Tắt mosaic (nếu dataset có) trước khi reset
                if hasattr(ds, "mosaic"):
                    try:
                        ds.mosaic = False
                    except Exception:
                        pass
                return None
            setattr(loader, "reset", _reset)

    def on_train_start(self, trainer):
        # Khởi tạo DataLoader ghép cặp thay thế trainer.train_loader
        ds = trainer.train_loader.dataset
        base_collate = getattr(ds, "collate_fn", None)
        workers = getattr(trainer.args, "workers", 4)
        data_yaml = self.data_yaml or getattr(trainer.args, "data", None)
        assert data_yaml, "[UsePairedLoader] data_yaml is required."
        # Tạo ngân hàng index và sampler
        im_files = getattr(ds, "im_files", None)  # lấy danh sách ảnh từ dataset nếu có
        bank = PairIndexBank(data_yaml=data_yaml, bgpair_json=self.bgpair_map, seed=self.seed, im_files=im_files)
        info = bank.summary()
        LOGGER.info(f"[Pairing] total_images={info['n_total']} | abnormal={info['n_abn']} | normal={info['n_norm']} | coverage={info['coverage']*100:.1f}%")
        self._sampler = PairedBatchSampler(bank, batch_size=self.batch_size, drop_last=False,
                                           seed=self.seed, rank=getattr(trainer, "rank", 0),
                                           world_size=getattr(trainer, "world_size", 1))
        # Tạo DataLoader mới với sampler và collate đặc biệt
        loader = DataLoader(ds, batch_sampler=self._sampler, num_workers=workers,
                            pin_memory=True, collate_fn=paired_collate_fn(base_collate))
        trainer.train_loader = loader
        try:
            trainer.nb = len(loader)  # cập nhật số batch mỗi epoch
        except Exception:
            pass
        self._attach_reset_noop(trainer.train_loader)
        self._trainer = trainer

    def on_fit_epoch_start(self, trainer):
        # Mỗi đầu epoch (sau khi tăng trainer.epoch), cập nhật epoch cho sampler
        self._sampler._epoch = getattr(trainer, "epoch", 0)
# stn_yolo11.py
from copy import deepcopy

import torch
import torch.nn as nn
from ultralytics import YOLO                    # YOLO11m gốc
from ultralytics.nn.modules.block import SpatialTransformer
# stn_utils.py
import torch
from ultralytics.utils import LOGGER
from ultralytics.nn.modules.block import SpatialTransformer as STN

class StnFeatureTapper:
    last = None            # lưu tensor [B,C,H,W]
    last_stride = 1.0      # stride tương ứng (ảnh sau STN là stride=1)
    def __call__(self, module, inp, out):
        # out: tensor sau STN (cùng kích thước ảnh vào khối Conv đầu)
        StnFeatureTapper.last = out.detach()
        StnFeatureTapper.last_stride = 1.0

class ParallelFrozen(nn.Module):
    """
    Bản sao YOLO core, đóng băng toàn bộ tham số.
    Dùng ở train-time để làm 'thầy' song song (SPEE).
    """
    def __init__(self, yolocore):
        super().__init__()
        self.model = deepcopy(yolocore).eval()   # deep copy
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)                     # trả logits giống YOLO

class STN_YOLO(nn.Module):
    """
    Nhánh A = SpatialTransformer + YOLO11m (trainable)
    (nhánh B song song sẽ thêm ở bước sau)
    """
    def __init__(self, ckpt_path, in_ch=3, use_parallel=True):
        super().__init__()
        self.stn  = SpatialTransformer(in_ch)          # (1) STN
        self.core = YOLO(ckpt_path).model              # (2) YOLO trainable

        # ----------- nhánh song song SPEE -------------
        self.parallel = ParallelFrozen(self.core) if use_parallel else None   # (3)

    def forward(self, x):
        # ---------- STN ----------
        x_stn = self.stn(x)
        theta = getattr(self.stn, "theta", None)

        # ---------- YOLO trainable ----------
        preds_A = self.core(x_stn)

        # ---------- Khi TRAIN: trả 3 phần ----------
        if self.training and self.parallel is not None:
            preds_B = self.parallel(x)  # no-grad
            return preds_A, preds_B, theta  # <-- 3 outputs

        # ---------- Khi EVAL: trả 1 phần ----------
        return preds_A




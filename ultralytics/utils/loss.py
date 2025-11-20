# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
from __future__ import annotations
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
from .metrics import bbox_iou, probiou
from .tal import bbox2dist
from torchvision.ops import roi_align
from types import SimpleNamespace
import os, json
import torch
import torch.nn.functional as F
from ultralytics.utils import LOGGER
import torch.nn as nn
from ultralytics.utils.tal import TaskAlignedAssigner
from typing import Optional

class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.
    https://arxiv.org/abs/2008.13367.
    """
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score, gt_score, label):
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss

class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn(), e.g. FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).
    """
    def __init__(self, gamma=1.5, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred, label):
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()

class DFLoss(nn.Module):
    """Distribution Focal Loss (DFL)."""
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        loss = (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        )
        return loss.mean(-1, keepdim=True)

class BboxLoss(nn.Module):
    """Compute IoU and DFL losses for bounding boxes."""
    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)
        return loss_iou, loss_dfl

class RotatedBboxLoss(BboxLoss):
    """Compute IoU and DFL losses for rotated bounding boxes."""
    def __init__(self, reg_max):
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)
        return loss_iou, loss_dfl

# ----------------------------------------------------------------------(ntnhan.0705)
# ======================= (Helpers — logging) =======================
# BỎ cơ chế 'log một lần' để dễ debug:
# - Mọi lần gọi _warn_once / _info_once đều in log đầy đủ.
# - Giữ lại tên hàm để không phải sửa các chỗ gọi bên dưới.

def _warn_once(key: str, msg: str):
    """Wrapper quanh LOGGER.warning, không còn giới hạn 1 lần."""
    LOGGER.warning(msg)

def _info_once(key: str, msg: str):
    """Wrapper quanh LOGGER.info, không còn giới hạn 1 lần."""
    LOGGER.info(msg)

def _safe_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x is None:
        return x
    denom = x.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    return x / denom

# ---------------- SupCon Projection ----------------
class SupConProjection(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, bn=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = None
        if bn == 1 or str(bn).lower() == 'bn':
            self.bn1 = nn.BatchNorm1d(hidden)
        elif bn == 2 or str(bn).lower() == 'ln':
            self.bn1 = nn.LayerNorm(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        return self.fc2(x)

_SUPCON_KEYS = (
    "supcon_on", "supcon_feat", "supcon_warp_gt", "supcon_out",
    "supcon_min_box", "supcon_max_per_class", "supcon_gain", "supcon_loss_weight",
    "supcon_temp", "supcon_warmup", "supcon_log", "supcon_use_mem", "supcon_queue",
    "supcon_neg_cap", "supcon_schedule", "supcon_proj_dim", "supcon_proj_hidden", "supcon_proj_bn",
    "supcon_neg_per_pos", "supcon_min_neg_w", "supcon_log_n",
    "supcon_neg_iou_ignore", "supcon_neg_sameimg_only",
    "stn_reg", "supcon_proj_lr","stn_grad_mult"
)

def _norm_pyramids_for_loss(preds, expect_n: int):
    """
    Thu gom tất cả map 4D [B,C,H,W] từ preds (kể cả cấu trúc lồng nhau của YOLO),
    đảm bảo cùng batch B. Nếu lệch B, log chi tiết và loại bỏ map lẻ.
    """
    out = []
    stack = [preds]
    while stack:
        x = stack.pop()
        if torch.is_tensor(x) and x.dim() == 4:
            out.append(x)
        elif isinstance(x, (list, tuple)):
            stack.extend(list(x))
        elif isinstance(x, dict):
            stack.extend(list(x.values()))
    if not out:
        return []

    B0 = out[0].shape[0]
    kept = [t for t in out if t.shape[0] == B0]
    dropped = [tuple(t.shape) for t in out if t.shape[0] != B0]
    if dropped:
        _warn_once("loss.pyramid.Bmismatch",
                   f"[Loss] Drop feature maps due to batch mismatch: kept_B={B0}, dropped_shapes={dropped}")
    return kept

# =====================[ v8DetectionLoss (w/ SupCon) ]=====================
class v8DetectionLoss:
    """YOLOv8 detection loss + SupCon (ROIAlign sau STN) với logging concat shape + guard."""
    def __init__(self, model, tal_topk=10):
        self.model = model
        self.device = next(model.parameters()).device

        raw_hyp = getattr(model, "args", None)
        self.hyp = SimpleNamespace(**(raw_hyp or {})) if raw_hyp is None or isinstance(raw_hyp, dict) else raw_hyp

        # SupCon cfg
        self._normalize_and_mirror_supcon(from_env=True)
        LOGGER.info(f"[LOSS INIT] use_supcon={bool(getattr(self.hyp,'supcon_on',0))}, "
                    f"supcon_weight={getattr(self.hyp,'supcon_loss_weight', None)}")

        head = model.model[-1]
        self.nc = head.nc
        self.reg_max = head.reg_max
        self.no = self.nc + self.reg_max * 4
        self.stride = head.stride
        self.use_dfl = self.reg_max > 1

        # Loss parts
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self._tal_topk = int(tal_topk)
        self.TaskAlignedAssigner = TaskAlignedAssigner
        self.BboxLoss = BboxLoss
        self.assigner = self.TaskAlignedAssigner(topk=self._tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = self.BboxLoss(self.reg_max).to(self.device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float32, device=self.device)

        # SupCon memory (CPU fp16)
        self._mq_feats: Optional[torch.Tensor] = None
        self._mq_labels: Optional[torch.Tensor] = None
        self._mq_ptr = 0
        self._mq_size = int(getattr(self.hyp, "supcon_queue", 4096))
        self._mq_ready = False

        # STN hooks
        self.theta_for_loss = None
        self._stn_module = None
        self._aft_min_channels = 128
        self._aft_hooks = []
        self._aft_last = None

        # Stat/log
        self._ep_pos_pairs = 0
        self._ep_neg_pairs = 0
        self._ep_drop_pairs = 0
        self._log_epoch = -1
        self._mem_log_cnt = 0
        self._roi_log_cnt = 0
        self._max_logs = int(getattr(self.hyp, "supcon_log_n", 6))
        self._supcon_stat = {}
        self._supcon_val = None
        self._supcon_prob = None
        self._printed_hyp = False

        # tìm STN và hook block sau STN
        stn_seen = False
        for m in model.modules():
            cname = m.__class__.__name__
            if cname in ("SpatialTransformer", "STN", "SpatialTransformerV2"):
                self._stn_module = m
                try:
                    m.record_theta = self.set_theta
                except Exception:
                    pass
                stn_seen = True
                continue
            if stn_seen and hasattr(m, "forward"):
                try:
                    h = m.register_forward_hook(self._after_hook)
                    self._aft_hooks.append(h)
                except Exception:
                    pass

    # ---- expose training flag cho callback ----
    @property
    def training(self) -> bool:
        try:
            return bool(self.model.training)
        except Exception:
            return True

    # ---- SupCon cfg merge/mirror ----
    def _merge_from_model_args(self):
        ma = getattr(self.model, "args", None)
        if isinstance(ma, SimpleNamespace):
            for k in _SUPCON_KEYS:
                v = getattr(ma, k, None)
                if v is not None:
                    setattr(self.hyp, k, v)
        elif isinstance(ma, dict):
            for k in _SUPCON_KEYS:
                if k in ma and ma[k] is not None:
                    setattr(self.hyp, k, ma[k])

    def _merge_from_env(self):
        js = os.environ.get("SUPCON_INJECT_JSON", "")
        if not js:
            return
        try:
            cfg = json.loads(js)
            if isinstance(cfg, dict):
                for k in _SUPCON_KEYS:
                    v = cfg.get(k, None)
                    if v is not None:
                        setattr(self.hyp, k, v)
        except Exception:
            pass

    def _finalize_supcon(self):
        # chốt w theo gain nhưng CHỈ trong self.hyp, không ghi ngược vào model.args
        gain = float(getattr(self.hyp, "supcon_gain", 2.5))
        w = getattr(self.hyp, "supcon_loss_weight", None)
        try:
            w = float(w if w is not None else gain)
        except (TypeError, ValueError):
            w = gain
        setattr(self.hyp, "supcon_loss_weight", w)
        if not getattr(self.hyp, "supcon_feat", None):
            setattr(self.hyp, "supcon_feat", "stn")

    def _normalize_and_mirror_supcon(self, from_env: bool = False):
        self._merge_from_model_args()
        if from_env:
            self._merge_from_env()
        self._finalize_supcon()

    # ---- Assigner guard (tự tạo nếu bị mất) ----
    def _ensure_assigner(self):
        if not hasattr(self, "assigner") or self.assigner is None:
            _warn_once("loss.assigner.missing", "[Loss] assigner missing -> recreate TaskAlignedAssigner on-the-fly")
            self.assigner = self.TaskAlignedAssigner(topk=self._tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)

    # ---- Memory queue ----
    @torch.no_grad()
    def _mq_ensure(self, cdim: int):
        qsize = int(getattr(self.hyp, "supcon_queue", self._mq_size))
        if (self._mq_feats is None) or (self._mq_feats.size(0) != qsize) or (self._mq_feats.size(1) != cdim):
            self._mq_size = qsize
            self._mq_feats = torch.zeros((qsize, cdim), dtype=torch.float16, device="cpu")
            self._mq_labels = torch.full((qsize,), -1, dtype=torch.long, device="cpu")
            self._mq_ptr = 0
            self._mq_ready = False

    @torch.no_grad()
    def _mq_enqueue(self, feats: torch.Tensor, labels: torch.Tensor):
        if feats is None or feats.numel() == 0 or labels is None or labels.numel() == 0:
            return
        labels = labels.view(-1)
        pos = labels.ge(0)
        if not pos.any():
            return
        feats = feats[pos].contiguous()
        labels = labels[pos].contiguous()
        N, C = feats.shape
        self._mq_ensure(C)

        ptr = int(self._mq_ptr)
        qsize = int(self._mq_size)
        end = ptr + N

        feats_cpu = feats.to("cpu", non_blocking=True).half()
        labels_cpu = labels.to("cpu", non_blocking=True)

        if end <= qsize:
            self._mq_feats[ptr:end] = feats_cpu
            self._mq_labels[ptr:end] = labels_cpu
        else:
            first = qsize - ptr
            if first > 0:
                self._mq_feats[ptr:] = feats_cpu[:first]
                self._mq_labels[ptr:] = labels_cpu[:first]
            rest = N - first
            if rest > 0:
                self._mq_feats[:rest] = feats_cpu[first:first + rest]
                self._mq_labels[:rest] = labels_cpu[first:first + rest]

        self._mq_ptr = (ptr + N) % qsize
        if not self._mq_ready and self._mq_ptr == 0:
            self._mq_ready = True

    # ---- SupCon Loss (w/ NEG overlap filtering) ----
    def _supcon_loss_memory(self, z: torch.Tensor, y: torch.Tensor, T: float,
                            *, anc_boxes=None, anc_bidx=None, key_boxes=None, key_bidx=None):
        if z is None or z.numel() == 0:
            return None

        device = z.device
        y = y.view(-1).long()

        a_mask = y.ge(0)
        if a_mask.sum() == 0:
            return None
        z_a = _safe_normalize(z[a_mask])
        y_a = y[a_mask]
        z_bg = z[~a_mask] if (~a_mask).any() else None

        keys_list, klabels_list = [z_a.detach()], [y_a]
        if z_bg is not None and z_bg.numel() > 0:
            keys_list.append(z_bg.detach())
            klabels_list.append(torch.full((z_bg.size(0),), -1, dtype=torch.long, device=y.device))

        use_mem = int(getattr(self.hyp, "supcon_use_mem", 1))
        # tắt queue lúc validate cho an toàn
        if self.training is False:
            use_mem = 0

        if use_mem:
            self._mq_ensure(int(z_a.size(1)))
            if (self._mq_feats is not None) and (self._mq_labels is not None):
                if self._mq_feats.size(1) != z_a.size(1):
                    _warn_once("supcon.mem.cdim_mismatch",
                               f"[SupCon/Mem] C mismatch mem={self._mq_feats.size(1)} vs cur={z_a.size(1)} -> reset")
                    self._mq_feats = None
                    self._mq_labels = None
                    self._mq_ptr = 0
                    self._mq_ready = False
                else:
                    valid_mem = self._mq_labels.ge(0)
                    if valid_mem.any():
                        mq_feats = self._mq_feats[valid_mem].to(device, non_blocking=True)
                        mq_labs = self._mq_labels[valid_mem].to(device, non_blocking=True)
                        keys_list.append(mq_feats)
                        klabels_list.append(mq_labs)

        # ---- safe cat w/ shape logs ----
        try:
            keys = torch.cat(keys_list, dim=0)  # [K, C]
            klabels = torch.cat(klabels_list, dim=0)
        except Exception as e:
            shapes = [tuple(k.shape) for k in keys_list]
            _warn_once("supcon.mem.cat_fail",
                       f"[SupCon/Mem] torch.cat(keys_list) failed shapes={shapes} err={type(e).__name__}: {e}")
            # thử lọc theo C phổ biến nhất
            from collections import Counter
            cs = [k.shape[1] for k in keys_list]
            common_c = Counter(cs).most_common(1)[0][0]
            filtered = [k for k in keys_list if k.shape[1] == common_c]
            if len(filtered) >= 1:
                keys = torch.cat(filtered, dim=0)
                klabels = torch.cat([klabels_list[i] for i, k in enumerate(keys_list) if k.shape[1] == common_c], dim=0)
                _warn_once("supcon.mem.cat_recover",
                           f"[SupCon/Mem] RECOVER keep_C={common_c} new_shapes={[tuple(t.shape) for t in filtered]}")
            else:
                return None

        K_base = z_a.size(0) + (z_bg.size(0) if z_bg is not None else 0)

        finite = torch.isfinite(keys).all(dim=1)
        if not finite.any():
            return None
        keys = keys[finite]
        klabels = klabels[finite]
        base_kept_mask = finite[:K_base] if K_base > 0 else torch.zeros(0, dtype=torch.bool, device=device)
        K_base_kept = int(base_kept_mask.sum().item())

        # ---- NEG IoU ignore (optional) ----
        iou_thr = float(getattr(self.hyp, "supcon_neg_iou_ignore", 0.30) or 0.0)
        ignore_small = None
        dropped = 0
        Ba = int(z_a.size(0))
        K_bg_kept = 0  # placeholder if bạn muốn thêm BG keys riêng

        if iou_thr <= 0.0:
            _info_once("supcon.negiou.off", "[SupCon/NegIoU] disabled")
        elif (anc_boxes is None) or (key_boxes is None):
            _warn_once("supcon.negiou.missing_boxes",
                       f"[SupCon/NegIoU:SKIP] missing boxes anc={anc_boxes is not None} key={key_boxes is not None}")
        elif K_base_kept == 0:
            _info_once("supcon.negiou.kbase0", "[SupCon/NegIoU:SKIP] no base keys kept")
        else:
            try:
                if (anc_boxes.dim() != 2) or (anc_boxes.size(1) != 4):
                    _warn_once("supcon.negiou.bad.anc_shape",
                               f"[SupCon/NegIoU:SKIP] anc_boxes shape invalid: {tuple(anc_boxes.shape)}")
                elif (key_boxes.dim() != 2) or (key_boxes.size(1) != 4):
                    _warn_once("supcon.negiou.bad.key_shape",
                               f"[SupCon/NegIoU:SKIP] key_boxes shape invalid: {tuple(key_boxes.shape)}")
                elif anc_boxes.size(0) != Ba:
                    _warn_once("supcon.negiou.ba_mismatch",
                               f"[SupCon/NegIoU:SKIP] Ba mismatch: Ba={Ba} anc_rows={anc_boxes.size(0)}")
                elif key_boxes.size(0) < (K_base if isinstance(base_kept_mask, torch.Tensor) else K_base):
                    _warn_once("supcon.negiou.klt_kbase",
                               f"[SupCon/NegIoU:SKIP] key_rows < K_base ({key_boxes.size(0)} < {K_base})")
                else:
                    if not isinstance(base_kept_mask, torch.Tensor):
                        _info_once("supcon.negiou.no_base_mask",
                                   "[SupCon/NegIoU] base_kept_mask missing — using all base keys.")
                        kb = key_boxes[:K_base]
                        K_base_kept_eff = K_base
                    else:
                        kb = key_boxes[:K_base][base_kept_mask]
                        K_base_kept_eff = K_base_kept

                    if kb.numel() > 0:
                        tl = torch.maximum(anc_boxes[:, None, :2], kb[None, :, :2])
                        br = torch.minimum(anc_boxes[:, None, 2:], kb[None, :, 2:])
                        wh = (br - tl).clamp(min=0)
                        inter = wh[..., 0] * wh[..., 1]
                        area_a = (anc_boxes[:, 2] - anc_boxes[:, 0]).clamp(min=0) * \
                                 (anc_boxes[:, 3] - anc_boxes[:, 1]).clamp(min=0)
                        area_b = (kb[:, 2] - kb[:, 0]).clamp(min=0) * \
                                 (kb[:, 3] - kb[:, 1]).clamp(min=0)
                        union = area_a[:, None] + area_b[None, :] - inter
                        iou_mat = inter / (union + 1e-9)
                        ignore_small = (iou_mat >= iou_thr)  # [Ba, K_base_kept_eff]

                        if K_bg_kept > 0:
                            pad_zeros = torch.zeros((Ba, K_bg_kept), dtype=torch.bool, device=z_a.device)
                            if ignore_small.size(0) == Ba:
                                ignore_small = torch.cat([ignore_small, pad_zeros], dim=1)
                        LOGGER.info(f"[SupCon/NegIoU] ok Ba={Ba} K_base_kept={K_base_kept_eff} "
                                    f"mask={None if ignore_small is None else tuple(ignore_small.shape)}")
            except Exception as ex:
                _warn_once("supcon.negiou.exception",
                           f"[SupCon/NegIoU:SKIP] exception -> disable for this batch: {type(ex).__name__}: {ex}")
                ignore_small = None

        keys = _safe_normalize(keys)
        logits = (z_a.float() @ keys.float().t()) / float(T)
        Ba, K = logits.shape

        pos_mask = y_a.view(-1, 1).eq(klabels.view(1, -1)) & klabels.view(1, -1).ge(0)
        logits_mask = torch.ones_like(logits, dtype=torch.bool)
        if K >= Ba:
            idx = torch.arange(Ba, device=device)
            logits_mask[idx, idx] = False
        neg_mask = logits_mask & (~pos_mask)

        if (ignore_small is not None) and (K_base_kept > 0):
            pad_right = K - K_base_kept
            if pad_right > 0:
                pad_zeros = torch.zeros((Ba, pad_right), dtype=torch.bool, device=device)
                ignore_mask_full = torch.cat([ignore_small, pad_zeros], dim=1)
            else:
                ignore_mask_full = ignore_small[:, :K]
            dropped = int((neg_mask & ignore_mask_full).sum().item())
            neg_mask = neg_mask & (~ignore_mask_full)

        logits = logits - logits.max(dim=1, keepdim=True).values
        same_f = pos_mask.to(logits.dtype)
        mask_f = logits_mask.to(logits.dtype)
        p = same_f.sum(dim=1).clamp(min=1.0)
        n = (mask_f - same_f).sum(dim=1).clamp(min=1.0)

        neg_per_pos = float(getattr(self.hyp, "supcon_neg_per_pos", 2.0))
        min_neg_w = float(getattr(self.hyp, "supcon_min_neg_w", 1e-3))
        lambda_neg = torch.minimum(torch.ones_like(p, dtype=logits.dtype),
                                   (neg_per_pos * p) / n).clamp(min=min_neg_w)
        log_w_neg = torch.log(lambda_neg).unsqueeze(1)

        logits_w = torch.where(neg_mask, logits + log_w_neg, logits)
        logits_w = torch.where(logits_mask, logits_w, torch.full_like(logits_w, float('-inf')))

        pos_only = torch.where(pos_mask, logits_w, torch.full_like(logits_w, float('-inf')))
        valid_anchor = pos_mask.any(dim=1)
        if not valid_anchor.any():
            return None

        num = torch.logsumexp(pos_only[valid_anchor], dim=1)
        den = torch.logsumexp(logits_w[valid_anchor], dim=1)
        loss = -(num - den).mean()

        with torch.no_grad():
            self._supcon_prob = torch.exp(num - den).mean().clamp(0.0, 1.0)

        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        # accumulate epoch stats
        with torch.no_grad():
            pos_pairs = int(pos_mask.sum().item())
            neg_pairs = int(neg_mask.sum().item())
            self._ep_pos_pairs += pos_pairs
            self._ep_neg_pairs += neg_pairs
            self._ep_drop_pairs += int(dropped)

        # enqueue positives
        with torch.no_grad():
            self._mq_enqueue(_safe_normalize(z_a.detach().float()), y_a.detach())
        if int(getattr(self.hyp, "supcon_log", 1)) == 1 and self._mem_log_cnt < self._max_logs:
            try:
                keys_pos = int((klabels >= 0).sum().item())
                keys_bg = int((klabels == -1).sum().item())
                LOGGER.info(
                    f"[SupConStat/mem] pos_pairs={pos_pairs} neg_pairs={neg_pairs} dropped={dropped} "
                    f"anchors={int(valid_anchor.sum().item())} keys_pos={keys_pos} keys_bg={keys_bg} "
                    f"val={float(self._supcon_prob) if self._supcon_prob is not None else None}"
                )
                self._mem_log_cnt += 1
            except Exception:
                pass
        return loss

    def _supcon_loss(self, z: torch.Tensor, y: torch.Tensor, T: float):
        if z is None or z.numel() == 0:
            return None
        z = _safe_normalize(z)
        y = y.view(-1).long()
        B = z.size(0)
        logits = (z @ z.t()) / float(T)
        idx = torch.arange(B, device=z.device)
        logits[idx, idx] = -float("inf")
        same = y.view(-1, 1).eq(y.view(1, -1))
        pos_mask = same & y.view(-1, 1).ge(0) & y.view(1, -1).ge(0)
        pos_cnt = pos_mask.sum(1)
        valid_anchor = (pos_cnt > 0) & y.ge(0)
        if not valid_anchor.any():
            return None
        log_den = torch.logsumexp(logits, dim=1)
        pos_logits = torch.where(pos_mask, logits, logits.new_full(logits.shape, -float("inf")))
        log_num = torch.logsumexp(pos_logits, dim=1)
        return (-(log_num - log_den))[valid_anchor].mean()

    # ---- yolo plumbing ----
    def set_theta(self, theta: torch.Tensor):
        self.theta_for_loss = theta.detach() if theta is not None else None

    def _after_hook(self, _module, _inp, out):
        if torch.is_tensor(out) and out.dim() == 4 and out.shape[1] >= self._aft_min_channels:
            self._aft_last = out

    def preprocess(self, targets, batch_size, scale_tensor):
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
        scale_tensor_full = scale_tensor[[1, 0, 1, 0]]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5] * scale_tensor_full)
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            pred_dist = torch.nan_to_num(pred_dist, nan=0.0, posinf=0.0, neginf=0.0)
            b, a, c = pred_dist.shape
            try:
                x = pred_dist.contiguous().view(b, a, 4, c // 4)
                x32 = x.float().softmax(3)
                pred_dist = x32.matmul(self.proj.float())
            except Exception as e:
                LOGGER.warning(f"[DFL decode] fallback due to {e}")
                x = pred_dist.reshape(b, a, 4, max(1, c // 4)).float()
                pred_dist = x.softmax(3).matmul(self.proj.float())
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        # epoch rollover logging
        cur_epoch = getattr(self, "epoch", -1)
        if cur_epoch != getattr(self, "_log_epoch", -2):
            if getattr(self, "_log_epoch", -1) >= 0:
                try:
                    LOGGER.info(f"[SupCon/epoch] e={self._log_epoch} "
                                f"pos_pairs={self._ep_pos_pairs} neg_pairs={self._ep_neg_pairs} dropped={self._ep_drop_pairs}")
                except Exception:
                    pass
            self._ep_pos_pairs = self._ep_neg_pairs = self._ep_drop_pairs = 0
            self._log_epoch = cur_epoch
            self._mem_log_cnt = self._roi_log_cnt = 0

        self._normalize_and_mirror_supcon(from_env=True)
        if not self._printed_hyp:
            LOGGER.info(f"[LOSS/HYP effective] on={getattr(self.hyp, 'supcon_on', 0)}, "
                        f"feat={getattr(self.hyp, 'supcon_feat', None)}, "
                        f"w={getattr(self.hyp, 'supcon_loss_weight', None)}, "
                        f"gain={getattr(self.hyp, 'supcon_gain', None)}")
            self._printed_hyp = True

        feats = _norm_pyramids_for_loss(preds, expect_n=len(self.stride))
        if not feats:
            zero_items = torch.zeros(4, device=self.device)
            return zero_items.sum(), zero_items

        B = feats[0].shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=torch.float32) * float(self.stride[0])

        # ---- build pred views with shape logs
        pred_views = []
        shape_notes = []
        for li, xi in enumerate(feats):
            Bi, Ci, Hi, Wi = xi.shape
            no = int(self.no)
            if Ci == no:
                xv = xi.view(Bi, no, -1)
            elif Ci % no == 0:
                g = Ci // no
                xg = xi.view(Bi, g, no, Hi, Wi)
                xv = xg.mean(1).contiguous().view(Bi, no, -1)
            else:
                B_, C_, H_, W_ = xi.shape
                if C_ < no:
                    shape_notes.append((li, tuple(xi.shape), "skip(C<no)"))
                    continue
                xv = xi.view(B_, no, -1)
            pred_views.append(xv)
            shape_notes.append((li, tuple(xi.shape), "ok->"+str(tuple(xv.shape))))

        if self._roi_log_cnt < 6:
            LOGGER.info(
                "[PredCat/shapes] " +
                ", ".join([f"L{li}:{s}->{note}" for (li, s, note) in shape_notes])
            )
            self._roi_log_cnt += 1

        if not pred_views:
            zero_items = torch.zeros(4, device=self.device)
            return zero_items.sum(), zero_items

        try:
            pred_cat = torch.cat(pred_views, 2)  # [B, no, sum(HW)]
        except Exception as e:
            shapes = [tuple(v.shape) for v in pred_views]
            _warn_once("loss.predcat.fail",
                       f"[PredCat] cat failed shapes={shapes} err={type(e).__name__}: {e}")
            zero_items = torch.zeros(4, device=self.device)
            return zero_items.sum(), zero_items

        pred_distri, pred_scores = pred_cat.split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        gt_all = self.preprocess(targets, B, imgsz)
        gt_labels = gt_all[:, :, :1]
        gt_bboxes = gt_all[:, :, 1:5]
        mask_gt = gt_labels.ge(0)

        # ---- SupCon ----
        self._supcon_val = None
        self._supcon_prob = None
        do_supcon = bool(self.model.training) and bool(int(getattr(self.hyp, "supcon_on", 0))) and torch.is_grad_enabled()
        if do_supcon:
            try:
                if str(getattr(self.hyp, "supcon_feat", "stn")).lower() == "stn" and (getattr(self, "_aft_last", None) is not None):
                    feat_map, src = self._aft_last, "after_stn"
                else:
                    feat_map, src = feats[0], "p3"

                valid = mask_gt.squeeze(-1)
                if valid.any():
                    b_idx, m_idx = valid.nonzero(as_tuple=False).T
                    boxes_bm = gt_bboxes.detach().clone()
                    if int(getattr(self.hyp, "supcon_warp_gt", 0)) and (getattr(self, "theta_for_loss", None) is not None) and (src == "after_stn"):
                        boxes_bm = self.warp_bbox(boxes_bm, self.theta_for_loss, imgsz)

                    boxes = boxes_bm[b_idx, m_idx]          # [N,4] xyxy
                    labels = gt_labels[b_idx, m_idx, 0].long()

                    wh = boxes[:, 2:] - boxes[:, :2]
                    keep = (wh[:, 0] >= int(getattr(self.hyp, "supcon_min_box", 1))) & (wh[:, 1] >= int(getattr(self.hyp, "supcon_min_box", 1)))
                    if keep.any():
                        boxes, labels, b_keep = boxes[keep], labels[keep], b_idx[keep]

                        max_pc = int(getattr(self.hyp, "supcon_max_per_class", 0))
                        if max_pc > 0:
                            uniq = labels.unique(sorted=True)
                            sel_idx = []
                            for c in uniq.tolist():
                                idxc = torch.nonzero(labels == c, as_tuple=False).view(-1)
                                sel_idx.append(idxc[:max_pc] if idxc.numel() > max_pc else idxc)
                            idxs = torch.cat(sel_idx, dim=0) if len(sel_idx) > 1 else (sel_idx[0] if sel_idx else torch.empty(0, dtype=torch.long, device=labels.device))
                            if idxs.numel():
                                boxes, labels, b_keep = boxes[idxs], labels[idxs], b_keep[idxs]

                        if boxes.numel() > 0:
                            H, W = int(imgsz[0].item()), int(imgsz[1].item())
                            _, Cf, Hf, Wf = feat_map.shape

                            x1, y1, x2, y2 = boxes.unbind(1)
                            x1 = x1.clamp(0, W - 1); y1 = y1.clamp(0, H - 1)
                            x2 = torch.maximum(x2.clamp(0, W - 1), x1 + 1)
                            y2 = torch.maximum(y2.clamp(0, H - 1), y1 + 1)
                            boxes = torch.stack([x1, y1, x2, y2], 1)

                            b_keep = b_keep.clamp_min(0).clamp_max(B - 1)
                            sx, sy = Wf / float(W), Hf / float(H)
                            fx1, fy1 = boxes[:, 0] * sx, boxes[:, 1] * sy
                            fx2, fy2 = boxes[:, 2] * sx, boxes[:, 3] * sy

                            # ROIAlign: log & guard stack
                            if self._roi_log_cnt < 12:
                                LOGGER.info(
                                    f"[ROI/in] src={src} feat={tuple(feat_map.shape)} imgsz={tuple(imgsz.tolist())} "
                                    f"N={int(boxes.size(0))} Bkeep={tuple(b_keep.shape)} "
                                    f"x1={tuple(x1.shape)} y1={tuple(y1.shape)} x2={tuple(x2.shape)} y2={tuple(y2.shape)} "
                                    f"sx={float(sx):.4f} sy={float(sy):.4f}"
                                )
                            try:
                                rois = torch.stack([b_keep.float(), fx1.float(), fy1.float(), fx2.float(), fy2.float()], 1)\
                                       .to(feat_map.device, torch.float32)
                            except Exception as e:
                                LOGGER.warning(
                                    "[ROI/stack:FAIL] cannot stack b,fx1,fy1,fx2,fy2 | "
                                    f"b={tuple(b_keep.shape)} fx1={tuple(fx1.shape)} fy1={tuple(fy1.shape)} "
                                    f"fx2={tuple(fx2.shape)} fy2={tuple(fy2.shape)} err={type(e).__name__}: {e}"
                                )
                                raise

                            pooled = roi_align(input=feat_map, boxes=rois,
                                               output_size=(int(getattr(self.hyp, "supcon_out", 7)),
                                                            int(getattr(self.hyp, "supcon_out", 7)) ),
                                               spatial_scale=1.0, sampling_ratio=0, aligned=True)
                            z = pooled.mean(dim=(2, 3))  # [N, C]
                            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

                            proj_dim = int(getattr(self.hyp, "supcon_proj_dim", 0))
                            if proj_dim > 0:
                                if getattr(self, "_proj_head", None) is None:
                                    in_dim = int(z.shape[1])
                                    hid = int(getattr(self.hyp, "supcon_proj_hidden", 0)) or max(128, in_dim)
                                    self._proj_head = SupConProjection(in_dim=in_dim, hidden=hid,
                                                                       out_dim=proj_dim,
                                                                       bn=int(getattr(self.hyp, "supcon_proj_bn", 1)))\
                                                      .to(device=z.device, dtype=torch.float32)
                                    setattr(self.model, "supcon_proj", self._proj_head)
                                    LOGGER.info(f"[SupConProj] created (in_dim={in_dim}, out_dim={proj_dim}, hidden={hid})")
                                self._proj_head = self._proj_head.to(device=z.device, dtype=torch.float32)
                                z = self._proj_head(z.float())

                            z = z / (z.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6))

                            pos_mask = labels.ge(0)
                            num_pos = int(pos_mask.sum().item())
                            if num_pos >= 1:
                                z_pos = z[pos_mask]
                                y_pos = labels[pos_mask]
                                loss_mem = self._supcon_loss_memory(
                                    z_pos, y_pos, float(getattr(self.hyp, "supcon_temp", 0.5)),
                                    anc_boxes=boxes[pos_mask],
                                    anc_bidx=b_keep[pos_mask],
                                    key_boxes=boxes,
                                    key_bidx=b_keep,
                                )
                                loss_batch = self._supcon_loss(z_pos, y_pos, float(getattr(self.hyp, "supcon_temp", 0.5))) if (loss_mem is None and num_pos >= 2) else None
                                self._supcon_val = loss_mem if (loss_mem is not None) else loss_batch
                            else:
                                self._supcon_val = None

                            with torch.no_grad():
                                if num_pos > 0:
                                    self._mq_enqueue(z[pos_mask].detach().float(), labels[pos_mask].detach())
            except Exception as e:
                LOGGER.warning(f"[SupCon] EXCEPTION: {e}")
                self._supcon_val = None
                self._supcon_prob = None
                self._supcon_stat.update({"used": "err", "roi": 0})

        # ---- Detector losses ----
        pred_distri = torch.nan_to_num(pred_distri, nan=0.0, posinf=0.0, neginf=0.0)
        pred_scores = torch.nan_to_num(pred_scores, nan=0.0, posinf=0.0, neginf=0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # đảm bảo assigner tồn tại
        self._ensure_assigner()
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss = torch.zeros(3, device=self.device)  # [box, cls, dfl]

        # cls
        loss[1] = self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum() / target_scores_sum

        # box + dfl
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        # Scale
        loss[0] *= getattr(self.hyp, "box", 1.0)
        loss[1] *= getattr(self.hyp, "cls", 1.0)
        loss[2] *= getattr(self.hyp, "dfl", 1.0)
        box_det_loss = loss[0].clone()

        # STN regularizer (optional)
        w_reg = float(getattr(self.hyp, "stn_reg", 0.0))
        if (w_reg > 0.0) and (getattr(self, "theta_for_loss", None) is not None):
            loss[0] = loss[0] + w_reg * self._stn_regularizer(self.theta_for_loss)

        # Add SupCon
        if do_supcon and (self._supcon_val is not None):
            gain = float(getattr(self.hyp, "supcon_loss_weight", getattr(self.hyp, "supcon_gain", 1.0)))
            warm = int(getattr(self.hyp, "supcon_warmup", 0))
            if warm and hasattr(self, "epoch"):
                gain *= min(1.0, float(self.epoch + 1) / float(warm))
            sv = torch.nan_to_num(self._supcon_val, nan=0.0, posinf=0.0, neginf=0.0)
            loss[0] = loss[0] + gain * sv

        supcon_log = torch.zeros((), device=loss.device)
        if self._supcon_val is not None:
            supcon_log = torch.nan_to_num(self._supcon_val.detach(), nan=0.0, posinf=0.0, neginf=0.0)

        # occasional log
        if int(getattr(self.hyp, "supcon_log", 1)):
            _cnt = getattr(self, "_supcon_log_cnt", 0)
            if cur_epoch != getattr(self, "_supcon_log_epoch", None):
                self._supcon_log_epoch = cur_epoch
                _cnt = 0
            if _cnt < 1:
                val_repr = "NONE" if self._supcon_prob is None else f"{float(self._supcon_prob):.4f}"
                LOGGER.info(f"[SupCon/log] e={-1 if cur_epoch is None else cur_epoch} "
                            f"on={int(getattr(self.hyp, 'supcon_on', 0))} val={val_repr}")
                _cnt += 1
            self._supcon_log_cnt = _cnt

        loss_items = torch.stack((box_det_loss.detach(), loss[1].detach(), loss[2].detach(), supcon_log))
        total_loss = loss.sum()
        return total_loss, loss_items

    # ---- misc ----
    def warp_bbox(self, gt_bboxes: torch.Tensor, theta: torch.Tensor, imgsz: torch.Tensor) -> torch.Tensor:
        if theta is None:
            return gt_bboxes
        B, M, _ = gt_bboxes.shape
        H, W = int(imgsz[0].item()), int(imgsz[1].item())
        x1, y1, x2, y2 = gt_bboxes.unbind(-1)
        xs = torch.stack([x1, x2, x2, x1], dim=-1)
        ys = torch.stack([y1, y1, y2, y2], dim=-1)
        x_norm = (xs / (W - 1)) * 2 - 1
        y_norm = (ys / (H - 1)) * 2 - 1
        ones = torch.ones_like(x_norm)
        pts = torch.stack([x_norm, y_norm, ones], dim=1)
        pts_flat = pts.view(B, 3, M * 4)
        bottom = torch.tensor([0, 0, 1], device=theta.device, dtype=theta.dtype).view(1, 1, 3).expand(B, 1, 3)
        full_affine = torch.cat([theta.to(theta.dtype), bottom], dim=1)
        inv_affine = torch.inverse(full_affine)[:, :2, :]
        warped = inv_affine.bmm(pts_flat).view(B, 2, M, 4).permute(0, 2, 3, 1)
        xw = (warped[..., 0] + 1) / 2 * (W - 1)
        yw = (warped[..., 1] + 1) / 2 * (H - 1)
        x_min = xw.min(dim=-1).values.clamp(0, W - 1)
        y_min = yw.min(dim=-1).values.clamp(0, H - 1)
        x_max = xw.max(dim=-1).values.clamp(1, W)
        y_max = yw.max(dim=-1).values.clamp(1, H)
        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    def _stn_regularizer(self, theta: torch.Tensor) -> torch.Tensor:
        if theta is None or not torch.is_tensor(theta):
            return torch.zeros((), device=self.device)
        T = theta.view(-1, 2, 3)
        M = T[:, :, :2]
        t = T[:, :, 2]
        I = torch.eye(2, device=self.device).unsqueeze(0).expand_as(M)
        loss_m = (M - I).pow(2).sum()
        loss_t = (t).pow(2).sum()
        return loss_m + 0.25 * loss_t
# ----------------------------------------------------------------------(ntnhan.0705)

class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the combined loss for detection and segmentation."""
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
            if (self.theta_for_loss is not None) and bool(getattr(self.hyp, "det_warp_gt", 1)):
                gt_bboxes = self.warp_bbox(gt_bboxes, self.theta_for_loss, imgsz)

        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it for pose estimation."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds, batch):
        """Calculate the loss for text-visual prompt detection."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, feats):
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1) for xi in feats]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

    def __call__(self, preds, batch):
        """Calculate the loss for text-visual prompt segmentation."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]

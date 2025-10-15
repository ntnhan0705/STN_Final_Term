# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
from .metrics import bbox_iou, probiou
from .tal import bbox2dist
from ultralytics.nn.modules.block import SpatialTransformer
from torchvision.ops import roi_align
from ultralytics.utils import LOGGER
from types import SimpleNamespace
import os
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
from types import SimpleNamespace
import os, json
import torch
import torch.nn.functional as F
from ultralytics.utils import LOGGER

# ========================================================================== #
#                           SupCon Projection Head                           #
# ========================================================================== #

class SupConProjection(torch.nn.Module):
    """Lazy MLP head: build at first forward once in_dim is known."""
    def __init__(self, out_dim: int, hidden: int = 0, bn: bool = True):
        super().__init__()
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.use_bn = bool(bn)
        self.net = None
        self.in_dim = None

    def _build(self, in_dim: int):
        import torch.nn as nn
        layers = []
        last = in_dim
        if self.hidden > 0:
            layers += [nn.Linear(in_dim, self.hidden), nn.ReLU(inplace=True)]
            if self.use_bn:
                layers.append(nn.BatchNorm1d(self.hidden))
            last = self.hidden
        layers.append(nn.Linear(last, self.out_dim))
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim

    def forward(self, x):
        if self.net is None:
            self._build(x.shape[-1])
            # đảm bảo các layer vừa tạo ở cùng device với input
            self.net.to(x.device)
        return self.net(x)


# Danh sách các key SupCon cần đồng bộ
_SUPCON_KEYS = (
    "supcon_on", "supcon_feat", "supcon_warp_gt", "supcon_out",
    "supcon_min_box", "supcon_max_per_class", "supcon_gain", "supcon_loss_weight",
    "supcon_temp", "supcon_warmup", "supcon_log", "supcon_use_mem", "supcon_queue",
    "supcon_schedule", "supcon_proj_dim", "supcon_proj_hidden", "supcon_proj_bn"
)

# ========================================================================== #
#                            v8DetectionLoss (STN)                           #
# ========================================================================== #

class v8DetectionLoss:
    """YOLOv8 detection loss + SupCon (ROIAlign trên fmap sau STN)."""

    def __init__(self, model, tal_topk=10):
        import torch.nn as nn
        from ultralytics.nn.modules.block import SpatialTransformer as _STN
        from ultralytics.utils.tal import TaskAlignedAssigner

        self.device = next(model.parameters()).device
        self.model = model

        # --- Safe hyp (SimpleNamespace) + defaults ---
        raw_hyp = getattr(model, "args", None)
        if raw_hyp is None or isinstance(raw_hyp, dict):
            self.hyp = SimpleNamespace(**(raw_hyp or {}))
        else:
            self.hyp = raw_hyp

        # Chuẩn hoá supcon_* ngay từ đầu (không để None)
        self._normalize_and_mirror_supcon(from_env=True)

        LOGGER.info(f"[LOSS INIT] use_subcon={bool(getattr(self.hyp,'supcon_on',0))}, "
                    f"subcon_weight={getattr(self.hyp,'supcon_loss_weight', None)}")

        head = model.model[-1]

        # ---- Head properties
        self.nc = head.nc
        self.reg_max = head.reg_max
        self.no = self.nc + self.reg_max * 4
        self.stride = head.stride
        self.use_dfl = self.reg_max > 1
        self._proj_head = None  # projection head (lazy)

        # ---- Core losses / utils
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

        # ---- SupCon memory queue
        self._mq_feats = None
        self._mq_labels = None
        self._mq_ptr = 0
        self._mq_size = int(getattr(self.hyp, "supcon_queue", 2048))
        self._mq_ready = False

        # ---- STN + feature-after-STN hooks (match TSNE)
        self.theta_for_loss = None
        self._stn_module = None

        # Sau STN: lấy fmap 4D, C>=128
        self._aft_min_channels = 128
        self._aft_hooks = []
        self._aft_last = None

        stn_seen = False
        for m in model.modules():
            if isinstance(m, _STN):
                self._stn_module = m
                try:
                    # Cho STN đẩy theta vào loss (nếu STN hỗ trợ)
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

        # ---- misc
        self._supcon_stat = {}
        self._printed_hyp = False

    # --------------------- Helpers: đồng bộ / normalize supcon_* ---------------------
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
        # Không để None, đảm bảo số thực cho weight
        gain = float(getattr(self.hyp, "supcon_gain", 2.5))
        w = getattr(self.hyp, "supcon_loss_weight", None)
        if w is None:
            w = gain
        try:
            w = float(w)
        except (TypeError, ValueError):
            w = gain
        setattr(self.hyp, "supcon_loss_weight", w)

        # Bật/tắt, chọn feat
        if getattr(self.hyp, "supcon_feat", None) in (None, ""):
            setattr(self.hyp, "supcon_feat", "stn")

        # Mirror ngược để nơi khác không thấy None
        if getattr(self.model, "args", None) is None or isinstance(self.model.args, dict):
            self.model.args = SimpleNamespace(**(getattr(self.model, "args", {}) or {}))
        for k in _SUPCON_KEYS:
            if hasattr(self.hyp, k):
                setattr(self.model.args, k, getattr(self.hyp, k))

    def _normalize_and_mirror_supcon(self, from_env=False):
        # Copy từ model.args -> self.hyp
        self._merge_from_model_args()
        # Copy từ ENV nếu muốn
        if from_env:
            self._merge_from_env()
        # Chốt giá trị cuối cùng và mirror ngược
        self._finalize_supcon()

    # --------------------- SupCon: memory queue helpers ---------------------
    @torch.no_grad()
    def _mq_enqueue(self, feats: torch.Tensor, labels: torch.Tensor):
        if self._mq_size <= 0 or feats is None or feats.numel() == 0:
            return
        N, C = feats.shape
        if self._mq_feats is None:
            self._mq_feats = torch.zeros(self._mq_size, C, device=self.device, dtype=feats.dtype)
            self._mq_labels = torch.full((self._mq_size,), -1, device=self.device, dtype=labels.dtype)
            self._mq_ptr = 0

        end = self._mq_ptr + N
        if end <= self._mq_size:
            self._mq_feats[self._mq_ptr:end] = feats
            self._mq_labels[self._mq_ptr:end] = labels
        else:
            first = self._mq_size - self._mq_ptr
            if first > 0:
                self._mq_feats[self._mq_ptr:] = feats[:first]
                self._mq_labels[self._mq_ptr:] = labels[:first]
            rest = N - first
            if rest > 0:
                self._mq_feats[:rest] = feats[first:]
                self._mq_labels[:rest] = labels[first:]
        self._mq_ptr = (self._mq_ptr + N) % self._mq_size
        if not self._mq_ready and self._mq_ptr == 0:
            self._mq_ready = True

    def _supcon_loss_memory(self, z: torch.Tensor, y: torch.Tensor, T: float):
        if z is None or z.numel() == 0:
            return None
        z = F.normalize(z, dim=1)
        y = y.view(-1).long()
        B = z.size(0)

        keys = z.detach()
        klabels = y.detach()

        if int(getattr(self.hyp, "supcon_use_mem", 1)) and (self._mq_feats is not None):
            valid = self._mq_labels.ge(0)
            if valid.any():
                keys = torch.cat([keys, self._mq_feats[valid]], dim=0)
                klabels = torch.cat([klabels, self._mq_labels[valid]], dim=0)

        K = keys.size(0)
        if K == 0:
            return None

        logits = (z @ keys.t()) / float(T)
        same = y.view(-1, 1).eq(klabels.view(1, -1))

        if K >= B:
            idx = torch.arange(B, device=logits.device)
            logits[idx, idx] = -float("inf")
            same[idx, idx] = False

        pos_cnt = same.sum(1)
        valid_anchor = pos_cnt > 0
        if not valid_anchor.any():
            self._supcon_stat["pos_batch"] = 0
            return None

        log_den = torch.logsumexp(logits, dim=1)
        pos_logits = torch.where(same, logits, logits.new_full(logits.shape, -float("inf")))
        log_num = torch.logsumexp(pos_logits, dim=1)
        loss = (-(log_num - log_den))[valid_anchor].mean()

        self._supcon_stat["pos_batch"] = int(valid_anchor.sum().item())
        return loss

    def _supcon_loss(self, z: torch.Tensor, y: torch.Tensor, T: float):
        if z is None or z.numel() == 0:
            return None
        z = F.normalize(z, dim=1)
        y = y.view(-1).long()
        B = z.size(0)
        logits = (z @ z.t()) / float(T)
        idx = torch.arange(B, device=z.device)
        logits[idx, idx] = -float("inf")
        same = y.view(-1, 1).eq(y.view(1, -1))
        pos_cnt = same.sum(1)
        valid_anchor = pos_cnt > 0
        if not valid_anchor.any():
            return None
        log_den = torch.logsumexp(logits, dim=1)
        pos_logits = torch.where(same, logits, logits.new_full(logits.shape, -float("inf")))
        log_num = torch.logsumexp(pos_logits, dim=1)
        return (-(log_num - log_den))[valid_anchor].mean()

    # --------------------- YOLO helpers ---------------------
    def set_theta(self, theta: torch.Tensor):
        self.theta_for_loss = theta.detach() if theta is not None else None

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
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _after_hook(self, _m, _inp, out):
        # Lấy fmap 4D, đủ sâu (C>=_aft_min_channels), giữ grad để SupCon backprop
        if torch.is_tensor(out) and out.dim() == 4 and out.shape[1] >= self._aft_min_channels:
            self._aft_last = out

    # --------------------- Forward ---------------------
    def __call__(self, preds, batch):
        from torchvision.ops import roi_align

        # Trước mỗi forward: REFRESH cấu hình (đọc lại model.args/ENV nếu callback vừa bơm)
        self._normalize_and_mirror_supcon(from_env=True)

        if not self._printed_hyp:
            LOGGER.info(f"[LOSS/HYP effective] on={getattr(self.hyp,'supcon_on',0)}, "
                        f"feat={getattr(self.hyp,'supcon_feat',None)}, "
                        f"w={getattr(self.hyp,'supcon_loss_weight',None)}, "
                        f"gain={getattr(self.hyp,'supcon_gain',None)}")
            self._printed_hyp = True

        # đọc cấu hình
        cfg = SimpleNamespace(
            on=int(getattr(self.hyp, "supcon_on", 0) or 0),
            feat=str(getattr(self.hyp, "supcon_feat", "stn")).lower(),
            warp=int(getattr(self.hyp, "supcon_warp_gt", 0)),
            out=int(getattr(self.hyp, "supcon_out", 7)),
            min_box=int(getattr(self.hyp, "supcon_min_box", 1)),
            max_pc=int(getattr(self.hyp, "supcon_max_per_class", 0)),
            temp=float(getattr(self.hyp, "supcon_temp", 0.2)),
            gain=float(getattr(self.hyp, "supcon_gain", 2.5)),
            warmup=int(getattr(self.hyp, "supcon_warmup", 0)),
            use_mem=int(getattr(self.hyp, "supcon_use_mem", 1)),
            proj_dim=int(getattr(self.hyp, "supcon_proj_dim", 0)),
            proj_hidden=int(getattr(self.hyp, "supcon_proj_hidden", 0)),
            proj_bn=int(getattr(self.hyp, "supcon_proj_bn", 1)),
            log=int(getattr(self.hyp, "supcon_log", 0)),
        )

        # parse YOLO preds
        loss = torch.zeros(3, device=self.device)
        self._supcon_val = None
        self._supcon_stat = {}

        feats = preds[1] if isinstance(preds, tuple) else preds  # [P3,P4,P5]
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        B = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), B, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # --- SupCon (ROI từ fmap SAU STN; fallback P3)
        if cfg.on:
            try:
                # chọn feature map
                if cfg.feat == "stn" and (self._aft_last is not None):
                    feat_map, src = self._aft_last, "after_stn"
                else:
                    feat_map, src = feats[0], "p3"

                valid = mask_gt.squeeze(-1)
                if valid.any():
                    b_idx, m_idx = valid.nonzero(as_tuple=False).T
                    boxes_bm = gt_bboxes.detach().clone()

                    # warp GT theo theta ngược nếu đang dùng fmap sau STN
                    if cfg.warp and (self.theta_for_loss is not None) and (src == "after_stn"):
                        boxes_bm = self.warp_bbox(boxes_bm, self.theta_for_loss, imgsz)

                    boxes = boxes_bm[b_idx, m_idx]                # [N,4] xyxy px
                    labels = gt_labels[b_idx, m_idx, 0].long()    # [N]

                    # bỏ bbox quá nhỏ + giới hạn per-class
                    wh = boxes[:, 2:] - boxes[:, :2]
                    keep = (wh[:, 0] >= cfg.min_box) & (wh[:, 1] >= cfg.min_box)
                    if keep.any():
                        boxes, labels, b_keep = boxes[keep], labels[keep], b_idx[keep]
                        if cfg.max_pc > 0:
                            uniq = labels.unique(sorted=True)
                            sel = []
                            for c in uniq.tolist():
                                idxc = torch.nonzero(labels == c, as_tuple=False).view(-1)
                                sel.append(idxc[:cfg.max_pc] if idxc.numel() > cfg.max_pc else idxc)
                            if sel:
                                idxs = torch.cat(sel)
                                boxes, labels, b_keep = boxes[idxs], labels[idxs], b_keep[idxs]

                        if boxes.numel() > 0:
                            # ảnh -> fmap
                            H, W = int(imgsz[0].item()), int(imgsz[1].item())
                            _, Cf, Hf, Wf = feat_map.shape
                            x1, y1, x2, y2 = boxes.unbind(1)
                            x1 = x1.clamp(0, W - 1); y1 = y1.clamp(0, H - 1)
                            x2 = torch.maximum(x2.clamp(0, W - 1), x1 + 1)
                            y2 = torch.maximum(y2.clamp(0, H - 1), y1 + 1)
                            boxes = torch.stack([x1, y1, x2, y2], 1)
                            sx, sy = Wf / float(W), Hf / float(H)
                            fx1, fy1 = boxes[:, 0] * sx, boxes[:, 1] * sy
                            fx2, fy2 = boxes[:, 2] * sx, boxes[:, 3] * sy
                            rois = torch.stack([b_keep.float(), fx1.float(), fy1.float(), fx2.float(), fy2.float()],
                                               1).to(feat_map.device, feat_map.dtype)

                            pooled = roi_align(
                                input=feat_map, boxes=rois,
                                output_size=(cfg.out, cfg.out),
                                spatial_scale=1.0, sampling_ratio=0, aligned=True
                            )
                            z = pooled.mean(dim=(2, 3))  # [N,C]

                            # projection head (lazy)
                            if cfg.proj_dim > 0:
                                if self._proj_head is None:
                                    self._proj_head = SupConProjection(
                                        out_dim=cfg.proj_dim,
                                        hidden=cfg.proj_hidden,
                                        bn=bool(cfg.proj_bn)
                                    ).to(self.device)
                                    # để optimizer bên ngoài có thể add params
                                    setattr(self.model, "supcon_proj", self._proj_head)
                                    LOGGER.info(f"[SupConProj] created (lazy): out_dim={cfg.proj_dim}, "
                                                f"hidden={cfg.proj_hidden}, bn={bool(cfg.proj_bn)}")
                                z = self._proj_head(z)

                            mem_valid = int(self._mq_labels.ge(0).sum().item()) if (self._mq_labels is not None) else 0
                            self._supcon_stat.update({"used": src, "roi": int(z.size(0)), "mem_valid": mem_valid})

                            # compute supcon loss (ưu tiên memory)
                            loss_mem = self._supcon_loss_memory(z, labels, cfg.temp)
                            loss_batch = self._supcon_loss(z, labels, cfg.temp) if loss_mem is None else None
                            self._supcon_val = loss_mem if (loss_mem is not None) else loss_batch

                            with torch.no_grad():
                                self._mq_enqueue(F.normalize(z, dim=1).detach(), labels.detach())
                # else: no valid targets -> supcon skipped
            except Exception as e:
                LOGGER.warning(f"[SupCon] EXCEPTION: {e}")
                self._supcon_val = None
                self._supcon_stat.update({"used": "err", "roi": 0, "pos_batch": 0, "mem_valid": 0, "val": None})

        # --- YOLO base losses
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        # cls
        loss[1] = self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum() / target_scores_sum

        # box + dfl
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        # apply gains (an toàn nếu thiếu)
        loss[0] *= getattr(self.hyp, "box", 1.0)
        loss[1] *= getattr(self.hyp, "cls", 1.0)
        loss[2] *= getattr(self.hyp, "dfl", 1.0)

        # add SupCon (weighted) — luôn an toàn với None
        if int(getattr(self.hyp, "supcon_on", 0)) and (self._supcon_val is not None):
            w = getattr(self.hyp, "supcon_loss_weight", None)
            if w is None:
                w = getattr(self.hyp, "supcon_gain", 2.5)
            gain = float(w)

            warm = int(getattr(self.hyp, "supcon_warmup", 0))
            if warm and hasattr(self, "epoch"):
                gain *= min(1.0, float(self.epoch + 1) / float(warm))

            loss[0] = loss[0] + gain * self._supcon_val

        supcon_display = (
            torch.nan_to_num(self._supcon_val.detach(), nan=0.0, posinf=50.0, neginf=0.0)
            if (int(getattr(self.hyp, "supcon_on", 0)) and (self._supcon_val is not None)) else
            torch.zeros((), device=loss.device)
        )
        loss_items = torch.stack((loss[0].detach(), loss[1].detach(), loss[2].detach(), supcon_display))
        total = loss.sum()
        return total, loss_items

    # --------------------- warp bbox bằng affine ngược của STN ---------------------
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
        pts = torch.stack([x_norm, y_norm, ones], dim=1)   # [B,3,M,4]
        pts_flat = pts.view(B, 3, M * 4)

        bottom = torch.tensor([0, 0, 1], device=theta.device, dtype=theta.dtype).view(1, 1, 3).expand(B, 1, 3)
        full_affine = torch.cat([theta.to(theta.dtype), bottom], dim=1)  # [B,3,3]
        inv_affine = torch.inverse(full_affine)[:, :2, :]

        warped = inv_affine.bmm(pts_flat).view(B, 2, M, 4).permute(0, 2, 3, 1)
        xw = (warped[..., 0] + 1) / 2 * (W - 1)
        yw = (warped[..., 1] + 1) / 2 * (H - 1)

        x_min = xw.min(dim=-1).values.clamp(0, W - 1)
        y_min = yw.min(dim=-1).values.clamp(0, H - 1)
        x_max = xw.max(dim=-1).values.clamp(1, W)
        y_max = yw.max(dim=-1).values.clamp(1, H)
        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

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
            # >>> ADD: warp GT cho detection loss <<<
            if (self.theta_for_loss is not None) and bool(getattr(self.hyp, "det_warp_gt", 1)):
                gt_bboxes = self.warp_bbox(gt_bboxes, self.theta_for_loss, imgsz)
            # <<< END ADD
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

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from ultralytics.utils import LOGGER

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Residual, SwiGLUFFN
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

from torch import Tensor
from .block import SpatialTransformer
__all__ = "Detect", "DetectSDTN", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect", "YOLOEDetect", "YOLOESegment"


class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc=80, ch=()):
        """Initialize the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (List[torch.Tensor]): Input feature maps from different levels.

        Returns:
            (dict | tuple):

                - If in training mode, returns a dictionary containing outputs of both one2many and one2one detections.
                - If not in training mode, returns processed detections or a tuple with processed detections and raw outputs.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """
        Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (List[torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and not (self.end2end or self.xyxy), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


# -------------------------------------------------------------------- (ntnhan.0705)
def _invert_affine(theta: Tensor) -> Tensor:
    """Đảo batch affine 2×3 (B,2,3) ⇒ (B,2,3). Đảm bảo hoạt động với float32."""
    B = theta.size(0)
    if B == 0: # Thêm kiểm tra batch rỗng
        return torch.empty((0, 2, 3), device=theta.device, dtype=torch.float32)

    # <<< THÊM ÉP KIỂU NGAY TRƯỚC KHI DÙNG >>>
    # Đảm bảo theta là float32 trước khi thực hiện các phép toán linalg
    theta_float = theta.float()
    # <<< KẾT THÚC ÉP KIỂU >>>

    # ghép hàng [0 0 1] ➜ 3×3, rồi inverse (sử dụng theta_float)
    # Tạo bottom tensor cùng device và dtype float32
    bottom = torch.tensor([[0.0, 0.0, 1.0]], device=theta_float.device, dtype=torch.float32).expand(B, -1, -1) # Shape (B, 1, 3)
    full   = torch.cat([theta_float, bottom], dim=1)              # (B,3,3), float32

    # Tính nghịch đảo - input `full` đảm bảo là float32
    try:
        inv = torch.linalg.inv(full)[:, :2]                     # (B,2,3), float32
    except Exception as e:
        LOGGER.error(f"Error during torch.linalg.inv: {e}. Input shape: {full.shape}, dtype: {full.dtype}")
        # Fallback: Trả về ma trận identity nếu không nghịch đảo được
        inv = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=theta_float.device, dtype=torch.float32).expand(B, -1, -1)

    # Trả về kết quả float32
    return inv

class DetectSDTN(Detect):
    """
    Detect head + Spatial **De**-Transformer.
    Khi STN bật, head tự “gỡ xoắn” feature-map bằng θ-¹ nên bbox trả
    về vẫn nằm trên hệ toạ độ ảnh gốc.
    """

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        self._stn: SpatialTransformer | None = None   # tìm lười biếng
        self.active = True                            # callback ngoài có thể tắt
        self._log_counter = 0 # <<< THÊM BIẾN ĐẾM >>>

    # ---------------- private helpers -----------------
    def _lazy_find_stn(self):
        """Tìm STN duy nhất qua registry; nếu không có thì sinh θ = identity."""
        if self._stn is None:  # Chỉ chạy 1 lần
            # 1) Ưu tiên lấy STN đầu tiên từ registry toàn cục
            if hasattr(SpatialTransformer, 'registry') and SpatialTransformer.registry:
                self._stn = SpatialTransformer.registry[0]
                LOGGER.info("[DetectSDTN DEBUG] Found STN via global registry.")
            else:
                LOGGER.warning("[DetectSDTN DEBUG] SpatialTransformer.registry is empty or not found. STN module instance not located.")
                self._stn = None # Đảm bảo _stn là None nếu không tìm thấy

            # 2) Nếu không tìm được STN thật, tạo "STN giả" chứa theta = identity
            if self._stn is None:
                LOGGER.info("[DetectSDTN DEBUG] Creating Mock STN with identity theta as fallback.")
                # Tạo một đối tượng đơn giản có thuộc tính 'theta'
                class MockSTN:
                    def __init__(self, identity_theta):
                        self.theta = identity_theta
                try:
                     # Lấy device và dtype từ một parameter của head để đảm bảo tương thích
                     device = next(self.parameters()).device
                     dtype = torch.float32 # Theta thường là float32
                     # Tạo theta identity với batch_size=1 (sẽ tự broadcast khi cần)
                     eye = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device, dtype=dtype)
                     theta_identity = eye.view(1, 2, 3)
                     self._stn = MockSTN(theta_identity)
                     # Kiểm tra lại xem mock có theta không
                     if not hasattr(self._stn, 'theta') or self._stn.theta is None:
                          LOGGER.error("[DetectSDTN DEBUG] Failed to initialize Mock STN theta.")
                          self._stn = None # Reset về None nếu mock lỗi
                except Exception as e:
                     LOGGER.error(f"[DetectSDTN DEBUG] Exception creating Mock STN: {e}")
                     self._stn = None # Đảm bảo _stn là None nếu có lỗi

        # Trả về _stn tìm được hoặc mock STN (hoặc None nếu cả hai đều thất bại)
        return self._stn

    @staticmethod
    def _dewarp_feats(
        feats: list[Tensor] | tuple[Tensor, ...],
        theta_inv: Tensor
    ) -> list[Tensor]:
        """Affine-grid + grid-sample cho từng feature‐map."""
        out = []
        for f in feats:                 # mỗi f: (B,C,H,W)
            B, C, H, W = f.shape
            grid = F.affine_grid(theta_inv, (B, C, H, W), align_corners=False)
            out.append(
                F.grid_sample(
                    f, grid,
                    mode="bilinear", padding_mode="zeros",
                    align_corners=False
                )
            )
        return out

    # ---------------- forward -------------------------
    def forward(self, x):
        """
        x: list feature P3, P4, P5.
        Nếu self.active=False ➜ bỏ qua de-warp.
        """
        # <<< THÊM BIẾN ĐẾM VÀ ĐIỀU KIỆN IN LOG >>>
        log_interval = 500 # In log mỗi 500 batch
        log_this_batch = False
        if hasattr(self, '_log_counter'):
             self._log_counter += 1
             # In ở batch đầu tiên (counter=1) hoặc các batch là bội số của interval
             if self._log_counter == 1 or self._log_counter % log_interval == 0:
                  log_this_batch = True
        else: # Khởi tạo nếu chưa có
             self._log_counter = 1
             log_this_batch = True
        # <<< KẾT THÚC THÊM BIẾN ĐẾM >>>

        theta_from_stn = None
        stn_found = False
        stn_instance = None # Biến tạm để giữ instance STN tìm được
        try:
            stn_instance = self._lazy_find_stn() # Gọi hàm đã sửa
            if stn_instance and hasattr(stn_instance, 'theta') and stn_instance.theta is not None:
                theta_from_stn = stn_instance.theta
                stn_found = True

            is_active = self.active and isinstance(x, (list, tuple))
            theta_shape = theta_from_stn.shape if theta_from_stn is not None else "None"

            # Chỉ in nếu đến lượt
            if log_this_batch:
                 # In thêm giá trị theta sample đầu tiên
                 theta_val_str = "N/A"
                 if theta_from_stn is not None and theta_from_stn.numel() > 0:
                      # Đảm bảo index không vượt quá batch size thực tế
                      sample_idx = 0
                      current_batch_size = theta_from_stn.shape[0]
                      if sample_idx < current_batch_size:
                           theta_sample = theta_from_stn[sample_idx].detach().cpu().numpy()
                           theta_val_str = f"[[{theta_sample[0, 0]:.4f}, {theta_sample[0, 1]:.4f}, {theta_sample[0, 2]:.4f}], [{theta_sample[1, 0]:.4f}, {theta_sample[1, 1]:.4f}, {theta_sample[1, 2]:.4f}]]"
                      else:
                           theta_val_str = f"Index {sample_idx} out of bounds for batch size {current_batch_size}"


                 LOGGER.info(f"[DetectSDTN DEBUG] (Batch {self._log_counter}) "
                             f"active={is_active}, stn_found={stn_found}, theta shape: {theta_shape}. "
                             f"Sample[0]: {theta_val_str}") # In giá trị theta

        except Exception as e:
            # Chỉ in nếu đến lượt
            if log_this_batch:
                 LOGGER.warning(f"[DetectSDTN DEBUG] (Batch {self._log_counter}) Error checking theta: {e}")

        # Thực hiện de-warp nếu cần
        if self.active and isinstance(x, (list, tuple)):
            # Dùng stn_instance đã lấy ở trên, không cần gọi lại _lazy_find_stn()
            if stn_instance and hasattr(stn_instance, 'theta') and stn_instance.theta is not None:
                try:
                    # Đảm bảo theta có cùng batch size với features trước khi invert và dewarp
                    current_batch_size_feat = x[0].shape[0]
                    theta_batch_size = stn_instance.theta.shape[0]

                    if theta_batch_size == current_batch_size_feat:
                         theta_to_use = stn_instance.theta
                    elif theta_batch_size == 1 and current_batch_size_feat > 1:
                         # Nếu theta là (1, 2, 3) và batch > 1, broadcast nó
                         theta_to_use = stn_instance.theta.expand(current_batch_size_feat, -1, -1)
                         if log_this_batch:
                              LOGGER.info(f"[DetectSDTN DEBUG] (Batch {self._log_counter}) Broadcasted theta from shape {stn_instance.theta.shape} to {theta_to_use.shape}")
                    else:
                         # Trường hợp không khớp batch size không mong muốn
                         if log_this_batch:
                              LOGGER.error(f"[DetectSDTN DEBUG] (Batch {self._log_counter}) Mismatch batch size! Theta: {theta_batch_size}, Feats: {current_batch_size_feat}. Skipping de-warp.")
                         # Không thực hiện de-warp nếu batch size không khớp
                         theta_to_use = None # Đặt lại để bỏ qua de-warp

                    # Chỉ de-warp nếu theta_to_use hợp lệ
                    if theta_to_use is not None:
                         theta_inv = _invert_affine(theta_to_use)
                         x_dewarped = self._dewarp_feats(x, theta_inv)

                         # Chỉ in nếu đến lượt
                         if log_this_batch:
                              feat_shapes_before = [f.shape for f in x]
                              feat_shapes_after = [f.shape for f in x_dewarped]
                              LOGGER.info(f"[DetectSDTN DEBUG] (Batch {self._log_counter}) De-warped features. "
                                          f"Before: {feat_shapes_before}, After: {feat_shapes_after}")
                         x = x_dewarped
                    # else: # Log lỗi đã được xử lý ở trên

                except Exception as e:
                     # Chỉ in nếu đến lượt
                     if log_this_batch:
                          LOGGER.error(f"[DetectSDTN DEBUG] (Batch {self._log_counter}) Error during de-warping: {e}. Using original features.")
                     # Giữ nguyên x nếu có lỗi
            else:
                 # Chỉ in nếu đến lượt
                 if log_this_batch:
                     LOGGER.warning(f"[DetectSDTN DEBUG] (Batch {self._log_counter}) "
                                    f"STN found={stn_found} but theta is None or invalid. Skipping de-warp.")

        # Gọi Detect.forward (original) để tính logits / loss
        return super().forward(x)
# -------------------------------------------------------------------- (ntnhan.0705)


class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # Precompute normalization factor to increase numerical stability
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::ndim] = y[:, 2::ndim].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False  # export mode

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class LRPCHead(nn.Module):
    """Lightweight Region Proposal and Classification Head for efficient object detection."""

    def __init__(self, vocab, pf, loc, enabled=True):
        """Initialize LRPCHead with vocabulary, proposal filter, and localization components."""
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    def conv2linear(self, conv):
        """Convert a 1x1 convolutional layer to a linear layer."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat, loc_feat, conf):
        """Process classification and localization features to generate detection proposals."""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return (self.loc(loc_feat), cls_feat.transpose(-1, -2)), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (loc_feat, cls_feat.flatten(2)), torch.ones(
                cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool
            )


class YOLOEDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    is_fused = False

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn is True
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )

        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats):
        """Fuse text features with model weights for efficient inference."""
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        for cls_head, bn_head in zip(self.cv3, self.cv4):
            assert isinstance(cls_head, nn.Sequential)
            assert isinstance(bn_head, BNContrastiveHead)
            conv = cls_head[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_head.logit_scale
            bias = bn_head.bias
            norm = bn_head.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_head[-1] = conv

            bn_head.fuse()

        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def get_tpe(self, tpe):
        """Get text prompt embeddings with normalization."""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x, vpe):
        """Get visual prompt embeddings with spatial awareness."""
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward_lrpc(self, x, return_mask=False):
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        masks = []
        assert self.is_fused, "Prompt-free inference requires model to be fused!"
        for i in range(self.nl):
            cls_feat = self.cv3[i](x[i])
            loc_feat = self.cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            x[i], mask = self.lrpc[i](
                cls_feat, loc_feat, 0 if self.export and not self.dynamic else getattr(self, "conf", 0.001)
            )
            masks.append(mask)
        shape = x[0][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors([b[0] for b in x], self.stride, 0.5))
            self.shape = shape
        box = torch.cat([xi[0].view(shape[0], self.reg_max * 4, -1) for xi in x], 2)
        cls = torch.cat([xi[1] for xi in x], 2)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        mask = torch.cat(masks)
        y = torch.cat((dbox if self.export and not self.dynamic else dbox[..., mask], cls.sigmoid()), 1)

        if return_mask:
            return (y, mask) if self.export else ((y, x), mask)
        else:
            return y if self.export else (y, x)

    def forward(self, x, cls_pe, return_mask=False):
        """Process features with class prompt embeddings to generate detections."""
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x, return_mask)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), cls_pe)), 1)
        if self.training:
            return x
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize biases for detection heads."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, c, s in zip(m.cv2, m.cv3, m.cv4, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)


class YOLOESegment(YOLOEDetect):
    """YOLO segmentation head with text embedding capabilities."""

    def __init__(self, nc=80, nm=32, npr=256, embed=512, with_bn=False, ch=()):
        """Initialize YOLOESegment with class count, mask parameters, and embedding dimensions."""
        super().__init__(nc, embed, with_bn, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)

    def forward(self, x, text):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        has_lrpc = hasattr(self, "lrpc")

        if not has_lrpc:
            x = YOLOEDetect.forward(self, x, text)
        else:
            x, mask = YOLOEDetect.forward(self, x, text, return_mask=True)

        if self.training:
            return x, mc, p

        if has_lrpc:
            mc = (mc * mask.int()) if self.export and not self.dynamic else mc[..., mask]

        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """
        Runs the forward pass of the module, returning bounding box and classification scores for the input.

        Args:
            x (List[torch.Tensor]): List of feature maps from the backbone.
            batch (dict, optional): Batch information for training.

        Returns:
            (tuple | torch.Tensor): During training, returns a tuple of bounding boxes, scores, and other metadata.
                During inference, returns a tensor of shape (bs, 300, 4+nc) containing bounding boxes and class scores.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """
        Generates anchor bounding boxes for given shapes with specific grid size and validates them.

        Args:
            shapes (list): List of feature map shapes.
            grid_size (float, optional): Base size of grid cells. Default is 0.05.
            dtype (torch.dtype, optional): Data type for tensors. Default is torch.float32.
            device (str, optional): Device to create tensors on. Default is "cpu".
            eps (float, optional): Small value for numerical stability. Default is 1e-2.

        Returns:
            (tuple): Tuple containing anchors and valid mask tensors.
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """
        Processes and returns encoder inputs by getting projection features from input and concatenating them.

        Args:
            x (List[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            (tuple): Tuple containing processed features and their shapes.
        """
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """
        Generates and prepares the input required for the decoder from the provided features and shapes.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings. Default is None.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes. Default is None.

        Returns:
            (tuple): Tuple containing embeddings, reference bounding boxes, encoded bounding boxes, and scores.
        """
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """Removes the one2many head."""
        self.cv2 = self.cv3 = nn.ModuleList([nn.Identity()] * self.nl)

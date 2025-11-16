# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Check a model's accuracy on a test or val split of a dataset.

Usage:
    $ yolo mode=val model=yolo11n.pt data=coco8.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolo11n.pt                 # PyTorch
                          yolo11n.torchscript        # TorchScript
                          yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolo11n_openvino_model     # OpenVINO
                          yolo11n.engine             # TensorRT
                          yolo11n.mlpackage          # CoreML (macOS-only)
                          yolo11n_saved_model        # TensorFlow SavedModel
                          yolo11n.pb                 # TensorFlow GraphDef
                          yolo11n.tflite             # TensorFlow Lite
                          yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolo11n_paddle_model       # PaddlePaddle
                          yolo11n.mnn                # MNN
                          yolo11n_ncnn_model         # NCNN
                          yolo11n_imx_model          # Sony IMX
                          yolo11n_rknn_model         # Rockchip RKNN
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis, DEFAULT_CFG
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from pathlib import Path


class BaseValidator:
    """
    A base class for creating validators.

    This class provides the foundation for validation processes, including model evaluation, metric computation, and
    result visualization.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary containing dataset information.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names mapping.
        seen (int): Number of images seen so far during validation.
        stats (dict): Statistics collected during validation.
        confusion_matrix: Confusion matrix for classification evaluation.
        nc (int): Number of classes.
        iouv (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (list): List to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
            batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.

    Methods:
        __call__: Execute validation process, running inference on dataloader and computing performance metrics.
        match_predictions: Match predictions to ground truth objects using IoU.
        add_callback: Append the given callback to the specified event.
        run_callbacks: Run all callbacks associated with a specified event.
        get_dataloader: Get data loader from dataset path and batch size.
        build_dataset: Build dataset from image path.
        preprocess: Preprocess an input batch.
        postprocess: Postprocess the predictions.
        init_metrics: Initialize performance metrics for the YOLO model.
        update_metrics: Update metrics based on predictions and batch.
        finalize_metrics: Finalize and return all metrics.
        get_stats: Return statistics about the model's performance.
        check_stats: Check statistics.
        print_results: Print the results of the model's predictions.
        get_desc: Get description of the YOLO model.
        on_plot: Register plots (e.g. to be consumed in callbacks).
        plot_val_samples: Plot validation samples during training.
        plot_predictions: Plot YOLO model predictions on batch images.
        pred_to_json: Convert predictions to JSON format.
        eval_json: Evaluate and return JSON format of prediction statistics.
    """

    def __init__(self, args=None, dataloader=None, save_dir=None, pbar=None):
        """Initializes a BaseValidator instance for validation."""

        super().__init__()  # Sửa lỗi TypeError
        self.args = args or get_cfg(DEFAULT_CFG)
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.pbar = pbar
        self.stride = 32
        self.data = None
        self.model = None
        self.metrics = None
        self.callbacks = callbacks.get_default_callbacks()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.loss = torch.zeros(3)
        self.skip_loss = False
        self.jdict = []
        self.plots = {}
        self.on_plot = self.on_plot
        self.pbar_desc = self.get_desc()
    # --- BEGIN PATCH: minimal compat for non-train evaluate ---
    def get_data(self):
        """
        Return parsed dataset dict for the current task.
        - If args.data is a path/str, parse it (detect -> check_det_dataset, fallback -> check_cls_dataset).
        - Else, reuse existing self.data.
        """
        d = getattr(self.args, "data", None)
        if isinstance(d, (str, Path)):
            try:
                return check_det_dataset(d)  # detect/segment-style data.yaml
            except Exception:
                try:
                    return check_cls_dataset(d)  # cls-style
                except Exception:
                    return d
        return d if d is not None else getattr(self, "data", None)
    # --- END PATCH ---

    @torch.no_grad()
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = False
            self.save_dir = trainer.save_dir
            self.loss = torch.zeros_like(trainer.loss_items, device=self.device)
            model = trainer.ema.ema or trainer.model
            model = model.float()
            model.eval()
            self.model = trainer.model
            if not self.dataloader:
                self.dataloader = self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            if self.pbar is None:
                self.pbar = TQDM(self.dataloader, desc=self.pbar_desc, total=len(self.dataloader))

        else:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_val_start")
            assert model is not None, "Either trainer or model is required for validation"
            self.device = select_device(self.args.device, self.args.batch)
            self.args.half &= self.device.type != "cpu"
            if self.args.half:
                model.half()
            # === PATCH: nhận đường dẫn weights và nạp bằng AutoBackend ===

            if isinstance(model, (str, Path)):
                # model là đường dẫn: nạp bằng AutoBackend
                model = AutoBackend(
                    model,
                    device=self.device,
                    dnn=getattr(self.args, "dnn", False),
                    fp16=self.args.half
                )
            elif isinstance(model, torch.nn.Module):
                # model là nn.Module thuần: chuẩn hóa dtype
                if self.args.half and self.device.type != "cpu" and hasattr(model, "half"):
                    model = model.half()
                else:
                    model = model.float()
            else:
                raise TypeError(f"validator: unsupported model type {type(model)}")

            self.model = model
            self.model.eval()

            self.data = self.get_data()
            if self.device.type == "cpu":
                self.args.workers = 0
            if not self.dataloader:
                self.dataloader = self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            if self.pbar is None:
                self.pbar = TQDM(self.dataloader, desc=self.pbar_desc, total=len(self.dataloader))

            # Chuẩn hoá imgsz (int -> (h, w))
            imsz = self.args.imgsz
            if isinstance(imsz, int):
                imsz = (imsz, imsz)

            bs = 1 if getattr(self.args, "pt", True) else getattr(self.args, "batch", 1)
            warm = getattr(model, "warmup", None)
            if callable(warm):
                try:
                    warm(imgsz=(bs, 3, *imsz))
                except Exception as e:
                    LOGGER.warning(f"Warmup skipped: {e}")
            else:
                try:
                    dev = next(model.parameters()).device
                    with torch.no_grad():
                        model(torch.zeros(bs, 3, imsz[0], imsz[1], device=dev))
                except Exception:
                    pass

        dt = (Profile(), Profile(), Profile(), Profile())
        bar = self.pbar

        if isinstance(self.data, dict) and 'names' in self.data:
            model.names = self.data['names']
        elif hasattr(self.model, 'names'):
            model.names = self.model.names
        else:
            LOGGER.warning("Validator could not find 'names'...")
            if hasattr(self, 'nc') and self.nc:
                model.names = {i: f'class_{i}' for i in range(self.nc)}

        self.init_metrics(de_parallel(model))  # Bây giờ model đã có .names
        self.jdict = []  # reset jdict
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)  # augment=self.args.augment

            def _to_float(x):
                if torch.is_tensor(x):
                    return x.float()
                if isinstance(x, (list, tuple)):
                    return [t.float() if torch.is_tensor(t) else t for t in x]
                return x

            preds = _to_float(preds)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Loss
            with dt[2]:
                if self.training and hasattr(self.model, "loss") and callable(self.model.loss):
                    if not getattr(self, "skip_loss", False):
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            self.loss += self.model.loss(batch, preds)[1]
                        except Exception as e:
                            LOGGER.warning(f"[Validator] skip loss this run: {e}")
                            self.skip_loss = True

            # <<< LOG DEBUG VỚI INTERVAL 50 BATCH (Giữ nguyên) >>>
            try:
                val_log_interval = 1
                if self.batch_i % val_log_interval == 0:
                    pred_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds
                    if isinstance(pred_tensor, torch.Tensor):
                        scores_sigmoid = pred_tensor[..., 4:].sigmoid()
                        LOGGER.info(
                            f"[Validator DEBUG] (Batch {self.batch_i}) Raw preds[0] - shape: {pred_tensor.shape}, "
                            f"max score: {scores_sigmoid.max():.4f}, "
                            f"num > 0.01: {(scores_sigmoid > 0.01).sum()}")
                    else:
                        LOGGER.info(
                            f"[Validator DEBUG] (Batch {self.batch_i}) Raw preds type before NMS: {type(preds)}")
            except Exception as e:
                if self.batch_i % val_log_interval == 0:
                    LOGGER.warning(f"[Validator DEBUG] (Batch {self.batch_i}) Error logging raw preds: {e}")
            # <<< KẾT THÚC LOG DEBUG >>>

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = {
            "preprocess": dt[0].dt * 1e3 / len(self.dataloader),
            "inference": dt[1].dt * 1e3 / len(self.dataloader),
            "loss": dt[2].dt * 1e3 / len(self.dataloader),
            "postprocess": dt[3].dt * 1e3 / len(self.dataloader),
        }
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            # lấy loss scalar an toàn
            try:
                loss_scalar = float(self.loss.detach().sum().cpu().item())
            except Exception:
                loss_scalar = float(getattr(self.loss, "sum", lambda: self.loss)())
            # nhét thêm vào stats để trainer không cần fallback
            stats.setdefault("val_loss", loss_scalar)
            # bơm fitness nếu thiếu (ưu tiên mAP50-95)
            stats.setdefault(
                "fitness",
                float(
                    stats.get("metrics/mAP50-95", stats.get("mAP50-95", stats.get("map50-95", 0.0)))
                ),
            )
        return stats

    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """
        Match predictions to ground truth objects using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (N,).
            true_classes (torch.Tensor): Target class indices of shape (M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        """Append the given callback to the specified event."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Run all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        """Build dataset from image path."""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocess an input batch."""
        return batch

    def postprocess(self, preds):
        """Postprocess the predictions."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Update metrics based on predictions and batch."""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """Finalize and return all metrics."""
        pass

    def get_stats(self):
        """Return statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Check statistics."""
        pass

    def print_results(self):
        """Print the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Return the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        # an toàn nếu __init__ bị sửa đổi trong tương lai
        if not hasattr(self, "plots"):
            self.plots = {}
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plot validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plot YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass

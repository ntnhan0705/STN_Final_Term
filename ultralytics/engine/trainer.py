# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo11n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import inspect
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim
from ultralytics.utils.loss import v8DetectionLoss
from types import SimpleNamespace
from ultralytics.utils import LOGGER
from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
)
from torch.utils.data import DataLoader
from ultralytics.utils.stn_pairing import PairIndexBank, PairedBatchSampler, paired_collate_fn
from ultralytics.nn.modules.block import SpatialTransformer

class BaseTrainer:
    """
    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Learning rate scheduler function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (torch.Tensor): Current loss tensor.
        tloss (torch.Tensor): Cumulative loss (averaged per batch).
        loss_names (tuple): Names of loss components.
        csv (Path): Path to results CSV file.
        metrics (dict): Dictionary of validation metrics.
        plots (dict): Dictionary of recorded plots.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the BaseTrainer class."""
        # Load & normalize args
        self.args = get_cfg(cfg, overrides)
        if isinstance(self.args, dict):
            self.args = SimpleNamespace(**self.args)
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                setattr(self.args, k, v)

        # Resume training if applicable
        self.check_resume(overrides)

        # Device selection and seed initialization
        self.device = select_device(self.args.device, self.args.batch)
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Basic holders
        self.validator = None
        self.metrics = None
        self.plots = {}

        # Setup directories for saving results and weights
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name
        self.wdir = self.save_dir / "weights"
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"
        self.save_period = self.args.save_period

        # Training parameters
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Reduce workers on CPU/MPS devices for stability
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0

        # Load model and dataset
        self.model = check_model_file_from_stem(self.args.model)
        with torch_distributed_zero_first(LOCAL_RANK):
            self.trainset, self.testset = self.get_dataset()
        self.ema = None

        # Placeholders for loss function and scheduler
        self.lf = None
        self.scheduler = None

        # Initialize loss names and tracker (`tloss`) based on SupCon flag
        self._supcon_on = int(getattr(self.args, "supcon_on", 0))
        if self._supcon_on == 1:
            self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "supcon_loss")
        else:
            self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        self.tloss = torch.zeros(len(self.loss_names), device=self.device)

        # Logging initialization
        LOGGER.info(self.progress_string())
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2, 3]  # Batches at which to plot training samples (if enabled)

        # Setup callbacks (including integration callbacks for logging, etc.)
        self.hub_session = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Append the given callback to the event's callback list."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Override existing callbacks with the given callback for the specified event."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Main training entrypoint. Handles Single-GPU, Multi-GPU (DDP), or CPU training."""
        # Ensure self.args is SimpleNamespace for attribute-style access
        if isinstance(self.args, dict):
            self.args = SimpleNamespace(**self.args)

        # Determine number of devices for training
        if isinstance(self.args.device, str) and self.args.device:
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:
            world_size = 0
        elif torch.cuda.is_available():
            world_size = 1
        else:
            world_size = 0

        # Launch DDP training if multiple GPUs are requested
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            if self.args.rect:
                LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning("'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting batch=16")
                self.args.batch = 16

            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} Launching Multi-GPU training, command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            # Single GPU or CPU training
            self._do_train(world_size)

    def _setup_scheduler(self):
        """Initialize learning rate scheduler based on linear or cosine strategy."""
        if self.args.cos_lr:
            # Cosine LR schedule from 1.0 to args.lrf (final LR fraction)
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        else:
            # Linear LR schedule from 1.0 to args.lrf
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initialize Distributed Data Parallel (DDP) training environment."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # enforce NCCL timeouts to prevent hangs
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours timeout
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Prepare model, datasets, dataloaders, and optimizers for training."""
        import os, types, math
        from types import SimpleNamespace

        # ---------- callbacks trước train ----------
        self.run_callbacks("on_pretrain_routine_start")

        # ---------- model & device ----------
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # ---------- SupCon: defaults ----------
        _SUPCON_DEFAULTS = dict(
            supcon_on=0,
            supcon_feat=None,
            supcon_warp_gt=0,
            supcon_out=7,
            supcon_min_box=1,
            supcon_max_per_class=0,
            supcon_gain=1.0,
            supcon_loss_weight=None,  # nếu None sẽ lấy supcon_gain khi loss dùng
            supcon_temp=0.2,
            supcon_warmup=0,
            supcon_log=0,
            supcon_use_mem=1,
            supcon_queue=4096,
            supcon_schedule="",  # "0-" | "5-" | "2,4,6-10"
            supcon_proj_dim=0,
            supcon_proj_hidden=0,
            supcon_proj_bn=1,
        )
        _SUPCON_KEYS = tuple(_SUPCON_DEFAULTS.keys())

        def _find_criterion(model):
            """Tìm đúng instance criterion (không quét modules bừa bãi)."""
            for cand in ("criterion", "loss_fcn", "_loss", "det_loss", "loss"):
                obj = getattr(model, cand, None)
                if obj is None:
                    continue
                # bound method -> lấy __self__
                if isinstance(obj, (types.FunctionType, types.MethodType)):
                    obj = getattr(obj, "__self__", obj)
                if obj is None:
                    continue
                if "Loss" in type(obj).__name__ or hasattr(obj, "loss_items"):
                    return obj
            return None

        def _merge_supcon_cfg(trainer):
            """Gộp nguồn theo ưu tiên: _supcon_cfg (cache) -> model.args -> crit.hyp -> trainer.args -> defaults (+ENV)."""
            merged = dict(_SUPCON_DEFAULTS)

            # 1) cache (ưu tiên cao nhất)
            sc = getattr(trainer, "_supcon_cfg", None)
            if sc is not None:
                for k in _SUPCON_KEYS:
                    v = getattr(sc, k, None)
                    if v is not None:
                        merged[k] = v

            # 2) model.args
            ma = getattr(trainer.model, "args", None)
            if isinstance(ma, dict):
                ma = SimpleNamespace(**ma)
            if ma is not None:
                for k in _SUPCON_KEYS:
                    v = getattr(ma, k, None)
                    if v is not None:
                        merged[k] = v

            # 3) crit.hyp
            crit = _find_criterion(trainer.model)
            hyp = getattr(crit, "hyp", None) if crit is not None else None
            if hyp is not None:
                for k in _SUPCON_KEYS:
                    v = getattr(hyp, k, None)
                    if v is not None:
                        merged[k] = v

            # 4) trainer.args (có thể đã bị sanitize)
            ta = getattr(trainer, "args", None)
            if ta is not None:
                for k in _SUPCON_KEYS:
                    v = getattr(ta, k, None)
                    if v is not None:
                        merged[k] = v

            # ENV overrides (tuỳ chọn)
            if str(os.environ.get("SUPCON_FORCE_ON", "0")).strip() == "1":
                merged["supcon_on"] = 1
            _env_sched = os.environ.get("SUPCON_SCHEDULE", "").strip()
            if _env_sched:
                merged["supcon_schedule"] = _env_sched

            return SimpleNamespace(**merged)

        # --- 1) Cache trainer: tôn trọng cache đã được injector đặt trước đó ---
        cache = getattr(self, "_supcon_cfg", None)
        if cache is None or isinstance(cache, dict):
            cache = SimpleNamespace(**(cache or {}))

        # Bù các khóa mặc định vào cache nhưng KHÔNG ghi đè giá trị đã có
        for k, v in _SUPCON_DEFAULTS.items():
            if not hasattr(cache, k):
                setattr(cache, k, v)
        self._supcon_cfg = cache
        # sau khi bù DEFAULTS vào cache, TRƯỚC khi merge từ các nguồn khác
        _is_dict = isinstance(self._supcon_cfg, dict)
        _pre_kv = ", ".join(
            f"{k}={(self._supcon_cfg.get(k) if _is_dict else getattr(self._supcon_cfg, k, None))}"
            for k in _SUPCON_KEYS
        )
        LOGGER.info(f"[SupCon/setup:init-cache] trainer._supcon_cfg(pre-merge) = {{ {_pre_kv} }}")

        # --- 2) Đảm bảo self.model.args là SimpleNamespace ---
        ma = getattr(self.model, "args", None)
        if ma is None or isinstance(ma, dict):
            ma = SimpleNamespace(**(ma or {}))
            self.model.args = ma

        # --- 3) Bù các khóa supcon_* vào model.args từ cache (ưu tiên cache),
        #         chỉ "nâng cấp" nếu hiện tại đang ở giá trị mặc định; không hạ cờ hiện có ---
        for k, default_v in _SUPCON_DEFAULTS.items():
            src_v = getattr(self._supcon_cfg, k, default_v)  # lấy từ cache nếu có
            if not hasattr(ma, k):
                setattr(ma, k, src_v)  # chưa có thì set
            else:
                cur_v = getattr(ma, k)
                # Nếu đang để mặc định và cache có giá trị tốt hơn -> nâng cấp
                if cur_v == default_v and src_v != default_v:
                    setattr(ma, k, src_v)

        # ===== COPY supcon_* từ CLI/inject (self.args) vào model.args & cache (trước sanitize) =====
        ta = getattr(self, "args", None)
        if ta is not None:
            for k in _SUPCON_KEYS:
                if hasattr(ta, k):
                    v = getattr(ta, k)
                    setattr(self.model.args, k, v)  # loss/hyp đọc được
                    setattr(self._supcon_cfg, k, v)  # cache ưu tiên cao

        # nếu model.args đã có supcon_* thì copy ngược lại cache để ưu tiên
        for k in _SUPCON_KEYS:
            v = getattr(self.model.args, k, None)
            if v is not None:
                setattr(self._supcon_cfg, k, v)

        # ---- parser lịch "0-", "5-", "2,4,6-10", ...
        def _parse_supcon_schedule(spec: str):
            spec = (spec or "").strip()
            if not spec:
                return None
            ranges = []
            for tok in spec.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if "-" in tok:
                    a, b = tok.split("-", 1)
                    a, b = a.strip(), b.strip()
                    if a == "" and b == "":
                        continue
                    if a == "":
                        start, end = 0, int(b)
                    elif b == "":
                        start, end = int(a), None
                    else:
                        start, end = int(a), int(b)
                    ranges.append((start, end))
                else:
                    k = int(tok);
                    ranges.append((k, k))

            def _in_schedule(epoch: int) -> bool:
                for (s, e) in ranges:
                    if e is None:
                        if epoch >= s: return True
                    else:
                        if s <= epoch <= e: return True
                return False

            return _in_schedule

        # --- log gọn: cấu hình hiệu lực sau merge
        try:
            cfg = _merge_supcon_cfg(self)
            self._supcon_cfg = cfg
            kv = ", ".join(f"{k}={getattr(cfg, k)}" for k in sorted(_SUPCON_KEYS))
            # thời điểm: SAU khi merge thứ tự ưu tiên
            # cache(_supcon_cfg) -> model.args -> crit.hyp -> trainer.args -> ENV -> DEFAULTS
            cfg = self._supcon_cfg
            _is_dict = isinstance(cfg, dict)
            _merged_kv = ", ".join(
                f"{k}={(cfg.get(k) if _is_dict else getattr(cfg, k, None))}"
                for k in _SUPCON_KEYS
            )
            LOGGER.info(f"[SupCon/setup:merged] trainer._supcon_cfg(effective) = {{ {_merged_kv} }}")


        except Exception as _e:
            LOGGER.warning(f"[SupCon/setup] cache exception: {_e}")
            self._supcon_cfg = SimpleNamespace(**_SUPCON_DEFAULTS)

        # ---- callback 1: quyết định bật/tắt theo epoch
        def _decide_supcon_per_epoch(trainer):
            merged = _merge_supcon_cfg(trainer)
            trainer._supcon_cfg = merged

            checker = _parse_supcon_schedule(getattr(merged, "supcon_schedule", ""))
            e = getattr(trainer, "epoch", 0)

            if checker is not None:
                want_on = bool(checker(e))
            else:
                want_on = bool(getattr(merged, "supcon_on", 0))

            old_on = getattr(merged, "supcon_on", 0)
            merged.supcon_on = int(want_on)
            if old_on != merged.supcon_on:
                LOGGER.info(f"[SupCon/sched] epoch {e}: supcon_on {old_on} -> {merged.supcon_on}")

            # đồng bộ ngay hyp.supcon_on nếu criterion đã sẵn sàng
            crit = _find_criterion(trainer.model)
            if crit is not None:
                if not hasattr(crit, "hyp") or crit.hyp is None:
                    crit.hyp = SimpleNamespace()
                setattr(crit.hyp, "supcon_on", merged.supcon_on)
                setattr(trainer.model.args, "supcon_on", merged.supcon_on)

        # ---- callback 2: attach & sync đầy đủ vào criterion trước khi train batch
        def _attach_and_sync_supcon(trainer):
            crit = _find_criterion(trainer.model)
            if crit is None:
                LOGGER.info("[SupCon/cb] criterion NOT ready; skip this epoch")
                return

            if not hasattr(crit, "hyp") or crit.hyp is None:
                crit.hyp = SimpleNamespace()

            merged = getattr(trainer, "_supcon_cfg", None) or _merge_supcon_cfg(trainer)

            # đồng bộ toàn bộ supcon_* vào crit.hyp + model.args (bù thiếu, không hạ cờ)
            for target in (crit.hyp, trainer.model.args):
                if target is None or isinstance(target, dict):
                    continue
                for k, v in vars(merged).items():
                    if not k.startswith("supcon_"):
                        continue
                    old = getattr(target, k, None)
                    if old is None or (k == "supcon_on" and old in (0, False) and v in (1, True)):
                        setattr(target, k, v)

            # cho loss đọc trainer handle (chỉ supcon_*)
            try:
                crit._trainer = SimpleNamespace(args=merged)
            except Exception:
                pass

            LOGGER.info(
                f"[SupCon/cb] attached -> hyp.on={getattr(crit.hyp, 'supcon_on', None)} "
                f"model.args.on={getattr(trainer.model.args, 'supcon_on', None)} "
                f"crit={type(crit).__name__}"
            )

        # đăng ký: quyết định ON/OFF rồi mới attach & sync
        self.add_callback("on_train_epoch_start", _decide_supcon_per_epoch)
        self.add_callback("on_train_epoch_start", _attach_and_sync_supcon)
        # ép sync 1 lần ngay sau setup để epoch 0 đã có đủ supcon_* trong crit.hyp
        _attach_and_sync_supcon(self)

        # ---------- Freeze layers ----------
        freeze_layers = []
        if isinstance(self.args.freeze, int):
            freeze_layers = [f"model.{i}." for i in range(self.args.freeze)]
        elif isinstance(self.args.freeze, list):
            freeze_layers = [f"model.{i}." for i in self.args.freeze]
        always_freeze = [".dfl"]
        self.freeze_layer_names = freeze_layers + always_freeze
        for name, param in self.model.named_parameters():
            if any(fl in name for fl in self.freeze_layer_names):
                LOGGER.info(f"Freezing layer '{name}'")
                param.requires_grad = False
            elif not param.requires_grad and param.dtype.is_floating_point:
                LOGGER.warning(f"Unfreezing layer '{name}' that was unexpectedly frozen.")
                param.requires_grad = True

        # ---------- AMP ----------
        self.amp = bool(torch.tensor(self.args.amp).to(self.device))
        if self.amp and RANK in {-1, 0}:
            backup_callbacks = callbacks.default_callbacks.copy()
            self.amp = bool(torch.tensor(check_amp(self.model), device=self.device))
            callbacks.default_callbacks = backup_callbacks
        if RANK > -1 and world_size > 1:
            dist.broadcast(torch.tensor(int(self.amp)).to(self.device), src=0)
        self.scaler = (torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4
                       else torch.cuda.amp.GradScaler(enabled=self.amp))
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # ---------- imgsz & stride ----------
        if hasattr(self.model, "stride"):
            try:
                gs = max(int(self.model.stride.max()), 32)
            except Exception:
                gs = max(int(self.model.stride[-1]) if isinstance(self.model.stride, (list, tuple)) else int(
                    self.model.stride), 32)
        else:
            gs = 32
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs

        # ---------- AutoBatch ----------
        if self.batch_size < 1 and RANK == -1:
            self.args.batch = self.batch_size = self.auto_batch()

        # ---------- Sanitize YOLO args (bỏ supcon_* tránh validator) ----------
        try:
            for k in list(vars(self.args).keys()):
                if k.startswith("supcon_"):
                    delattr(self.args, k)
            LOGGER.info("[SupCon/sanitize] stripped supcon_* from self.args for validator")
        except Exception:
            pass

        # ---------- Dataloaders & validator ----------
        batch_size = self.batch_size // max(world_size, 1) or 1
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            val_batch_size = batch_size if self.args.task == "obb" else batch_size * 2
            self.test_loader = self.get_dataloader(self.testset, batch_size=val_batch_size, rank=-1, mode="val")
            self.validator = self.get_validator()
            self.validator.plots = self.plots
            _val_loss = self.label_loss_items(prefix="val")
            _val_keys = list(_val_loss.keys()) if isinstance(_val_loss, dict) else list(_val_loss)
            _metric_keys = list(self.validator.metrics.keys) + _val_keys
            self.metrics = dict.fromkeys(_metric_keys, 0.0)

            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # ---------- Optimizer & Scheduler ----------
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        scaled_wd = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        total_batches = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, 1))
        iterations = total_batches * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=scaled_wd,
            iterations=iterations,
        )
        self._setup_scheduler()
        # --- ghi lại initial_lr để warmup không KeyError ---
        for pg in self.optimizer.param_groups:
            if "initial_lr" not in pg:
                pg["initial_lr"] = pg.get("lr", self.args.lr0)

        self.stopper = EarlyStopping(patience=self.args.patience)
        self.stop = False

        # fix: init best_fitness nếu chưa có
        if not hasattr(self, "best_fitness"):
            self.best_fitness = None

        # ---------- resume & end ----------
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Execute the training loop for the model."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches per epoch
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")

        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for {self.epochs if not self.args.time else f'{self.args.time} hours'}..."
        )

        # If mosaic augmentation is scheduled to close, mark indices for sample plotting
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        # Helper to update loss layout dynamically if number of loss items changes
        self._loss_item_len = len(self.loss_names)
        def _update_loss_layout(n_items: int):
            if n_items == self._loss_item_len:
                return
            self._loss_item_len = n_items
            # Update loss_names and tloss tensor to new size
            if n_items == 4:
                self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "supcon_loss")
            elif n_items == 3:
                self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
            else:
                self.loss_names = tuple(f"loss{i}" for i in range(n_items))
            self.tloss = torch.zeros(n_items, device=self.device)
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())  # print updated header with new loss columns

        epoch = self.start_epoch
        self.optimizer.zero_grad(set_to_none=True)  # clear any residual gradients
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")

            # Update learning rate scheduler for this epoch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress LR scheduler warnings
                self.scheduler.step()

            self._model_train()  # set model to training mode (and handle BatchNorm freezing)
            if RANK != -1:
                # In DDP, set sampler epoch for shuffling
                self.train_loader.sampler.set_epoch(epoch)

            # Apply mosaic closure at specified epoch
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            # Initialize progress bar for this epoch
            batch_iter = enumerate(self.train_loader)
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())  # print epoch header
                batch_iter = TQDM(batch_iter, total=nb)

            self.tloss = None  # reset cumulative loss for this epoch
            for i, batch in batch_iter:
                self.run_callbacks("on_train_batch_start")
                ni = i + nb * epoch  # number of iterations since start

                # Warmup logic: gradually increase LR and momentum from 0 to initial values over warmup period
                if ni <= nw:
                    xi = [0, nw]  # x interpolation points
                    # Compute interpolation scale for learning rate and momentum
                    accum = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    self.accumulate = accum
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Interpolate learning rate for param group j
                        initial_lr = x["initial_lr"] * self.lf(epoch)
                        x["lr"] = np.interp(ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, initial_lr])
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward pass (with autocast if AMP is enabled)
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    loss, self.loss_items = self.model(batch)
                    # Guard nhẹ: nếu loss không phải Tensor có grad thì bỏ qua batch này
                    if not (torch.is_tensor(loss) and loss.requires_grad):
                        LOGGER.warning("[WARN] Model returned a non-differentiable loss. Skipping this batch.")
                        continue

                    # (Optional) soft guard – just warn, don't raise
                    if not (torch.is_tensor(loss) and loss.requires_grad):
                        LOGGER.warning(
                            "[WARN] Loss appears non-differentiable (no grad_fn). "
                            "If this repeats, check that the internal v8DetectionLoss instance is used by the model."
                        )

                # Normalize loss_items to a 1D tensor
                if not isinstance(self.loss_items, torch.Tensor):
                    self.loss_items = torch.as_tensor(self.loss_items, device=self.device, dtype=loss.dtype)
                self.loss_items = self.loss_items.view(-1)  # flatten to 1D (if scalar, becomes shape [1])

                # If SupCon is enabled but no supcon_loss returned, pad a zero for supcon_loss to maintain 4 columns
                if self._supcon_on == 1 and self.loss_items.numel() == 3:
                    self.loss_items = torch.cat([self.loss_items.detach(), torch.zeros(1, device=self.device, dtype=self.loss_items.dtype)], 0)
                    if RANK in {-1, 0}:
                        LOGGER.info("[PadSupCon] appended supcon_loss = 0.0 to loss_items")

                # Update loss layout if number of loss components has changed
                _update_loss_layout(int(self.loss_items.numel()))

                # Total loss for backprop: dùng chính `loss` do model trả về (là Tensor có grad)
                self.loss = loss
                if not (torch.is_tensor(self.loss) and self.loss.requires_grad):
                    LOGGER.warning("[WARN] Loss is not differentiable (no grad_fn). Skipping this batch.")
                    continue  # tránh .backward() bị lỗi

                if RANK != -1:
                    self.loss = self.loss * world_size

                # Update running average of loss components for display (`tloss`)
                if self.tloss is None:
                    self.tloss = self.loss_items.detach().clone()
                else:
                    self.tloss = (self.tloss * i + self.loss_items.detach()) / (i + 1)

                # (Optional) SupCon debug info: show usage stats in progress bar postfix
                if RANK in {-1, 0}:
                    try:
                        crit = getattr(self.model, "loss", None) or getattr(self.model, "criterion", None)
                        if crit is None:
                            # If model wraps loss inside a module, find it
                            for m in self.model.modules():
                                if hasattr(m, "_supcon_stat"):
                                    crit = m
                                    break
                        stat = getattr(crit, "_supcon_stat", None)
                        if stat:
                            used = stat.get("used", "-")
                            roi = int(stat.get("roi", 0))
                            pos = int(stat.get("pos_batch", 0))
                            mem_valid = int(stat.get("mem_valid", 0))
                            postfix_str = f"SC:{used} roi={roi} pos={pos} mem={mem_valid}"
                            batch_iter.set_postfix_str(postfix_str, refresh=False)
                    except Exception:
                        pass

                # Backward pass with gradient scaling (if AMP)
                self.scaler.scale(self.loss).backward()

                # Optimizer step (if we've accumulated enough gradients or at end of epoch)
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed training stop (if --time is set, stop training after specified hours)
                    if self.args.time:
                        elapsed = time.time() - self.train_time_start
                        if elapsed > self.args.time * 3600:
                            self.stop = True
                        # Broadcast stop signal in DDP
                        if RANK != -1:
                            stop_tensor = torch.tensor(int(self.stop)).to(self.device)
                            dist.broadcast(stop_tensor, 0)
                            self.stop = bool(stop_tensor.item())
                        if self.stop:
                            break

                # Update progress bar with current losses and batch info
                if RANK in {-1, 0}:
                    loss_vals = self.tloss.tolist()
                    fmt = ("%11s %11s " + "%11.4g " * len(loss_vals) + "%11s %11s")
                    batch_iter.set_description(fmt % (
                        f"{epoch + 1}/{self.epochs}",
                        f"{self._get_memory():.3g}G",
                        *loss_vals,
                        int(batch["cls"].shape[0]) if isinstance(batch, dict) else "-",  # number of instances
                        int(batch["img"].shape[-1]) if isinstance(batch, dict) else "-"  # image size (assuming square)
                    ))
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
                self.run_callbacks("on_train_batch_end")

                if self.stop:
                    break  # break out of batch loop if stop flag is set
            # --- End of batch loop ---

            # Gather and log metrics at epoch end (on main process)
            self.lr = {f"lr/pg{idx}": grp["lr"] for idx, grp in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")

            if RANK in {-1, 0}:
                final_epoch = (epoch + 1 >= self.epochs)
                # Update EMA (on main process only)
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()

                # Lưu metrics (loss cột động + val metrics + lr) 1 lần
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})

                # Log loss cột (kể cả supcon nếu có)
                loss_values = [float(x) for x in self.tloss]
                LOGGER.info(
                    f"Epoch {epoch + 1}/{self.epochs} - " +
                    ", ".join(f"{name}={val:.4f}" for name, val in zip(self.loss_names, loss_values))
                )

                # Early stop
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch

                # Check for early stopping
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    # If time-based stop is set, adjust total epochs dynamically
                    elapsed_hours = (time.time() - self.train_time_start) / 3600
                    if elapsed_hours > self.args.time:
                        self.stop = True
                    else:
                        # Recompute epochs based on time left and average epoch duration
                        mean_epoch_time = (time.time() - self.train_time_start) / (epoch - self.start_epoch + 1)
                        remaining_epochs = math.ceil((self.args.time * 3600 - (time.time() - self.train_time_start)) / mean_epoch_time)
                        self.epochs = self.args.epochs = epoch + 1 + remaining_epochs
                        self._setup_scheduler()
                        self.scheduler.last_epoch = epoch

                # Save model checkpoints (last and best)
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Cleanup after epoch, prepare for next
            epoch_time_end = time.time()
            self.epoch_time = epoch_time_end - self.epoch_time_start
            self.epoch_time_start = epoch_time_end
            self.run_callbacks("on_fit_epoch_end")
            # Clear GPU memory if usage is high (to avoid fragmentation over long training)
            if self._get_memory(fraction=True) > 0.5:
                self._clear_memory()

            # In DDP, broadcast the stop flag from main process to all processes
            if RANK != -1:
                stop_tensor = torch.tensor(int(self.stop)).to(self.device)
                dist.broadcast(stop_tensor, src=0)
                self.stop = bool(stop_tensor.item())

            if self.stop:
                break  # early break out of epoch loop
            epoch += 1
        # --- End of epoch loop ---

        if RANK in {-1, 0}:
            total_time = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {total_time / 3600:.3f} hours.")
            # Final evaluation on best model
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")

        # Cleanup
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )

    def _get_memory(self, fraction=False):
        """Get GPU/MPS memory utilization in GB or as a fraction of total."""
        if self.device.type == "mps":
            mem_used = torch.mps.driver_allocated_memory()
            return (__import__("psutil").virtual_memory().percent / 100) if fraction else (mem_used / 2**30)
        if self.device.type == "cpu":
            return 0.0 if fraction else 0.0
        # CUDA device
        mem_used = torch.cuda.memory_reserved(self.device)
        if fraction:
            total = torch.cuda.get_device_properties(self.device).total_memory
            return mem_used / total if total > 0 else 0.0
        return mem_used / (2**30)

    def _clear_memory(self):
        """Release unused memory caches."""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
        # (CPU has no cache to clear)

    def read_results_csv(self):
        """Read results.csv into a dictionary."""
        import pandas as pd
        return pd.read_csv(self.csv).to_dict(orient="list")

    def _model_train(self):
        """Set model to training mode and freeze batch norm layers if specified."""
        self.model.train()
        for name, module in self.model.named_modules():
            if any(frozen in name for frozen in getattr(self, "freeze_layer_names", [])):
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()  # keep batch norm in eval mode if its layer is frozen

    def save_model(self):
        """Save model checkpoint (last and best weights)."""
        import io
        buffer = io.BytesIO()
        torch.save({
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": None,  # model architecture is known and weights are EMA below
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            "train_args": vars(self.args),
            "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
            "train_results": self.read_results_csv(),
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }, buffer)
        ckpt_bytes = buffer.getvalue()
        # Save last checkpoint
        self.last.write_bytes(ckpt_bytes)
        # Save best checkpoint if improved
        if self.best_fitness == self.fitness:
            self.best.write_bytes(ckpt_bytes)
        # Save periodic checkpoint if save_period is set
        if self.save_period and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(ckpt_bytes)

    def get_dataset(self):
        """Load dataset (train and val) based on self.args.data configuration."""
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            else:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # use actual yaml file path
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        self.data = data
        if self.args.single_cls:
            LOGGER.info("Overriding dataset class names with single class 'item'.")
            self.data["names"] = {0: "item"}
            self.data["nc"] = 1
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """
        Load or create the model based on self.args.model configuration.
        Returns:
            ckpt (dict or None): Checkpoint if resuming from a saved model, else None.
        """
        if isinstance(self.model, torch.nn.Module):
            # Model is already loaded (no further setup needed)
            return
        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml  # model configuration from checkpoint
        elif isinstance(self.args.pretrained, (str, Path)):
            # Load weights from a pretrained model path if provided
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        # Instantiate model (calls underlying Model class with cfg and weights)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=(RANK == -1))
        # Attach loss
        self.loss = v8DetectionLoss(self.model, self.args)

        # >>> THÊM: để các callback SupCon tìm thấy criterion ở trong model
        self.model.criterion = self.loss
        self.model.loss = self.loss

        return ckpt

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)

        # --- ADD: scale + clip riêng cho STN, vẫn dùng 1 LR ---
        stn_grad_mult = float(getattr(self.args, "stn_grad_mult", 0.2))  # hệ số < 1 => update nhẹ
        stn_params = []
        for m in self.model.modules():
            if isinstance(m, SpatialTransformer):
                for p in m.parameters():
                    if p.grad is not None:
                        p.grad.mul_(stn_grad_mult)  # scale grad tại chỗ
                        stn_params.append(p)

        # clip “trust region” riêng cho STN (nhỏ hơn clip chung)
        if stn_params:
            torch.nn.utils.clip_grad_norm_(stn_params, max_norm=0.5)

        # clip chung toàn model (nếu đang dùng)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Custom preprocessing of batch data (if needed for different tasks)."""
        return batch

    def validate(self):
        """
        Run validation using the validator.
        Returns:
            metrics (dict): Validation metrics.
            fitness (float): Fitness score (higher is better).
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -float(self.loss.detach().cpu().numpy()))  # default fitness is negative loss if not provided
        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Must be implemented by subclasses to return a model instance."""
        raise NotImplementedError("This trainer does not implement get_model for cfg files.")

    def get_validator(self):
        """Must be implemented by subclasses to return a validator instance."""
        raise NotImplementedError("get_validator function not implemented in BaseTrainer.")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataset = self.build_dataset(dataset_path, mode=mode)  # như hiện tại bạn dùng
        base_collate = getattr(dataset, "collate_fn", None)

        if getattr(self.args, "pairing", False) and mode == "train":
            bank = PairIndexBank(self.args.data,
                                 getattr(self.args, "bgpair_map", "bgpair_map.json"),
                                 seed=getattr(self.args, "seed", 0))
            world_size = getattr(self, "world_size", 1)
            rank = getattr(self, "rank", rank)
            batch_sampler = PairedBatchSampler(bank, batch_size=batch_size, drop_last=False,
                                               seed=getattr(self.args, "seed", 0),
                                               rank=rank, world_size=world_size)
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=getattr(self.args, "workers", 4),
                pin_memory=True,
                collate_fn=paired_collate_fn(base_collate)
            )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == "train"),
            num_workers=getattr(self.args, "workers", 4),
            pin_memory=True,
            collate_fn=base_collate
        )

    def build_dataset(self, img_path, mode="train", batch=None):
        """Optional: build a dataset given image paths (for tasks like classification)."""
        raise NotImplementedError("build_dataset function not implemented in BaseTrainer.")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Create a dictionary labeling each component of the loss for logging.
        Handles 3 components (box/cls/dfl) or 4 components (with supcon_loss).
        """
        if loss_items is None:
            return {f"{prefix}/loss": 0.0}
        li = loss_items.detach().float().cpu()
        n = int(li.numel())
        if n == 4:
            keys = ("box_loss", "cls_loss", "dfl_loss", "supcon_loss")
        elif n == 3:
            keys = ("box_loss", "cls_loss", "dfl_loss")
        else:
            keys = tuple(f"loss{i}" for i in range(n))
        return {f"{prefix}/{k}": v for k, v in zip(keys, li.tolist())}

    def set_model_attributes(self):
        """Set additional attributes on the model (e.g., class names)."""
        if "names" in self.data:
            self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Build target tensors for training (task-specific, not needed for base class)."""
        pass

    def progress_string(self):
        """Return a formatted header string for training progress display."""
        columns = ("Epoch", "GPU_mem") + self.loss_names + ("Instances", "Size")
        fmt = "\n" + " ".join("%11s" for _ in columns)
        return fmt % columns

    def plot_training_samples(self, batch, ni):
        """Plot training batch images and labels (optional, to be implemented if needed)."""
        pass

    def plot_training_labels(self):
        """Plot distribution of training labels (optional, to be implemented if needed)."""
        pass

    def save_metrics(self, metrics):
        """Append training/validation metrics to the results CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        # Write header if file is new
        if not self.csv.exists():
            header = ",".join(["epoch", "time"] + keys) + "\n"
        else:
            header = ""
        t = time.time() - self.train_time_start
        row = ",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in [self.epoch + 1, t] + vals) + "\n"
        with open(self.csv, "a", encoding="utf-8") as f:
            if header:
                f.write(header)
            f.write(row)

    def plot_metrics(self):
        """Plot training metrics from the results CSV (optional, to be implemented)."""
        pass

    def on_plot(self, name, data=None):
        """Record a plot (e.g., for use in callbacks or integrations)."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Evaluate the final model (best.pt) using the validator and log results."""
        ckpt = {}
        for f in (self.last, self.best):
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)  # load last.pt
                elif f is self.best:
                    # Update best.pt with training results if available
                    strip_optimizer(f, updates={"train_results": ckpt.get("train_results")} if "train_results" in ckpt else None)
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """If resume is requested, load resume checkpoint and update self.args accordingly."""
        resume_flag = self.args.resume
        self.resume = False
        if resume_flag:
            try:
                # Determine checkpoint path
                exists = isinstance(resume_flag, (str, Path)) and Path(resume_flag).exists()
                last_ckpt = Path(check_file(resume_flag) if exists else get_latest_run())
                ckpt_args = attempt_load_weights(last_ckpt).args
                # If original data path is missing, override with current data path
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data
                # Merge checkpoint args with current overrides
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last_ckpt)
                for k in ("imgsz", "batch", "device", "close_mosaic"):
                    if k in overrides:
                        setattr(self.args, k, overrides[k])
                self.resume = True
            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Provide a valid checkpoint path, e.g., 'yolo train resume=model.pt'"
                ) from e

    def resume_training(self, ckpt):
        """Resume training from a checkpoint (ckpt)."""
        if not ckpt or not self.resume:
            return
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt.get("updates", self.ema.updates)
        if start_epoch <= 0:
            raise AssertionError(
                f"{self.args.model} has already been trained for {ckpt['epoch']} epochs. Nothing to resume."
            )
        LOGGER.info(f"Resuming training from epoch {start_epoch} of {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(f"Extending training to {self.epochs + ckpt['epoch']} total epochs")
            self.epochs += ckpt["epoch"]  # extend total epochs for fine-tuning
        self.best_fitness = ckpt.get("best_fitness", 0.0)
        self.start_epoch = start_epoch
        # If resuming past mosaic phase, disable mosaic augmentation
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Disable mosaic augmentation in the dataset (if applicable)."""
        ds = getattr(self.train_loader, "dataset", None)
        if hasattr(ds, "mosaic"):
            ds.mosaic = False
        if hasattr(ds, "close_mosaic"):
            LOGGER.info("Closing mosaic augmentation for subsequent epochs")
            ds.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Construct an optimizer with separate parameter groups for biases, BatchNorm, and other weights.
        If name='auto', automatically choose between SGD and AdamW based on iterations.
        """
        # Parameter groups: g0 = weights with decay, g1 = weights without decay (BN, etc.), g2 = biases without decay
        g = ([], [], [])
        norm_classes = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        if name.lower() == "auto":
            LOGGER.info(f"{colorstr('optimizer:')} auto-configuring optimizer, LR and momentum based on dataset size.")
            nc = self.data.get("nc", 10)
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            if iterations > 10000:
                name, lr, momentum = "SGD", 0.01, 0.9
            else:
                name, lr, momentum = "AdamW", lr_fit, 0.9
            # Adjust warmup bias LR for Adam optimizers
            self.args.warmup_bias_lr = 0.0

        name = name.lower()
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    g[2].append(param)  # biases
                elif isinstance(module, norm_classes) or "logit_scale" in fullname:
                    g[1].append(param)  # batch norm or similar (no decay)
                else:
                    g[0].append(param)  # weights with decay

        # Instantiate optimizer
        if name in {"adam", "adamax", "adamw", "nadam", "radam"}:
            optim_class = getattr(optim, name.capitalize(), optim.Adam)
            optimizer = optim_class(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "rmsprop":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "sgd":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f"Optimizer '{name}' not recognized. Available: SGD, AdamW, etc.")

        # Add parameter groups with appropriate weight decay
        optimizer.add_param_group({"params": g[0], "weight_decay": decay})
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with "
            f"{len(g[0])} params (weight_decay={decay}), {len(g[1])} params (no decay), {len(g[2])} params (bias no decay)"
        )
        return optimizer

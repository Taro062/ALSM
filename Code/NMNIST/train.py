from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# NumPy 2.x compatibility for older SpikingJelly/CuPy kernels.
_NUMPY_COMPAT_ALIASES = {
    "int": int,
    "float": float,
    "complex": complex,
    "bool": bool,
    "object": object,
}
for _name, _value in _NUMPY_COMPAT_ALIASES.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import tonic.transforms as transforms
from tonic import DiskCachedDataset

try:
    from spikingjelly.activation_based import functional as sj_functional
except ImportError:
    from spikingjelly.clock_driven import functional as sj_functional

try:
    from spikformer7 import ALSMNMNIST
except ImportError as exc:
    raise ImportError("请确保修改后的 spiklsmAG.py 与训练代码位于同一目录") from exc


# ==============================================================================
# 1. 原 N-MNIST 本地数据读取逻辑：保持不变
# ==============================================================================
class ToFloat32:
    def __call__(self, x):
        return x.astype(np.float32)


class LocalNMNIST(Dataset):

    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = "Train" if train else "Test"
        self.data_dir = os.path.join(root_dir, self.split)
        self.samples = []

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"❌ 找不到数据文件夹: {self.data_dir}\n"
                f"请确保你的目录结构是: {root_dir}/Train 和 {root_dir}/Test"
            )

        print(f"🔍 Scanning files in {self.data_dir}...")
        for digit in range(10):
            digit_dir = os.path.join(self.data_dir, str(digit))
            if os.path.exists(digit_dir):
                files = [
                    os.path.join(digit_dir, name)
                    for name in os.listdir(digit_dir)
                    if name.endswith(".bin")
                ]
                for file_path in files:
                    self.samples.append((file_path, digit))

        print(f"✅ Loaded {len(self.samples)} samples from {self.split} set.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        with open(file_path, "rb") as file:
            raw = np.fromfile(file, dtype=np.uint8)

        x = raw[0::5]
        y = raw[1::5]
        data_p_ts = raw[2::5]
        data_ts_mid = raw[3::5]
        data_ts_low = raw[4::5]

        p = (data_p_ts >> 7).astype(np.int8)
        t = (
            ((data_p_ts & 127).astype(np.int64) << 16)
            | (data_ts_mid.astype(np.int64) << 8)
            | data_ts_low.astype(np.int64)
        )

        events = np.zeros(
            len(x),
            dtype=[("x", "<i2"), ("y", "<i2"), ("t", "<i8"), ("p", "<i2")],
        )
        events["x"] = x
        events["y"] = y
        events["t"] = t
        events["p"] = p

        if self.transform:
            events = self.transform(events)

        return events, label


# ==============================================================================
# 2. 参数设置
# ==============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Improved ALSM Training for Local N-MNIST"
    )
    parser.add_argument("-f", "--file", default="file")

    parser.add_argument(
        "--data-path",
        default="D:/PythonProject/data/N-MNIST",
        help="Path containing Train/Test folders",
    )
    parser.add_argument(
        "--output-dir",
        default="./logs_nmnist",
        help="path where to save",
    )
    parser.add_argument(
        "--cache-dir",
        default="./cache_nmnist",
        help="Tonic disk cache directory",
    )

    parser.add_argument("--seed", default=42, type=int)

    # Model: preserve the N-MNIST comparison configuration.
    parser.add_argument("--T", default=1, type=int)
    parser.add_argument("--embed-dim", default=128, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--lsm-ratio", default=4.0, type=float)
    parser.add_argument("--drop-path", default=0.10, type=float)
    parser.add_argument("--attn-drop", default=0.0, type=float)
    parser.add_argument("--gate-dropout", default=0.0, type=float)
    parser.add_argument("--head-dropout", default=0.10, type=float)
    parser.add_argument("--attn-scale", default=0.125, type=float)
    parser.add_argument(
        "--backend",
        choices=("auto", "cupy", "torch"),
        default="auto",
    )

    # Optimization.
    parser.add_argument("--batch-size", default=192, type=int)
    parser.add_argument("--eval-batch-size", default=256, type=int)
    parser.add_argument("--accum-steps", default=1, type=int)

    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--warmup-epochs", default=5, type=int)

    # 优化器
    parser.add_argument("--lr", default=8e-4, type=float)
    parser.add_argument("--min-lr", default=1e-6, type=float)
    parser.add_argument("--weight-decay", default=0.02, type=float)
    parser.add_argument("--clip-grad", default=1.0, type=float)

    parser.add_argument(
        "--temporal-aux-weight",
        default=0.05,
        type=float,
    )

    # Mild event Mixup/CutMix;
    parser.add_argument("--mixup", default=0.10, type=float)
    parser.add_argument("--cutmix", default=0.0, type=float)
    parser.add_argument("--mix-prob", default=0.20, type=float)
    parser.add_argument("--mix-switch-prob", default=0.50, type=float)
    parser.add_argument("--smoothing", default=0.05, type=float)

    # 最后 30 epoch 使用干净数据
    parser.add_argument(
        "--mixup-off-epoch",
        default=70,
        type=int,
    )

    # 减弱事件增强
    parser.add_argument("--spatial-shift", default=2, type=int)
    parser.add_argument(
        "--spatial-shift-prob",
        default=0.35,
        type=float,
    )

    parser.add_argument("--temporal-shift", default=1, type=int)
    parser.add_argument(
        "--temporal-shift-prob",
        default=0.10,
        type=float,
    )

    parser.add_argument(
        "--frame-drop-prob",
        default=0.03,
        type=float,
    )
    parser.add_argument(
        "--polarity-drop-prob",
        default=0.005,
        type=float,
    )
    parser.add_argument(
        "--erase-prob",
        default=0.03,
        type=float,
    )

    # Runtime.
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--prefetch-factor", default=2, type=int)
    parser.add_argument("--print-freq", default=50, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--deterministic", action="store_true")
    return parser


def parse_args():
    return build_parser().parse_args()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(args):
    sensor_size = (34, 34, 2)
    target_size = (32, 32, 2)

    frame_transform = transforms.Compose(
        [
            transforms.CenterCrop(
                sensor_size=sensor_size,
                size=(32, 32),
            ),
            transforms.ToFrame(
                sensor_size=target_size,
                n_time_bins=args.T,
            ),
            ToFloat32(),
        ]
    )

    full_dataset_train = LocalNMNIST(
        root_dir=args.data_path,
        train=True,
        transform=frame_transform,
    )
    full_dataset_test = LocalNMNIST(
        root_dir=args.data_path,
        train=False,
        transform=frame_transform,
    )

    cached_trainset = DiskCachedDataset(
        full_dataset_train,
        cache_path=os.path.join(args.cache_dir, "train"),
    )
    cached_testset = DiskCachedDataset(
        full_dataset_test,
        cache_path=os.path.join(args.cache_dir, "test"),
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        cached_trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        cached_testset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    print(
        f"Train samples: {len(cached_trainset)}, "
        f"Test samples: {len(cached_testset)}, T={args.T}"
    )
    return train_loader, test_loader


# ==============================================================================
# 4. Event augmentation and Mixup/CutMix
# ==============================================================================
class NMNISTEventAugment(nn.Module):
    """Mild sequence-consistent augmentation applied after loading frames."""

    def __init__(self, args) -> None:
        super().__init__()
        self.spatial_shift = args.spatial_shift
        self.spatial_shift_prob = args.spatial_shift_prob
        self.temporal_shift = args.temporal_shift
        self.temporal_shift_prob = args.temporal_shift_prob
        self.frame_drop_prob = args.frame_drop_prob
        self.polarity_drop_prob = args.polarity_drop_prob
        self.erase_prob = args.erase_prob

    @staticmethod
    def _translate_zero_fill(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        if dy == 0 and dx == 0:
            return x
        shifted = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        if dy > 0:
            shifted[..., :dy, :] = 0
        elif dy < 0:
            shifted[..., dy:, :] = 0
        if dx > 0:
            shifted[..., :, :dx] = 0
        elif dx < 0:
            shifted[..., :, dx:] = 0
        return shifted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        # x: [B,T,C,H,W]
        if self.spatial_shift > 0 and torch.rand((), device=x.device) < self.spatial_shift_prob:
            dy = int(
                torch.randint(
                    -self.spatial_shift,
                    self.spatial_shift + 1,
                    (1,),
                    device=x.device,
                ).item()
            )
            dx = int(
                torch.randint(
                    -self.spatial_shift,
                    self.spatial_shift + 1,
                    (1,),
                    device=x.device,
                ).item()
            )
            x = self._translate_zero_fill(x, dy, dx)

        if self.temporal_shift > 0 and torch.rand((), device=x.device) < self.temporal_shift_prob:
            shift = int(
                torch.randint(
                    -self.temporal_shift,
                    self.temporal_shift + 1,
                    (1,),
                    device=x.device,
                ).item()
            )
            if shift > 0:
                shifted = torch.zeros_like(x)
                shifted[:, shift:] = x[:, :-shift]
                x = shifted
            elif shift < 0:
                shifted = torch.zeros_like(x)
                shifted[:, :shift] = x[:, -shift:]
                x = shifted

        if x.shape[1] > 2 and torch.rand((), device=x.device) < self.frame_drop_prob:
            frame = int(torch.randint(0, x.shape[1], (1,), device=x.device).item())
            x[:, frame] = 0

        if x.shape[2] == 2 and torch.rand((), device=x.device) < self.polarity_drop_prob:
            polarity = int(torch.randint(0, 2, (1,), device=x.device).item())
            x[:, :, polarity] = 0

        if torch.rand((), device=x.device) < self.erase_prob:
            height, width = x.shape[-2:]
            erase_h = int(torch.randint(2, max(3, height // 5 + 1), (1,), device=x.device).item())
            erase_w = int(torch.randint(2, max(3, width // 5 + 1), (1,), device=x.device).item())
            top = int(torch.randint(0, height - erase_h + 1, (1,), device=x.device).item())
            left = int(torch.randint(0, width - erase_w + 1, (1,), device=x.device).item())
            x[..., top : top + erase_h, left : left + erase_w] = 0

        return x.contiguous()


def smooth_one_hot(
    targets: torch.Tensor,
    num_classes: int,
    smoothing: float,
) -> torch.Tensor:
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    soft = torch.full(
        (targets.shape[0], num_classes),
        off,
        device=targets.device,
        dtype=torch.float32,
    )
    return soft.scatter_(1, targets[:, None], on)


class EventMixupCutMix:
    """Mix complete event samples; one CutMix box is shared over all time bins."""

    def __init__(
        self,
        *,
        num_classes: int,
        mixup_alpha: float,
        cutmix_alpha: float,
        probability: float,
        switch_probability: float,
        label_smoothing: float,
    ) -> None:
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.probability = probability
        self.switch_probability = switch_probability
        self.label_smoothing = label_smoothing
        self.enabled = True

    @staticmethod
    def _sample_beta(alpha: float) -> float:
        if alpha <= 0:
            return 1.0
        return float(np.random.beta(alpha, alpha))

    @staticmethod
    def _cutmix_box(height: int, width: int, lam: float) -> Tuple[int, int, int, int]:
        ratio = math.sqrt(max(0.0, 1.0 - lam))
        cut_h = max(1, int(height * ratio))
        cut_w = max(1, int(width * ratio))
        cy = int(torch.randint(0, height, (1,)).item())
        cx = int(torch.randint(0, width, (1,)).item())
        y1 = max(0, cy - cut_h // 2)
        y2 = min(height, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(width, cx + cut_w // 2)
        return y1, y2, x1, x2

    def __call__(
        self,
        samples: torch.Tensor,
        hard_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        soft_targets = smooth_one_hot(
            hard_targets,
            self.num_classes,
            self.label_smoothing,
        )

        if (
            not self.enabled
            or self.probability <= 0
            or torch.rand((), device=samples.device) > self.probability
        ):
            return samples, soft_targets

        permutation = torch.randperm(samples.shape[0], device=samples.device)
        use_cutmix = (
            self.cutmix_alpha > 0
            and (
                self.mixup_alpha <= 0
                or torch.rand((), device=samples.device) < self.switch_probability
            )
        )

        if use_cutmix:
            lam = self._sample_beta(self.cutmix_alpha)
            y1, y2, x1, x2 = self._cutmix_box(
                samples.shape[-2],
                samples.shape[-1],
                lam,
            )
            mixed = samples.clone()
            mixed[..., y1:y2, x1:x2] = samples[
                permutation, ..., y1:y2, x1:x2
            ]
            area = (y2 - y1) * (x2 - x1)
            lam = 1.0 - area / (samples.shape[-2] * samples.shape[-1])
        else:
            lam = self._sample_beta(self.mixup_alpha)
            mixed = samples * lam + samples[permutation] * (1.0 - lam)

        mixed_targets = (
            soft_targets * lam + soft_targets[permutation] * (1.0 - lam)
        )
        return mixed, mixed_targets


# ==============================================================================
# 5. Loss, optimizer and schedule
# ==============================================================================
def soft_target_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def temporal_classification_loss(
    temporal_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    *,
    temporal_aux_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_logits = temporal_logits.mean(dim=0)
    main_loss = soft_target_cross_entropy(mean_logits, soft_targets)

    if temporal_aux_weight <= 0:
        return main_loss, mean_logits

    t, b, c = temporal_logits.shape
    repeated_targets = soft_targets.unsqueeze(0).expand(t, -1, -1)
    auxiliary_loss = soft_target_cross_entropy(
        temporal_logits.reshape(t * b, c),
        repeated_targets.reshape(t * b, c),
    )
    return main_loss + temporal_aux_weight * auxiliary_loss, mean_logits


def no_weight_decay(name: str, parameter: torch.Tensor) -> bool:
    return (
        parameter.ndim <= 1
        or name.endswith(".bias")
        or "pos_embed" in name
        or "raw_rec_gain" in name
        or "layer_scale" in name
    )


def parameter_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if no_weight_decay(name, parameter) else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> float:
    if epoch < warmup_epochs:
        progress = (epoch + 1) / max(1, warmup_epochs)
        learning_rate = min_lr + progress * (base_lr - min_lr)
    else:
        progress = (epoch - warmup_epochs) / max(
            1,
            total_epochs - warmup_epochs - 1,
        )
        learning_rate = min_lr + 0.5 * (base_lr - min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def resolve_backend(value: str) -> str:
    if value in {"torch", "cupy"}:
        return value
    try:
        import cupy  # noqa: F401

        return "cupy"
    except Exception:
        return "torch"


def reset_net(model: nn.Module) -> None:
    sj_functional.reset_net(model)


@torch.no_grad()
def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Sequence[int] = (1, 5),
):
    max_k = min(max(topk), logits.shape[-1])
    predictions = logits.topk(max_k, dim=1).indices.t()
    correct = predictions.eq(targets.view(1, -1))
    results = []
    for k in topk:
        k = min(k, logits.shape[-1])
        results.append(
            correct[:k].reshape(-1).float().sum()
            * (100.0 / targets.numel())
        )
    return results


# ==============================================================================
# 6. Train and raw-only evaluation
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    mix_operator: EventMixupCutMix,
    augmentor: NMNISTEventAugment,
    device: torch.device,
    epoch: int,
    args,
) -> Dict[str, float]:
    model.train()
    augmentor.train()
    optimizer.zero_grad(set_to_none=True)
    mix_operator.enabled = epoch < args.mixup_off_epoch

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    updates = 0
    start_time = time.time()

    for step, (samples, hard_targets) in enumerate(loader):
        samples = samples.to(device, non_blocking=True).float()
        hard_targets = hard_targets.to(device, non_blocking=True).long()
        samples = augmentor(samples)
        samples, soft_targets = mix_operator(samples, hard_targets)

        remainder = len(loader) % args.accum_steps
        final_window_start = len(loader) - remainder if remainder else len(loader)
        divisor = (
            remainder
            if remainder and step >= final_window_start
            else args.accum_steps
        )

        with torch.amp.autocast(device_type="cuda", enabled=args.amp):
            temporal_logits = model(samples)
            loss, mean_logits = temporal_classification_loss(
                temporal_logits,
                soft_targets,
                temporal_aux_weight=args.temporal_aux_weight,
            )
            scaled_loss = loss / divisor

        scaler.scale(scaled_loss).backward()
        reset_net(model)

        should_update = (
            (step + 1) % args.accum_steps == 0
            or step + 1 == len(loader)
        )
        if should_update:
            scaler.unscale_(optimizer)
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1

        batch_size = samples.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_correct += int(
            mean_logits.detach().argmax(dim=1).eq(hard_targets).sum()
        )
        total_samples += batch_size

        if (step + 1) % args.print_freq == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"epoch={epoch:03d} step={step + 1:04d}/{len(loader):04d} "
                f"loss={total_loss / total_samples:.4f} "
                f"train_acc={100.0 * total_correct / total_samples:.2f} "
                f"samples/s={total_samples / elapsed:.1f}"
            )

    return {
        "loss": total_loss / max(1, total_samples),
        "acc1": 100.0 * total_correct / max(1, total_samples),
        "updates": updates,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct_1 = 0.0
    correct_5 = 0.0
    total_samples = 0

    for samples, targets in loader:
        samples = samples.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).long()

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            temporal_logits = model(samples)
            logits = temporal_logits.mean(dim=0)
            loss = F.cross_entropy(logits, targets)

        reset_net(model)
        acc1, acc5 = topk_accuracy(logits, targets, (1, 5))
        batch_size = samples.shape[0]
        total_loss += float(loss) * batch_size
        correct_1 += float(acc1) * batch_size / 100.0
        correct_5 += float(acc5) * batch_size / 100.0
        total_samples += batch_size

    return {
        "loss": total_loss / max(1, total_samples),
        "acc1": 100.0 * correct_1 / max(1, total_samples),
        "acc5": 100.0 * correct_5 / max(1, total_samples),
    }


# ==============================================================================
# 7. Raw checkpoint logic
# ==============================================================================
def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    best_raw_acc: float,
    current_raw_acc: float,
    checkpoint_metric: str,
    args,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "best_raw_acc": float(best_raw_acc),
            "current_raw_acc": float(current_raw_acc),
            "checkpoint_metric": checkpoint_metric,
            "best_acc": float(best_raw_acc),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler=None,
):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


# ==============================================================================
# 8. Main
# ==============================================================================
def main(args) -> None:
    if os.path.exists(args.cache_dir):
        print(
            f"⚠️ 警告: 如果修改了 T 或帧转换设置后遇到缓存错误，"
            f"请手动删除缓存目录: {args.cache_dir}"
        )

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(
            f"N-MNIST 数据集路径不存在: {args.data_path}\n"
            "请保持 Train/Test 目录结构，并确认 --data-path 设置。"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cudnn.deterministic = args.deterministic

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the recommended configuration.")

    device = torch.device(args.device)
    args.backend = resolve_backend(args.backend)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    print("🚀 Start Training Improved ALSM on N-MNIST")
    print(f"   Data Path: {args.data_path}")
    print(f"   Cache Directory: {args.cache_dir}")
    print(f"   NumPy: {np.__version__}")
    print(f"   Spiking backend: {args.backend}")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    train_loader, test_loader = build_loaders(args)

    model = ALSMNMNIST(
        img_size=32,
        in_channels=2,
        num_classes=10,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        lsm_ratio=args.lsm_ratio,
        drop_path_rate=args.drop_path,
        attn_drop=args.attn_drop,
        gate_dropout=args.gate_dropout,
        head_dropout=args.head_dropout,
        attn_scale=args.attn_scale,
        backend=args.backend,
        T=args.T,
    ).to(device)

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {trainable / 1e6:.3f} M")

    optimizer = torch.optim.AdamW(
        parameter_groups(model, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    mix_operator = EventMixupCutMix(
        num_classes=10,
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        probability=args.mix_prob,
        switch_probability=args.mix_switch_prob,
        label_smoothing=args.smoothing,
    )
    augmentor = NMNISTEventAugment(args).to(device)

    start_epoch = 0
    best_raw_acc = 0.0
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            optimizer=None if args.eval else optimizer,
            scaler=None if args.eval else scaler,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_raw_acc = float(
            checkpoint.get(
                "best_raw_acc",
                checkpoint.get(
                    "current_raw_acc",
                    checkpoint.get("best_acc", checkpoint.get("acc", 0.0)),
                ),
            )
        )
        print(
            f"Loaded {args.resume}; start_epoch={start_epoch}, "
            f"best_raw={best_raw_acc:.3f}"
        )

    if args.eval:
        raw_stats = evaluate(
            model,
            test_loader,
            device,
            amp_enabled=args.amp,
        )
        print(f"Raw evaluation: {raw_stats}")
        return

    writer = SummaryWriter(log_dir=args.output_dir)
    log_path = output_dir / "log.jsonl"
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        learning_rate = set_learning_rate(
            optimizer,
            epoch=epoch,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            base_lr=args.lr,
            min_lr=args.min_lr,
        )

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            mix_operator,
            augmentor,
            device,
            epoch,
            args,
        )
        raw_stats = evaluate(
            model,
            test_loader,
            device,
            amp_enabled=args.amp,
        )

        raw_improved = raw_stats["acc1"] > best_raw_acc
        if raw_improved:
            best_raw_acc = raw_stats["acc1"]

        record = {
            "epoch": epoch,
            "lr": learning_rate,
            "train": train_stats,
            "raw_eval": raw_stats,
            "best_raw_acc1": best_raw_acc,
            "best_acc1": best_raw_acc,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        writer.add_scalar("Train/Loss", train_stats["loss"], epoch)
        writer.add_scalar("Train/Acc1", train_stats["acc1"], epoch)
        writer.add_scalar("RawEval/Loss", raw_stats["loss"], epoch)
        writer.add_scalar("RawEval/Acc1", raw_stats["acc1"], epoch)
        writer.add_scalar("RawEval/Acc5", raw_stats["acc5"], epoch)
        writer.add_scalar("Train/LR", learning_rate, epoch)

        save_checkpoint(
            output_dir / "checkpoint_latest.pth",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_raw_acc=best_raw_acc,
            current_raw_acc=raw_stats["acc1"],
            checkpoint_metric="latest",
            args=args,
        )

        if raw_improved:
            save_checkpoint(
                output_dir / "checkpoint_best_raw.pth",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_raw_acc=best_raw_acc,
                current_raw_acc=raw_stats["acc1"],
                checkpoint_metric="raw_eval",
                args=args,
            )
            print(
                f"🌟 Saved checkpoint_best_raw.pth "
                f"(raw Acc@1={best_raw_acc:.2f})"
            )

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}: "
            f"lr={learning_rate:.3e}, "
            f"train_loss={train_stats['loss']:.4f}, "
            f"raw_acc={raw_stats['acc1']:.2f}, "
            f"best_raw={best_raw_acc:.2f}"
        )

    elapsed = str(
        datetime.timedelta(seconds=int(time.time() - training_start))
    )
    print(
        f"Training completed. Best raw Acc@1={best_raw_acc:.3f}; "
        f"time={elapsed}"
    )
    writer.close()


if __name__ == "__main__":
    main(parse_args())

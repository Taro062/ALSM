from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

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
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
import sys

sys.path.append(r'/root/LSM/model')
from model import ALSMCIFAR10DVS

try:
    from spikingjelly.activation_based import functional as sj_functional
except ImportError:
    from spikingjelly.clock_driven import functional as sj_functional

from spikingjelly.datasets import cifar10_dvs

try:
    from spikingjelly.datasets import split_to_train_test_set
except ImportError:
    split_to_train_test_set = None


class OfflineCIFAR10DVS(cifar10_dvs.CIFAR10DVS):

    def downloadable(self):
        return False


class DatasetWithTransform(Dataset):
    def __init__(self, subset: Dataset, transform) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        frames, label = self.subset[index]
        return self.transform(frames), int(label)


class EventFrameTransform:

    def __init__(
            self,
            *,
            train: bool,
            size: int = 128,
            crop_padding: int = 8,
            affine_prob: float = 0.35,
            max_rotate: float = 10.0,
            max_translate: int = 6,
            erase_prob: float = 0.20,
            temporal_shift: int = 2,
            frame_drop_prob: float = 0.20,
            polarity_drop_prob: float = 0.04,
            log1p: bool = False,
    ) -> None:
        self.train = train
        self.size = size
        self.crop_padding = crop_padding
        self.affine_prob = affine_prob
        self.max_rotate = max_rotate
        self.max_translate = max_translate
        self.erase_prob = erase_prob
        self.temporal_shift = temporal_shift
        self.frame_drop_prob = frame_drop_prob
        self.polarity_drop_prob = polarity_drop_prob
        self.log1p = log1p

    def __call__(self, frames) -> torch.Tensor:
        x = torch.as_tensor(frames, dtype=torch.float32).contiguous()
        if x.ndim != 4:
            raise ValueError(
                f"CIFAR10-DVS frames must be [T,C,H,W], got {tuple(x.shape)}"
            )
        if self.log1p:
            x = torch.log1p(x)

        if not self.train:
            return x

        t, c, h, w = x.shape

        # The same crop is used for all T*C maps.
        if self.crop_padding > 0:
            p = self.crop_padding
            x = F.pad(x, (p, p, p, p))
            max_top = x.shape[-2] - self.size
            max_left = x.shape[-1] - self.size
            top = int(torch.randint(0, max_top + 1, (1,)).item())
            left = int(torch.randint(0, max_left + 1, (1,)).item())
            x = x[..., top:top + self.size, left:left + self.size]

        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(-1,))

        if torch.rand(()) < self.affine_prob:
            merged = x.reshape(t * c, self.size, self.size)
            angle = float(
                torch.empty(1).uniform_(-self.max_rotate, self.max_rotate).item()
            )
            tx = int(
                torch.randint(
                    -self.max_translate, self.max_translate + 1, (1,)
                ).item()
            )
            ty = int(
                torch.randint(
                    -self.max_translate, self.max_translate + 1, (1,)
                ).item()
            )
            scale = float(torch.empty(1).uniform_(0.92, 1.08).item())
            merged = TF.affine(
                merged,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )
            x = merged.reshape(t, c, self.size, self.size)

        # Non-wrapping temporal shift.
        if self.temporal_shift > 0:
            shift = int(
                torch.randint(
                    -self.temporal_shift, self.temporal_shift + 1, (1,)
                ).item()
            )
            if shift > 0:
                shifted = torch.zeros_like(x)
                shifted[shift:] = x[:-shift]
                x = shifted
            elif shift < 0:
                shifted = torch.zeros_like(x)
                shifted[:shift] = x[-shift:]
                x = shifted

        # Mild event-frame dropout promotes temporal robustness without
        # generating a costly element-wise random mask.
        if t > 2 and torch.rand(()) < self.frame_drop_prob:
            max_drop = max(1, min(2, t // 4))
            count = int(torch.randint(1, max_drop + 1, (1,)).item())
            idx = torch.randperm(t)[:count]
            x[idx] = 0

        if c == 2 and torch.rand(()) < self.polarity_drop_prob:
            channel = int(torch.randint(0, 2, (1,)).item())
            x[:, channel] = 0

        if torch.rand(()) < self.erase_prob:
            erase_h = int(
                torch.randint(self.size // 12, self.size // 4 + 1, (1,)).item()
            )
            erase_w = int(
                torch.randint(self.size // 12, self.size // 4 + 1, (1,)).item()
            )
            top = int(
                torch.randint(0, self.size - erase_h + 1, (1,)).item()
            )
            left = int(
                torch.randint(0, self.size - erase_w + 1, (1,)).item()
            )
            x[..., top:top + erase_h, left:left + erase_w] = 0

        # Mild count scaling; BN in the stem absorbs global amplitude changes.
        gain = float(torch.empty(1).uniform_(0.90, 1.10).item())
        return (x * gain).contiguous()


def fallback_classwise_split(
        dataset: Dataset,
        train_ratio: float,
        num_classes: int,
        *,
        random_split: bool,
        seed: int,
):
    labels = [[] for _ in range(num_classes)]
    for index in range(len(dataset)):
        _, label = dataset[index]
        labels[int(label)].append(index)

    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls_idx in labels:
        if random_split:
            cls_idx = list(cls_idx)
            rng.shuffle(cls_idx)
        pivot = math.ceil(len(cls_idx) * train_ratio)
        train_idx.extend(cls_idx[:pivot])
        test_idx.extend(cls_idx[pivot:])

    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, test_idx),
    )


def build_loaders(args):
    dataset_cls = (
        cifar10_dvs.CIFAR10DVS if args.allow_download else OfflineCIFAR10DVS
    )
    full_dataset = dataset_cls(
        root=args.data_path,
        data_type="frame",
        frames_number=args.frames,
        split_by="number",
        transform=None,
    )

    if split_to_train_test_set is not None:
        try:
            train_subset, test_subset = split_to_train_test_set(
                0.9,
                full_dataset,
                10,
                random_split=args.random_split,
            )
        except TypeError:
            train_subset, test_subset = split_to_train_test_set(
                0.9, full_dataset, 10
            )
    else:
        train_subset, test_subset = fallback_classwise_split(
            full_dataset,
            0.9,
            10,
            random_split=args.random_split,
            seed=args.seed,
        )

    train_transform = EventFrameTransform(
        train=True,
        size=args.img_size,
        crop_padding=args.crop_padding,
        affine_prob=args.affine_prob,
        max_rotate=args.max_rotate,
        max_translate=args.max_translate,
        erase_prob=args.erase_prob,
        temporal_shift=args.temporal_shift,
        frame_drop_prob=args.frame_drop_prob,
        polarity_drop_prob=args.polarity_drop_prob,
        log1p=args.log1p,
    )
    test_transform = EventFrameTransform(
        train=False,
        size=args.img_size,
        log1p=args.log1p,
    )

    train_set = DatasetWithTransform(train_subset, train_transform)
    test_set = DatasetWithTransform(test_subset, test_transform)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    print(
        f"Dataset split: train={len(train_set)}, test={len(test_set)}, "
        f"frames={args.frames}"
    )
    return train_loader, test_loader


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def smooth_one_hot(
        targets: torch.Tensor,
        num_classes: int,
        smoothing: float,
) -> torch.Tensor:
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    y = torch.full(
        (targets.shape[0], num_classes),
        off,
        device=targets.device,
        dtype=torch.float32,
    )
    return y.scatter_(1, targets[:, None], on)


class EventMixupCutMix:
    """Mix entire event samples with one shared spatial box over all time bins."""

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
    def _cutmix_box(
            height: int,
            width: int,
            lam: float,
    ) -> Tuple[int, int, int, int]:
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
            targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        soft_targets = smooth_one_hot(
            targets, self.num_classes, self.label_smoothing
        )
        if (
                not self.enabled
                or self.probability <= 0
                or torch.rand((), device=samples.device) > self.probability
        ):
            return samples, soft_targets

        perm = torch.randperm(samples.shape[0], device=samples.device)
        use_cutmix = (
                self.cutmix_alpha > 0
                and (
                        self.mixup_alpha <= 0
                        or torch.rand((), device=samples.device)
                        < self.switch_probability
                )
        )

        if use_cutmix:
            lam = self._sample_beta(self.cutmix_alpha)
            y1, y2, x1, x2 = self._cutmix_box(
                samples.shape[-2], samples.shape[-1], lam
            )
            mixed = samples.clone()
            mixed[..., y1:y2, x1:x2] = samples[
                                       perm, ..., y1:y2, x1:x2
                                       ]
            area = (y2 - y1) * (x2 - x1)
            lam = 1.0 - area / (
                    samples.shape[-2] * samples.shape[-1]
            )
        else:
            lam = self._sample_beta(self.mixup_alpha)
            mixed = samples * lam + samples[perm] * (1.0 - lam)

        mixed_targets = (
                soft_targets * lam + soft_targets[perm] * (1.0 - lam)
        )
        return mixed, mixed_targets


def soft_target_cross_entropy(
        logits: torch.Tensor,
        targets: torch.Tensor,
) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def temporal_classification_loss(
        temporal_logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        temporal_aux_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # temporal_logits: [T, B, C]
    mean_logits = temporal_logits.mean(dim=0)
    main_loss = soft_target_cross_entropy(mean_logits, targets)
    if temporal_aux_weight <= 0:
        return main_loss, mean_logits

    t, b, c = temporal_logits.shape
    repeated_targets = targets.unsqueeze(0).expand(t, -1, -1)
    aux_loss = soft_target_cross_entropy(
        temporal_logits.reshape(t * b, c),
        repeated_targets.reshape(t * b, c),
    )
    return main_loss + temporal_aux_weight * aux_loss, mean_logits


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
        if no_weight_decay(name, parameter):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
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
        lr = min_lr + progress * (base_lr - min_lr)
    else:
        progress = (epoch - warmup_epochs) / max(
            1, total_epochs - warmup_epochs - 1
        )
        lr = min_lr + 0.5 * (base_lr - min_lr) * (
                1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


@torch.no_grad()
def topk_accuracy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        topk: Sequence[int] = (1, 5),
):
    maxk = min(max(topk), logits.shape[-1])
    pred = logits.topk(maxk, dim=1).indices.t()
    correct = pred.eq(targets.view(1, -1))
    results = []
    for k in topk:
        k = min(k, logits.shape[-1])
        results.append(
            correct[:k].reshape(-1).float().sum() * (100.0 / targets.numel())
        )
    return results


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        mix_operator: EventMixupCutMix,
        device: torch.device,
        epoch: int,
        args,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    update_count = 0
    start = time.time()

    mix_operator.enabled = epoch < args.mixup_off_epoch

    for step, (samples, hard_targets) in enumerate(loader):
        samples = samples.to(device, non_blocking=True)
        hard_targets = hard_targets.to(device, non_blocking=True)
        samples, soft_targets = mix_operator(samples, hard_targets)

        # Correctly normalize the final, possibly shorter accumulation window.
        remainder = len(loader) % args.accum_steps
        final_window_start = len(loader) - remainder if remainder else len(loader)
        divisor = (
            remainder
            if remainder and step >= final_window_start
            else args.accum_steps
        )

        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
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
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_count += 1

        batch_size = samples.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_correct += int(
            mean_logits.detach().argmax(dim=1).eq(hard_targets).sum()
        )
        total_samples += batch_size

        if (step + 1) % args.print_freq == 0:
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"epoch={epoch:03d} step={step + 1:04d}/{len(loader):04d} "
                f"loss={total_loss / total_samples:.4f} "
                f"train_acc={100.0 * total_correct / total_samples:.2f} "
                f"samples/s={total_samples / elapsed:.1f}"
            )

    return {
        "loss": total_loss / max(1, total_samples),
        "acc1": 100.0 * total_correct / max(1, total_samples),
        "updates": update_count,
    }


@torch.no_grad()
def forward_mean_logits(
        model: nn.Module,
        samples: torch.Tensor,
) -> torch.Tensor:
    logits = model(samples).mean(dim=0)
    reset_net(model)
    return logits


@torch.no_grad()
def evaluate(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        *,
        amp_enabled: bool,
) -> Dict[str, float]:
    """Evaluate only the raw model with a single, unaugmented forward pass."""
    model.eval()
    loss_sum = 0.0
    correct1 = 0.0
    correct5 = 0.0
    total = 0

    for samples, targets in loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = forward_mean_logits(model, samples)
            loss = F.cross_entropy(logits, targets)

        acc1, acc5 = topk_accuracy(logits, targets, (1, 5))
        batch_size = samples.shape[0]
        loss_sum += float(loss) * batch_size
        correct1 += float(acc1) * batch_size / 100.0
        correct5 += float(acc5) * batch_size / 100.0
        total += batch_size

    return {
        "loss": loss_sum / max(1, total),
        "acc1": 100.0 * correct1 / max(1, total),
        "acc5": 100.0 * correct5 / max(1, total),
    }


def save_checkpoint(
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        epoch: int,
        best_raw_acc: float,
        current_raw_acc: float,
        checkpoint_metric: str,
        args,
) -> None:
    """
    Save the raw model only.

    checkpoint_metric:
        - "latest": latest training state
        - "raw_eval": epoch selected by raw single-view accuracy
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "best_raw_acc": float(best_raw_acc),
        "current_raw_acc": float(current_raw_acc),
        "checkpoint_metric": checkpoint_metric,
        "args": vars(args),

        # Backward-compatible field for older log/checkpoint readers.
        "best_acc": float(best_raw_acc),
    }
    torch.save(state, path)


def load_checkpoint(
        path: str,
        *,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
):
    """
    Load only the raw model state.

    Extra keys such as an old checkpoint's "ema" entry are safely ignored.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "ALSM CIFAR10-DVS high-accuracy training",
        add_help=False,
    )

    parser.add_argument("-f", "--file", default="file")
    parser.add_argument(
        "--data-path",
        type=str,
        default=r"./data",
        help="CIFAR10-DVS dataset root.",
    )
    parser.add_argument("--output-dir", type=str, default="./runs/alsm_cifar10dvs")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--random-split", action="store_true")

    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=10)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)

    parser.add_argument("--lsm-ratio", type=float, default=4.0)

    parser.add_argument("--drop-path", type=float, default=0.10)
    parser.add_argument("--attn-drop", type=float, default=0.0)
    parser.add_argument("--gate-dropout", type=float, default=0.0)
    parser.add_argument("--head-dropout", type=float, default=0.10)
    parser.add_argument("--attn-scale", type=float, default=0.125)
    parser.add_argument(
        "--backend",
        choices=("auto", "cupy", "torch"),
        default="cupy",
    )

    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--warmup-epochs", type=int, default=10)

    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip-grad", type=float, default=1.0)

    parser.add_argument(
        "--temporal-aux-weight",
        type=float,
        default=0.10,
    )

    parser.add_argument("--mixup", type=float, default=0.5)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=0.50)
    parser.add_argument("--mix-switch-prob", type=float, default=0.5)
    parser.add_argument("--smoothing", type=float, default=0.10)

    parser.add_argument("--mixup-off-epoch", type=int, default=270)

    parser.add_argument("--crop-padding", type=int, default=8)

    parser.add_argument("--affine-prob", type=float, default=0.25)
    parser.add_argument("--max-rotate", type=float, default=8.0)
    parser.add_argument("--max-translate", type=int, default=4)

    parser.add_argument("--erase-prob", type=float, default=0.10)
    parser.add_argument("--temporal-shift", type=int, default=1)
    parser.add_argument("--frame-drop-prob", type=float, default=0.10)
    parser.add_argument("--polarity-drop-prob", type=float, default=0.02)

    parser.add_argument("--log1p", action="store_true")

    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1207)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--deterministic", action="store_true")
    return parser


def main(args) -> None:
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(
            f"CIFAR10-DVS 数据集路径不存在: {args.data_path}\n"
            "请在 build_parser() 中修改 --data-path 的 default 值。"
        )

    if not args.data_path:
        raise ValueError(
            "CIFAR10-DVS path is empty. Please edit the default value of "
            "--data-path in build_parser()."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    device = torch.device("cuda")

    args.backend = resolve_backend(args.backend)
    print(f"NumPy version: {np.__version__}")
    print(f"Spiking backend: {args.backend}")
    if args.backend == "cupy" and int(np.__version__.split(".")[0]) >= 2:
        print("Applied NumPy 2.x compatibility aliases for SpikingJelly CuPy kernels.")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    train_loader, test_loader = build_loaders(args)

    model = ALSMCIFAR10DVS(
        img_size=args.img_size,
        in_channels=2,
        num_classes=args.num_classes,
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
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable / 1e6:.3f} M")

    optimizer = torch.optim.AdamW(
        parameter_groups(model, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    mix_operator = EventMixupCutMix(
        num_classes=args.num_classes,
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        probability=args.mix_prob,
        switch_probability=args.mix_switch_prob,
        label_smoothing=args.smoothing,
    )

    start_epoch = 0
    best_raw_acc = 0.0
    checkpoint = None

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            optimizer=None if args.eval else optimizer,
            scaler=None if args.eval else scaler,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1

        # New raw-only checkpoints use best_raw_acc. Dual/older checkpoints
        # are also accepted; current_raw_acc is preferred over an EMA-based
        # historical best_acc when available.
        best_raw_acc = float(
            checkpoint.get(
                "best_raw_acc",
                checkpoint.get(
                    "current_raw_acc",
                    checkpoint.get("best_acc", 0.0),
                ),
            )
        )

        print(
            f"Loaded {args.resume}; start_epoch={start_epoch}, "
            f"best_raw={best_raw_acc:.3f}"
        )

    if args.eval:
        stats = evaluate(
            model,
            test_loader,
            device,
            amp_enabled=args.amp,
        )
        print(f"Raw evaluation: {stats}")
        return

    log_path = output_dir / "log.jsonl"
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        lr = set_learning_rate(
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
            "lr": lr,
            "train": train_stats,
            "raw_eval": raw_stats,
            "best_raw_acc1": best_raw_acc,

            # Retained for compatibility with existing log readers.
            "best_acc1": best_raw_acc,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Always overwrite the latest state for interruption recovery.
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

        # Save the best raw single-view model independently.
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
                f"  Saved checkpoint_best_raw.pth "
                f"(raw Acc@1={best_raw_acc:.2f})"
            )

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}: "
            f"lr={lr:.3e}, train_loss={train_stats['loss']:.4f}, "
            f"raw_acc={raw_stats['acc1']:.2f}, "
            f"best_raw={best_raw_acc:.2f}"
        )

    elapsed = str(
        datetime.timedelta(seconds=int(time.time() - train_start))
    )
    print(
        f"Training completed. Best raw Acc@1={best_raw_acc:.3f}; "
        f"time={elapsed}"
    )


def parse_args():

    parser = build_parser()
    return parser.parse_args(args=[])


if __name__ == "__main__":
    args = parse_args()
    main(args)

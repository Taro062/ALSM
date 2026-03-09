import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.loss import SoftTargetCrossEntropy
from timm.data import Mixup

import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from model import spikformerLSM




class ToFloat32(object):
    def __call__(self, x):
        return x.astype(np.float32)


class LocalNMNIST(Dataset):
    """
    自定义的本地 N-MNIST 读取器。
    """

    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = 'Train' if train else 'Test'
        self.data_dir = os.path.join(root_dir, self.split)  # 例如: .../N-MNIST/Train
        self.samples = []

        # 检查路径是否存在
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"❌ 找不到数据文件夹: {self.data_dir}\n请确保你的目录结构是: {root_dir}/Train 和 {root_dir}/Test")

        # 遍历文件夹收集文件 (0-9)
        print(f"🔍 Scanning files in {self.data_dir}...")
        for digit in range(10):
            digit_dir = os.path.join(self.data_dir, str(digit))
            if os.path.exists(digit_dir):
                files = [os.path.join(digit_dir, f) for f in os.listdir(digit_dir) if f.endswith('.bin')]
                for f in files:
                    self.samples.append((f, digit))

        print(f"✅ Loaded {len(self.samples)} samples from {self.split} set.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        # 使用 NumPy 快速读取二进制文件
        with open(file_path, 'rb') as f:
            raw = np.fromfile(f, dtype=np.uint8)

        x = raw[0::5]
        y = raw[1::5]
        data_p_ts = raw[2::5]
        data_ts_mid = raw[3::5]
        data_ts_low = raw[4::5]

        # 解析时间戳 (ts) 和 极性 (p)
        p = (data_p_ts >> 7).astype(np.int8)
        t = ((data_p_ts & 127).astype(np.int64) << 16) | \
            (data_ts_mid.astype(np.int64) << 8) | \
            (data_ts_low.astype(np.int64))

        # 构建 Tonic 需要的结构化数组 (Structure Array)
        events = np.zeros(len(x), dtype=[('x', '<i2'), ('y', '<i2'), ('t', '<i8'), ('p', '<i2')])
        events['x'] = x
        events['y'] = y
        events['t'] = t
        events['p'] = p

        if self.transform:
            events = self.transform(events)

        return events, label


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='SpikformerLSM Training (Local Dataset + Cache)')

    parser.add_argument('--data-path', default='D:/PythonProject/data/N-MNIST',
                        help='Path containing Train/Test folders')

    parser.add_argument('--output-dir', default='./logs_nmnist', help='path where to save')
    parser.add_argument('--cache-dir', default='./cache_nmnist', help='Tonic disk cache directory')

    parser.add_argument('--seed', default=42, type=int, help='Random seed')

    # 训练超参
    parser.add_argument('--batch-size', default=192, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0.05)

    # 增强参数
    parser.add_argument('--mixup', type=float, default=0.5)
    parser.add_argument('--cutmix', type=float, default=0.0)
    parser.add_argument('--mixup-prob', type=float, default=0.5)
    parser.add_argument('--smoothing', type=float, default=0.1)

    # 模型参数
    parser.add_argument('--T', default=4, type=int)

    # 硬件配置
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--amp', action='store_true', default=True)

    return parser.parse_args()


class SNNAugmentWide(nn.Module):
    def __init__(self, shift=2, noise=0.05):
        super().__init__()
        self.shift = shift
        self.noise = noise

    def forward(self, x):
        if self.training:
            dx = random.randint(-self.shift, self.shift)
            dy = random.randint(-self.shift, self.shift)
            if dx != 0 or dy != 0:
                x = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))

            if self.noise > 0:
                mask = torch.rand_like(x) < self.noise
                x[mask] = 0.0
        return x


def train_epoch(model, criterion, optimizer, loader, device, epoch, scaler, mixup_fn):
    model.train()
    losses = []
    correct = 0
    total = 0
    augmentor = SNNAugmentWide().to(device)

    for i, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).long()
        images = augmentor(images)

        if mixup_fn is not None:
            images, target_soft = mixup_fn(images, target)
        else:
            target_soft = target

        for m in model.modules():
            if hasattr(m, 'reset') and callable(m.reset): m.reset()

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            output = model(images)
            mean_out = output.mean(0)
            loss = criterion(mean_out, target_soft)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        if mixup_fn is not None:
            _, target_idx = target_soft.max(1)
        else:
            target_idx = target

        _, pred = mean_out.max(1)
        correct += pred.eq(target_idx).sum().item()
        total += target.size(0)

        if i % 50 == 0:
            print(
                f'Epoch [{epoch}] Batch [{i}/{len(loader)}] Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%')

    return sum(losses) / len(losses), 100. * correct / total


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    losses = []

    for images, target in loader:
        images, target = images.to(device, non_blocking=True).float(), target.to(device, non_blocking=True).long()
        for m in model.modules():
            if hasattr(m, 'reset') and callable(m.reset): m.reset()

        output = model(images)
        mean_out = output.mean(0)
        loss = criterion(mean_out, target)

        losses.append(loss.item())
        _, pred = mean_out.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return sum(losses) / len(losses), 100. * correct / total



def main():
    args = parse_args()

    # 0. 清理缓存提示
    if os.path.exists(args.cache_dir):
        print(f"⚠️ 警告: 如果遇到 FileNotFoundError，请手动删除缓存目录: {args.cache_dir}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"🔒 Seed set to: {args.seed}")

    print(f"🚀 Start Training N-MNIST (Local Dataset + Tonic Cache)")
    print(f"   Data Path: {args.data_path}")
    print(f"   Cache Directory: {args.cache_dir}")

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # N-MNIST 传感器尺寸
    sensor_size = (34, 34, 2)
    target_size = (32, 32, 2) # 裁剪后的尺寸

    # ===============================================================
    # ✅ 修复点: 调整 transform 顺序
    # ===============================================================
    frame_transform = transforms.Compose([
        # 1. 先裁剪事件 (Events -> Cropped Events)
        transforms.CenterCrop(
            sensor_size=sensor_size,
            size=(32, 32)
        ),
        # 2. 再转 Frame (Cropped Events -> Tensor)
        # 注意: 输入给 ToFrame 的已经是 32x32 的数据了
        transforms.ToFrame(
            sensor_size=target_size,
            n_time_bins=args.T
        ),
        ToFloat32()
    ])

    # 1. 实例化本地数据集
    full_dataset_train = LocalNMNIST(root_dir=args.data_path, train=True, transform=frame_transform)
    full_dataset_test = LocalNMNIST(root_dir=args.data_path, train=False, transform=frame_transform)

    # 2. 包装缓存层
    cached_trainset = DiskCachedDataset(
        full_dataset_train,
        cache_path=os.path.join(args.cache_dir, "train"),
    )

    cached_testset = DiskCachedDataset(
        full_dataset_test,
        cache_path=os.path.join(args.cache_dir, "test"),
    )

    # 3. DataLoader
    train_loader = DataLoader(
        cached_trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        cached_testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print(f"Train samples: {len(cached_trainset)}, Test samples: {len(cached_testset)}")
    print("Creating SpikformerLSM...")

    model = spikformerLSM(
        img_size_h=32,
        img_size_w=32,
        patch_size=4,
        embed_dims=128,
        num_heads=8,
        mlp_ratios=4,
        in_channels=2,
        num_classes=10,
        T=args.T
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.min_lr
    )

    mixup_fn = None
    if args.mixup > 0:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                         prob=args.mixup_prob, switch_prob=0.5, mode='batch',
                         label_smoothing=args.smoothing, num_classes=10)
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    best_acc = 0.0
    writer = SummaryWriter(log_dir=args.output_dir)

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, criterion, optimizer, train_loader, device, epoch, scaler, mixup_fn)
        val_loss, val_acc = validate(model, test_loader, device)

        scheduler.step()

        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Acc', train_acc, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Acc', val_acc, epoch)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} ({epoch_time:.1f}s) | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"🌟 New Best Accuracy: {best_acc:.2f}%")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'acc': best_acc,
                'args': args
            }, os.path.join(args.output_dir, 'best_model.pth'))

    print(f"Training Finished. Best Accuracy: {best_acc:.2f}%")
    writer.close()


if __name__ == '__main__':
    main()
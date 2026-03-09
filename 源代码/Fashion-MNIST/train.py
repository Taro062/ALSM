import argparse
import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, accuracy
from torch.utils.tensorboard import SummaryWriter


from model import spikformerLSM



def get_args():
    parser = argparse.ArgumentParser('SpikformerLSM Training Script')

    # 基础参数
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--T', default=6, type=int, help='Time steps')

    # 模型参数
    parser.add_argument('--img-size', default=28, type=int, help='Input image size')
    parser.add_argument('--embed-dims', default=128, type=int)
    parser.add_argument('--num-heads', default=4, type=int)
    parser.add_argument('--depths', default=2, type=int)

    # 优化器参数
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')

    # 调度器参数
    parser.add_argument('--sched', default='cosine', type=str)
    parser.add_argument('--warmup-epochs', default=5, type=int)
    parser.add_argument('--min-lr', default=1e-5, type=float)
    parser.add_argument('--clip-grad', default=1.0, type=float)

    # 增强参数
    parser.add_argument('--smoothing', type=float, default=0.1)

    # 路径与设备
    parser.add_argument('--output-dir', default='./output', help='path where to save')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--seed', default=42, type=int)

    return parser.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    writer = SummaryWriter(log_dir=args.output_dir)
    print(f"Loading Fashion-MNIST (Size: {args.img_size}x{args.img_size})...")


    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_dataset = datasets.FashionMNIST(root='D:\PythonProject\data\FashionMNIST', train=True, download=True, transform=transform_train)
    val_dataset = datasets.FashionMNIST(root='D:\PythonProject\data\FashionMNIST', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Creating model: SpikformerLSM (T={args.T}, Dim={args.embed_dims})")
    model = spikformerLSM(
        img_size_h=args.img_size,
        img_size_w=args.img_size,
        patch_size=4,
        in_channels=1,  # 灰度图
        num_classes=10,
        embed_dims=args.embed_dims,
        num_heads=args.num_heads,
        depths=args.depths,
        T=args.T
    )
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_parameters}")

    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    max_accuracy = 0.0

    print("Start training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch,
            loss_scaler, args.clip_grad, args.T
        )

        lr_scheduler.step(epoch)

        test_stats = evaluate(model, val_loader, device, args.T)
        print(f"Epoch {epoch}: Train Loss {train_stats['loss']:.4f}, Test Acc {test_stats['acc1']:.2f}%")

        writer.add_scalar('Train/Loss', train_stats['loss'], epoch)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Test/Loss', test_stats['loss'], epoch)
        writer.add_scalar('Test/Accuracy', test_stats['acc1'], epoch)
        writer.flush()

        if test_stats["acc1"] > max_accuracy:
            max_accuracy = test_stats["acc1"]
            save_path = os.path.join(args.output_dir, 'best_checkpoint.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }, save_path)
            print(f"New Best Accuracy: {max_accuracy:.2f}%")

    writer.close()

    total_time = time.time() - start_time
    print(f"Training time {str(datetime.timedelta(seconds=int(total_time)))}")


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, clip_grad, T):
    model.train()
    losses = []

    for batch_idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = targets.to(device)


        samples = samples.unsqueeze(1).repeat(1, T, 1, 1, 1)

        functional_reset(model)

        with torch.amp.autocast('cuda'):

            outputs = model(samples)

            outputs_mean = outputs.mean(0)
            loss = criterion(outputs_mean, targets)

        optimizer.zero_grad()
        loss_scaler(loss, optimizer, clip_grad=clip_grad, parameters=model.parameters())

        losses.append(loss.item())

    return {'loss': np.mean(losses)}


@torch.no_grad()
def evaluate(model, data_loader, device, T):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    metric_logger = {'loss': [], 'acc1': [], 'acc5': []}

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = targets.to(device)


        samples = samples.unsqueeze(1).repeat(1, T, 1, 1, 1)

        functional_reset(model)

        output = model(samples)
        output_mean = output.mean(0)

        loss = criterion(output_mean, targets)
        acc1, acc5 = accuracy(output_mean, targets, topk=(1, 5))

        metric_logger['loss'].append(loss.item())
        metric_logger['acc1'].append(acc1.item())
        metric_logger['acc5'].append(acc5.item())

    return {k: np.mean(v) for k, v in metric_logger.items()}


def functional_reset(model):
    """安全重置网络状态"""
    for m in model.modules():
        if hasattr(m, 'reset'):
            m.reset()


if __name__ == '__main__':
    main()
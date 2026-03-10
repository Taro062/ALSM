import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

import os
from pathlib import Path
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

import sys
import os

sys.path.append(r'/root/LSM/model')
from spiklsmAG3 import spikformerLSM
import utils
from spikingjelly.datasets import dvs128_gesture

# 自定义DVS128Gesture数据集包装类
class DVS128GestureDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, train=True, num_frames=20, transform=None):

        self.root_dir = root_dir
        self.train = train
        self.num_frames = num_frames
        self.transform = transform

        # 使用spikingjelly提供的DVS128Gesture数据集
        self.dataset = dvs128_gesture.DVS128Gesture(
            root=root_dir,
            train=train,
            data_type='frame',
            frames_number=num_frames,
            split_by='number'
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 获取帧数据和标签
        frames, label = self.dataset[idx]

        # 转换为PyTorch张量
        frames_tensor = torch.from_numpy(frames).float()

        # 应用变换
        if self.transform:
            frames_tensor = self.transform(frames_tensor)

        return frames_tensor, label


def load_data(args):

    os.makedirs(args.data_path, exist_ok=True)

    train_set = DVS128GestureDataset(
        root_dir=args.data_path,
        train=True,
        num_frames=args.T_train
    )

    test_set = DVS128GestureDataset(
        root_dir=args.data_path,
        train=False,
        num_frames=args.T_test
    )

    data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"训练集大小: {len(train_set)}")
    print(f"测试集大小: {len(test_set)}")

    return data_loader, data_loader_test

# 安全重置网络状态
def safe_reset_net(net):
    try:
        from spikingjelly.clock_driven import functional
        functional.reset_net(net)
    except:
        # 手动重置
        for m in net.modules():
            if hasattr(m, 'reset') and callable(getattr(m, 'reset', None)):
                try:
                    m.reset()
                except Exception as e:
                    print(f"重置模块 {type(m).__name__} 时出错: {e}")
            elif hasattr(m, 'v'):
                try:
                    m.v.data.fill_(0.)
                except Exception:
                    pass
            elif hasattr(m, 'h'):
                try:
                    m.h.data.fill_(0.)
                except Exception:
                    pass

# 训练一个epoch
def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None, T_train=None,
                    mixup_fn=None, clip_grad=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, target in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 安全重置网络状态
        safe_reset_net(model)

        # 应用mixup
        if mixup_fn is not None:
            images, target = mixup_fn(images, target)

        if scaler is not None:
            with amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 计算准确率（仅在非mixup情况下）
        if mixup_fn is None:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if mixup_fn is None:
        return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    else:
        return metric_logger.loss.global_avg, 0.0, 0.0


# 评估函数
def evaluate(model, criterion, data_loader, device, print_freq, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = header

    with torch.no_grad():
        for images, target in metric_logger.log_every(data_loader, print_freq, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # 安全重置网络状态
            safe_reset_net(model)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print(
        f'{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}')
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def parse_args():
    parser = argparse.ArgumentParser('SpikformerLSM training script', add_help=False)

    # 基本参数
    parser.add_argument("-f", "--file", default="file")
    parser.add_argument('--batch-size', default=72, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=400, type=int, help='Number of epochs')
    parser.add_argument('--num-classes', type=int, default=11, help='Number of classes')

    # # 模型参数 - 优化配置
    parser.add_argument('--patch-size', type=int, default=8, help='Patch size')  # 减小patch size
    parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension')  # 增加嵌入维度
    parser.add_argument('--depth', type=int, default=2, help='Depth')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--img-size-h', type=int, default=128, help='Image height')
    parser.add_argument('--img-size-w', type=int, default=128, help='Image width')

    # # LSM参数优化
    parser.add_argument('--lsm-hidden', type=int, default=768, help='LSM hidden size')
    parser.add_argument('--lsm-tau-mem', type=float, default=0.95, help='LSM membrane time constant')
    parser.add_argument('--lsm-tau-syn', type=float, default=0.8, help='LSM synaptic time constant')

    # 优化器参数
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')  # 提高初始学习率
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')  # 减少权重衰减
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')

    # 学习率调度
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--min-lr', type=float, default=1e-7, help='Lower lr bound')  # 降低最小学习率
    parser.add_argument('--warmup-epochs', type=int, default=40, help='Warmup epochs')  # 延长预热期
    parser.add_argument('--warmup-lr', type=float, default=1e-6, help='Warmup learning rate')
    parser.add_argument('--lr-restart', action='store_true', help='Enable cosine annealing with restarts')
    parser.add_argument('--restart-epochs', type=int, default=80, help='Epochs for each restart cycle')

    # 数据增强 - 强化配置
    parser.add_argument('--mixup', type=float, default=0.5, help='Mixup alpha')  # 增强mixup
    parser.add_argument('--cutmix', type=float, default=0.5, help='Cutmix alpha')  # 启用cutmix
    parser.add_argument('--mixup-prob', type=float, default=1.0, help='Mixup probability')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='Mixup switch probability')
    parser.add_argument('--mixup-mode', type=str, default='batch', help='Mixup mode')
    parser.add_argument('--smoothing', type=float, default=0.15, help='Label smoothing')  # 增加标签平滑

    # 正则化
    parser.add_argument('--drop', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--drop_path', type=float, default=0.2, help='Drop path rate')
    parser.add_argument('--clip-grad', type=float, default=1.0, help='Gradient clipping')

    # 训练设置
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--T-train', type=int, default=10, help='Simulation time steps for training')
    parser.add_argument('--T-test', type=int, default=10, help='Simulation time steps for testing')

    # TTA设置
    parser.add_argument('--use-tta', action='store_true', default=True, help='Use test time augmentation')
    parser.add_argument('--tta-transforms', type=int, default=15, help='Number of TTA transforms')

    # 渐进式训练
    parser.add_argument('--progressive-training', action='store_true', help='Enable progressive training')
    parser.add_argument('--progressive-epochs', type=int, default=100, help='Epochs for each progressive stage')

    # 数据和输出
    parser.add_argument('--data-path', default='./root', type=str, help='Dataset path')
    parser.add_argument('--output-dir', default='./logsnew', type=str, help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device to use for training')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='Start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num-workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--pin-mem', action='store_true', default=True, help='Pin CPU memory in DataLoader')

    # 日志
    parser.add_argument('--print-freq', default=50, type=int, help='Print frequency')
    parser.add_argument('--tb', action='store_true', default=True, help='Use tensorboard')

    return parser.parse_args()


def save_on_master(state, path):
    torch.save(state, path)


def main(args):
    print(args)

    device = torch.device(args.device)

    # 设置随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 初始化TensorBoard
    writer = None
    if args.tb:
        writer = SummaryWriter(log_dir=args.output_dir)

    # 加载数据
    data_loader, data_loader_test = load_data(args)

    print(f"Creating model: SpikformerLSM")

    # 创建优化的模型
    model = spikformerLSM(
        patch_size=args.patch_size,
        in_channels=2,
        num_classes=args.num_classes,
        embed_dims=args.embed_dim,
        num_heads=args.num_heads,
        drop_path_rate=args.drop_path,
        depths=args.depth,
        T=args.T_train
    )
    model.to(device)

    # 计算参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of params: {n_parameters}')

    # 同步批归一化
    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.DataParallel(model)

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # 创建学习率调度器
    if args.lr_restart:
        # 使用带重启的余弦退火
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.restart_epochs,
            T_mult=2,
            eta_min=args.min_lr
        )
    else:
        # 原有的余弦退火调度器
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=args.min_lr
        )

    # 添加预热调度器
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.warmup_lr / args.lr,
            end_factor=1.0,
            total_iters=args.warmup_epochs
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, lr_scheduler],
            milestones=[args.warmup_epochs]
        )

    # 创建混合精度训练的缩放器
    scaler = amp.GradScaler() if args.amp else None

    # 初始化变量
    start_epoch = args.start_epoch
    best_acc = 0.0

    # 从检查点恢复
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"从检查点 {args.resume} 恢复")
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            # 恢复学习率调度器状态
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                print("恢复学习率调度器状态")

            # 从上一个epoch开始
            start_epoch = checkpoint['epoch'] + 1

            # 恢复最佳准确率
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']

            if scaler and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])

            print(f"从epoch {start_epoch} 恢复训练，最佳准确率: {best_acc:.2f}%")
        else:
            print(f"未找到检查点: {args.resume}")

    # 创建mixup增强
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            cutmix_minmax=None, prob=args.mixup_prob, switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode, label_smoothing=args.smoothing, num_classes=args.num_classes
        )
        # 训练时使用SoftTargetCrossEntropy损失函数
        train_criterion = SoftTargetCrossEntropy()
        # 评估时使用标准交叉熵损失函数
        eval_criterion = torch.nn.CrossEntropyLoss()
    else:
        # 使用组合损失函数
        train_criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        eval_criterion = torch.nn.CrossEntropyLoss()

    # 仅评估模式
    if args.eval:
        # 评估
        if args.use_tta:
            test_loss, test_acc1, test_acc5 = evaluate_with_tta(
                model, eval_criterion, data_loader_test, device, args.print_freq,
                tta_transforms=args.tta_transforms
            )
        else:
            test_loss, test_acc1, test_acc5 = evaluate(
                model, eval_criterion, data_loader_test, device, args.print_freq
            )
        print(f"测试准确率: {test_acc1:.2f}%")
        return

    # 记录开始时间
    start_time = time.time()

    print(f"开始训练")

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        if args.progressive_training and epoch < args.progressive_epochs:
            if mixup_fn is not None:
                mixup_fn.mixup_alpha = args.mixup * 1.5
                mixup_fn.cutmix_alpha = args.cutmix * 1.5
        elif args.progressive_training and epoch >= args.progressive_epochs:
            if mixup_fn is not None:
                mixup_fn.mixup_alpha = args.mixup * 0.5
                mixup_fn.cutmix_alpha = args.cutmix * 0.5

        # 训练一个epoch
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_criterion, optimizer, data_loader, device, epoch, args.print_freq, scaler=scaler,
            T_train=args.T_train, mixup_fn=mixup_fn, clip_grad=args.clip_grad
        )

        # 更新学习率
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.SequentialLR):
            lr_scheduler.step()
        else:
            lr_scheduler.step(epoch + 1)

        # 评估模型

        test_loss, test_acc1, test_acc5 = evaluate(
            model, eval_criterion, data_loader_test, device, args.print_freq
        )

        # 记录到TensorBoard
        if args.tb:
            writer.add_scalar('train/loss', train_loss, epoch)
            if train_acc1 > 0:
                writer.add_scalar('train/acc1', train_acc1, epoch)
                writer.add_scalar('train/acc5', train_acc5, epoch)
            writer.add_scalar('test/loss', test_loss, epoch)
            writer.add_scalar('test/acc1', test_acc1, epoch)
            writer.add_scalar('test/acc5', test_acc5, epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # 保存检查点
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'args': args,
        }

        if scaler is not None:
            checkpoint['scaler'] = scaler.state_dict()

        save_on_master(
            checkpoint,
            os.path.join(args.output_dir, f'checkpoint_latest.pth')
        )

        # 保存最佳模型
        if test_acc1 > best_acc:
            best_acc = test_acc1
            checkpoint['best_acc'] = best_acc
            save_on_master(
                checkpoint,
                os.path.join(args.output_dir, f'checkpoint_best.pth')
            )
            print(f" 新的最佳模型！准确率: {best_acc:.2f}%")

            # 打印当前训练状态
        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"训练损失: {train_loss:.4f}, "
              f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc1:.2f}%, "
              f"最佳准确率: {best_acc:.2f}%, "
              f"学习率: {optimizer.param_groups[0]['lr']:.2e}")

    # 训练结束
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\n🏁 训练完成！')
    print(f'总时间: {total_time_str}')
    print(f'最佳测试准确率: {best_acc:.2f}%')

    # 关闭TensorBoard
    if args.tb:
        writer.close()


if __name__ == '__main__':
    import math
    import sys

    args = parse_args()
    main(args)
    print("🎊 训练完成！")


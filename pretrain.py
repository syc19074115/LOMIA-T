#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
from functools import partial
import math
import os
import random
import shutil
import time
import warnings

import pretrain_moco.builder
import pretrain_moco.loader
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from main_model.LOMIA_T import LOMIA_T
from torchvision.models import swin_b, Swin_B_Weights

# import config
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1, 3]))

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--data", default='./', metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
) 
parser.add_argument(
    "-j",
    "--workers",
    default=40,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=1000, type=int, metavar="N", help="number of total epochs to run"
)

parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=60, #ESCC--64
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=0, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://localhost:10001",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=42, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    default=True,
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# moco specific configs:
parser.add_argument(
    "--moco-dim", default=576, type=int, help="feature dimension (default: 128)" #ESCC--128 
)
parser.add_argument(
    "-k",
    "--moco-k",
    default=0, #ESCC--256
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)
parser.add_argument('--mode', type=str, default='only_post')

# options for moco v2
parser.add_argument("--mlp", action="store_true", help="use mlp head")
parser.add_argument(
    "--aug-plus", action="store_true", help="use moco v2 data augmentation"
)
parser.add_argument("--cos", default=True ,action="store_true", help="use cosine lr schedule")
parser.add_argument('--dataset_path',default = './dataset',help='fixed trainset root path')

#DropKey
parser.add_argument('--DropKey', type=bool, default=False, help='use DropKey')
parser.add_argument('--mask_ratio', type=float, default=0., help='ratio of dropkey')

#pre and post
parser.add_argument('--type', type=str, default="post", help='pre or post')


def main():
    args = parser.parse_args()
    print(args.type)

    if args.seed is not None:
        torch.cuda.empty_cache()
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    # args1 = config.args
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    # print(args.batch_size)
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    # print("dad12313123213123a",args.workers)
    net = LOMIA_T(image_size=32, dim=128, channels=1, time_emb="One", TRRN_depth=7, TRRN_dropKey=0., use_DropKey=False,mode="pre_post",num_classes=5, data_type="ESCC") 
    
    print("=> creating model pretrain_cross")
    model = pretrain_moco.builder.MoCo(
        #models.__dict__[args.arch],
        net,
        args.moco_dim,
        args.moco_k,
        args.moco_m,
        args.moco_t,
        args.mlp,
    )
   # print(model)  #到目前为止没有什么错误，打印出来的模型是对的

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu],
                find_unused_parameters=True
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.003) #0.03

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args.data, "train")

    from dataset.ESCC import train_dataset_moco
    train_ds = train_dataset_moco.Train_Dataset(args,args.type)
    

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        train_sampler = None

    # print(args.workers,)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        #shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    #选loss最小的那个存
    loss_total = 1000000

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss_sum = train(train_loader, model, criterion, optimizer, epoch, args)

        print(loss_total)
        if epoch != 0:
            if loss_sum <= loss_total:
                loss_total = loss_sum
                save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "net": model.module.encoder_q.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                filename="".format(args.type),
                is_best = False
                )

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            #print("1111111111",args.multiprocessing_distributed,args.rank,ngpus_per_node)
            """ save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename="checkpoint_{:04d}.pth.tar".format(epoch),
            ) """


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.6f")

    #losses = AverageMeter("Loss", ":.4e")
    #top1 = AverageMeter("Acc@1", ":6.2f")
    #top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        #[batch_time, data_time, losses, top1, top5],
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    
    loss_sum=0

    end = time.time()
    for i, (q, k) in enumerate(train_loader):
        # measure data loading time
        #print(images[0].shape)
        data_time.update(time.time() - end)

        """ if i == 0:
            for name, parameters in model.named_parameters():
                if 'module.encoder_q.3.mlp.fc2.weight' in name:
	                print(name, ':', parameters) """

        if args.gpu is not None:
            """ images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True) """
            q = q.cuda(args.gpu, non_blocking=True)
            k = k.cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=q, im_k=k)
        loss = criterion(output, target)

        loss_sum += loss.item()
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #losses.update(loss.item(), images[0].size(0))
        losses.update(loss.item(), q.size(0))

        #top1.update(acc1[0], images[0].size(0))
        #top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(retain_graph = True)

        """ if i == 0:
            for name, parameters in model.named_parameters():
                if 'module.encoder_q.3.mlp.fc2.weight' in name:
	                print(name, '_back:', parameters.grad) """

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return loss_sum


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    warmup_factor = args.lr
    warm_lr = []
    for iter in range(10):
        alpha = (iter+10)/10
        warmup_factor_temp = warmup_factor**alpha
        warm_lr.append(warmup_factor_temp)
    #print(warm_lr)

    if args.cos:  # cosine lr schedule
        if epoch < 10 :
            lr = warm_lr[9-epoch]
        else:
            lr *= 0.5 * (1.0 + math.cos(math.pi * (epoch-10) / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        print("lr  ",lr)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()

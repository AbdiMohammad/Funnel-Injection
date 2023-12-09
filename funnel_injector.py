import random
import numpy as np
import os

import torch
from torch import nn
import torch.nn.utils.prune
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

import cifar_resnet, cifar_resnet_tiny

from trainer_ddp import Trainer

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.barrier()

def main():
    # Reproducibility
    random.seed(1994)
    np.random.seed(1994)
    torch.manual_seed(1994)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ddp_setup()

    # Get dataset
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    train_ds = datasets.CIFAR10('data', train=True, transform=train_transform, download=True)
    val_ds = datasets.CIFAR10('data', train=False, transform=val_transform, download=False)

    # Get model
    model = cifar_resnet_tiny.resnet56(num_classes=10)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Get data loader
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=128,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_ds)

    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=128,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(val_ds, shuffle=False)
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 150, 180])
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler, 10, 'run', 15)

    trainer.train_and_validate(200)

    destroy_process_group()


if __name__ == "__main__":
    main()
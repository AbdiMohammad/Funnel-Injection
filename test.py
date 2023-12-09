import random
import numpy as np
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune

import cifar_resnet, cifar_resnet_tiny
from trainer import Trainer

import partial_freezing

def main():
    # Reproducibility
    random.seed(1994)
    np.random.seed(1994)
    torch.manual_seed(1994)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    # Get data loader
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=4
    )

    # Get model
    model = cifar_resnet_tiny.resnet56(num_classes=10)

    # Load best pretrained weights
    checkpoint = torch.load("run/best_20230731_105505.pt")
    model.load_state_dict(checkpoint['model_state'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 150, 180])
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    prune_ratios = {}
    for basicBlock_idx in range(9):
        for conv_idx in range(2):
            prune_ratios[model.get_submodule(f'layer1.{basicBlock_idx}.conv{conv_idx + 1}')] = 0.0

    weight_hook_handle = {}
    bias_hook_handle = {}

    iterative_steps = 100
    for idx, prune_module in enumerate(prune_ratios.keys()):
        prune_ratios[prune_module] = (idx + 1) / len(prune_ratios)
        weight_hook_handle[prune_module] = None
        bias_hook_handle[prune_module] = None


    for iterative_step in range(iterative_steps - 1, iterative_steps):
        last_model = model
        for prune_module, prune_ratio in prune_ratios.items():
            prune.ln_structured(prune_module, name="weight", amount=(prune_ratio * iterative_step / iterative_steps), n=2, dim=0)
            # prune_indices = (torch.norm(prune_module.get_buffer('weight_mask'), p=2, dim=(1, 2, 3)) == 0).nonzero(as_tuple=True)[0]
            # if prune_indices.numel():
            #     weight_hook_handle, bias_hook_handle = partial_freezing.freeze_conv2d_params(prune_module, prune_indices, weight_hook_handle=weight_hook_handle, bias_hook_handle=bias_hook_handle)
            prune.remove(prune_module, 'weight')
            prune_indices = (torch.norm(prune_module.weight, p=2, dim=(1, 2, 3)) == 0).nonzero(as_tuple=True)[0]
            if prune_indices.numel():
                weight_hook_handle[prune_module], bias_hook_handle[prune_module] = partial_freezing.freeze_conv2d_params(prune_module, prune_indices, weight_hook_handle=weight_hook_handle[prune_module], bias_hook_handle=bias_hook_handle[prune_module])
        
        trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler)
        terminate_epoch = trainer.train_and_validate(100, checkpoint['val_acc'])
        if terminate_epoch == 100:
            model = last_model
            break
        # for name, module in model.named_modules():
        #     if isinstance(module, torch.nn.modules.conv._ConvNd):
        #         print(f'{name}\t{torch.sum(torch.norm(module.weight, p=2, dim=(1, 2, 3)) == 0) / module.weight.shape[0]}')

    print(iterative_step)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.conv._ConvNd):
            print(f'{name}\t{torch.sum(torch.norm(module.weight, p=2, dim=(1, 2, 3)) == 0) / module.weight.shape[0]}')

    torch.save(model.state_dict(), f'{os.path.join("run", "pruned_resnet56.pt")}')
    
    return
    # prune.ln_structured(module, name='weight', amount=0.2, n=2, dim=0)
    # print(module.weight.shape)
    # print(module.weight)

    


    # module = model.get_submodule('layer1.0.bn1')
    # print(module.weight.shape)
    # print(module.weight)
    # print(module.bias.shape)
    # print(module.bias)

    # with torch.no_grad():
    #     module.weight.index_fill_(0, torch.tensor(np.arange(5)), 1.0)
    #     module.bias.index_fill_(0, torch.tensor(np.arange(5)), 0.0)

    # print(module.weight.shape)
    # print(module.weight)
    # print(module.bias.shape)
    # print(module.bias)

    # return


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler, 50, "run", 20)

    trainer.train_and_validate(200)

    # module = model.conv1

    # print(module.weight.shape)
    # for i in range(10):
    #     prune.ln_structured(module, 'weight', amount=0.1, dim=0, n=2)
    #     prune.remove(module, 'weight')
    #     print(torch.sum(torch.norm(module.weight, p=2, dim=(1, 2, 3)) == 0))
    #     print(torch.sum(torch.norm(module.weight, p=2, dim=(1, 2, 3)) == 0) / module.weight.shape[0])

if __name__ == "__main__":
    main()
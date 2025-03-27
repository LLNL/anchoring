# Copyright 2024 Lawrence Livermore National Security, LLC and other
# Authors: Vivek Sivaraman Narayanaswamy. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

'''Train CIFAR10/100 with Anchoring.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset

import sys
import numpy as np
from sklearn.model_selection import train_test_split

import datetime
import os
import argparse
from shutil import copyfile
import random

from wrapper import AnchoringWrapper
from utils.models import *

def encode(inps, anchs):
    return inps - anchs

def return_concatenated_anchorset(dataset1, dataset2):
    combined_dataset = ConcatDataset([dataset1, dataset2])
    print(f'Dataset Size = {len(combined_dataset)}')
    return combined_dataset

def return_anchorset(dataset, num_classes, K):
    selected_classes = torch.randperm(num_classes)[:K]
    print(f'Selected classes = {selected_classes}')
    subset_dataset = Subset(dataset, [i for i in range(len(dataset)) if dataset.targets[i] in selected_classes])
    print(f'Dataset Size = {len(subset_dataset)}')
    return subset_dataset

def return_anchorset_random_samples(dataset, num_samples): 
    sample_indices = random.sample(range(len(dataset)), num_samples)
    print(f'Selected samples indices = {sample_indices}')
    subset_dataset = Subset(dataset, sample_indices)
    print(f'Dataset Size = {len(subset_dataset)}')
    return subset_dataset

def return_anchorset_stratified(dataset, num_classes, K, num_samples, seed):
    selected_classes = torch.randperm(num_classes)[:K]
    print(f'Selected classes = {selected_classes}')
    idx = [i for i in range(len(dataset)) if dataset.targets[i] in selected_classes]
    l = [dataset.targets[i] for i in range(len(dataset)) if dataset.targets[i] in selected_classes]
    _, idx_stratified, _, _ = train_test_split(idx, l, test_size=num_samples,  stratify=l, shuffle=True, random_state=seed)
    subset_dataset = Subset(dataset, idx_stratified)
    print(f'Dataset Size = {len(subset_dataset)}')
    print(f'Selected samples indices = {idx_stratified}')
    return subset_dataset

def run_training(datasettype='cifar10',modeltype='resnet18',alpha=0.25,seed=1, subselect_classes=1, epochs=200):
    
    # Seting random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Setting up training parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_epochs = epochs
    best_acc = 0  # Initializing best test accuracy
    start_epoch = 0
    crop = 0
    imsize = 32  # CIFAR10/100 image size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data Transform
    transform_train = transforms.Compose([
                    transforms.RandomCrop(imsize, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Data
    if datasettype=='cifar10':
        nclass = 10
        print('==> Preparing CIFAR10 data..')
        
        trainset = torchvision.datasets.CIFAR10(root='../data/CIFAR10/', train=True, download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='../data/CIFAR10/', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

        anchorset = return_anchorset(torchvision.datasets.CIFAR10(root='../data/CIFAR10/', train=True, download=False, transform=transform_test), num_classes=nclass, K=subselect_classes)
        anchorloader = torch.utils.data.DataLoader(anchorset, batch_size=128, shuffle=True, num_workers=2)
        print('Loaded Train, Val, Anchor Loaders')

        
    elif datasettype =='cifar100':
        nclass = 100
        print('==> Preparing CIFAR100 data..')
        trainset = torchvision.datasets.CIFAR100(root='../data/CIFAR100/', train=True, download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='../data/CIFAR100/', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

        anchorset = return_anchorset(torchvision.datasets.CIFAR100(root='../data/CIFAR100/', train=True, download=False, transform=transform_test), num_classes=nclass, K=subselect_classes)
        anchorloader = torch.utils.data.DataLoader(anchorset, batch_size=128, shuffle=True, num_workers=2)
        print('Loaded Train, Val, Anchor Loaders')

    print('==> Building model..')
    if modeltype=='resnet18':
        model = ResNet18(nc=3,num_classes=nclass)
        modelname = 'ResNet18'
        model.conv1 = nn.Conv2d(6, 64, kernel_size=3,stride=1, padding=1, bias=False) # Modifying the first layer to accommodate twice the number of channels for anchoring
    else:
        raise NotImplementedError    

    # Wrap the model to support anchoring functionality
    net = AnchoringWrapper(model,crop=crop,imsize=imsize)
    
    # Setting checkpoint path
    modelname = modeltype + f'_seed_{seed}'
    logname = f'alpha_{str(alpha)}_anchor_size_{subselect_classes}'
    print(logname)
    modelpath = f'chkpts/{datasettype}/{modelname}/{logname}/'
    if not os.path.isdir(modelpath):
        os.makedirs(modelpath)
    
    # Model settings
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Loss function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones= [60, 120, 160], gamma=0.2) #learning rate decay
    
    # Train Function
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        iterator = iter(anchorloader)

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            try:
                anchors, _ = next(iterator)
                anchors = anchors.to(device)
            except StopIteration:
                iterator = iter(anchorloader)
                anchors, _ = next(iterator)
                anchors = anchors.to(device)

            optimizer.zero_grad()
            corrupt = bool(torch.bernoulli(torch.tensor(alpha))==1)  # Making corrupt==True for alpha % of times during training
            outputs = net(inputs,anchors=anchors, corrupt=corrupt)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx %100==0:
                print(f'Epoch# {epoch}, Batch# {batch_idx}, Loss: {train_loss/(batch_idx+1):.3f}, Acc:{100.*correct/total:.3f}:')

    def test(epoch,best_acc):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print(f'********** Epoch {epoch} of {max_epochs} -- Test Loss: {test_loss/(batch_idx+1):.3f}, Test Acc: {100.*correct/total:.3f} **********')

        # Save checkpoint.
        acc = 100.*correct/total
        if epoch > (max_epochs-5):
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }

            torch.save(state, f'{modelpath}/ckpt-{epoch}.pth')
            best_acc = acc

        return best_acc

    for epoch in range(start_epoch, start_epoch+max_epochs):
        train(epoch)
        best_acc = test(epoch,best_acc)
        scheduler.step()
    return best_acc

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CIFAR10 with PyTorch Training')
    parser.add_argument('--modeltype', default="resnet18", type=str,
                        help='neural network name and training set')
    parser.add_argument('--seed', default=1, type=int,
                        help='model training seed')
    parser.add_argument('--alpha',default=0.25,type=float, help='Alpha for controlling masking regularization')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--subselect_classes', default=1, type=int,help='Sub-selecting classes for anchor distribution')
    parser.add_argument('--epochs', default=200, type=int, help='Max. no of epochs')

    args = parser.parse_args()
    accs = run_training(datasettype=args.dataset,modeltype=args.modeltype,seed=args.seed,alpha=args.alpha, subselect_classes=args.subselect_classes, epochs=args.epochs)

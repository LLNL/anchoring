# Copyright 2024 Lawrence Livermore National Security, LLC and other
# Authors: Vivek Sivaraman Narayanaswamy. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch import linalg as LA
from torchmetrics.classification import MulticlassCalibrationError
import relplot

from wrapper import AnchoringWrapperInference
from utils.models import *

import sys
import os
import argparse
import random
import yaml
import datetime
import sys
import logging
import time

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

class CIFAR_Corruptions(torch.utils.data.Dataset):
    def __init__(self, root_dir, corruption='gaussian_blur', transform=None,level=0):
        numpy_path = f'{root_dir}/{corruption}.npy'
        t = 10000
        self.transform = transform
        self.data_ = np.load(numpy_path)[level*10000:(level+1)*10000,:,:,:]
        self.data = self.data_[:t,:,:,:]
        self.targets_ = np.load(f'{root_dir}/labels.npy')
        self.targets = self.targets_[:t]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx,:,:,:]
        if self.transform:
            image = self.transform(image)
        targets = self.targets[idx]
        return image, targets
    
def compute_anchoring_accuracy(args):
    # Initializing parameters
    modeltype = args.modeltype
    seed = args.seed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    severity = args.severity
    cifar10c_corruption = args.cifar10c_corruption
    dataset = args.dataset
    log_path = args.log_path
    subselect_classes = args.subselect_classes
    root_dir = args.cifar10c_data_path
    ckpt_path = args.ckpt_path

    # Setting seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Logging parameters
    log_path = f'{log_path}/{dataset}/{modeltype}/{str(seed)}'
    logfile = f'{log_path}/{args.filename}.log' 
    os.makedirs(log_path,exist_ok = True)
    loglevel = logging.INFO
    logging.basicConfig(level=loglevel,filename=logfile, filemode='a', format='%(levelname)s - %(message)s')
    logger = logging.getLogger()

    # Define test transform
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':
        nclass = 10
        
        # Define the same anchor set used while training
        anchorset = return_anchorset(torchvision.datasets.CIFAR10(root='../data/CIFAR10/', train=True, download=False, transform=transform_test), num_classes=nclass, K=subselect_classes)
        anchorloader = torch.utils.data.DataLoader(anchorset, batch_size=128, shuffle=True, num_workers=2)
        
        if (args.eval_dataset == 'cifar10c') | (args.eval_dataset == 'cifar10cbar'):
            testset = CIFAR_Corruptions(root_dir=root_dir, corruption=cifar10c_corruption, transform=transform_test,level=severity)
            testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=2)
        elif args.eval_dataset == 'clean':
            testset = torchvision.datasets.CIFAR10(root='../data/CIFAR10/', train=False, download=False, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=2)
        else:
            raise NotImplementedError
        print('Loaded Anchor, Test Loaders')


    elif dataset == 'cifar100':
        nclass = 100
        anchorset = return_anchorset(torchvision.datasets.CIFAR100(root='../data/CIFAR100/', train=True, download=False, transform=transform_test), num_classes=nclass, K=subselect_classes)
        anchorloader = torch.utils.data.DataLoader(anchorset, batch_size=128, shuffle=True, num_workers=2)
        print('Loaded Anchor, Test Loaders')
        
        if (args.eval_dataset == 'cifar100c') (args.eval_dataset == 'cifar100cbar'):
            testset = CIFAR_Corruptions(root_dir=root_dir, corruption=cifar10c_corruption, transform=transform_test,level=severity)
            testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=2)
        elif args.eval_dataset == 'clean':
            testset = torchvision.datasets.CIFAR100(root='../data/CIFAR100/', train=False, download=False, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=2)
        else:
            raise NotImplementedError
        print('Loaded Anchor, Test Loaders')

    # Load Model and Checkpoints
    if modeltype=='resnet18':
        model = ResNet18(nc=6,num_classes=nclass)
        modelname = 'ResNet18'
    else:
        raise NotImplementedError
    
    # Loading model checkpoints
    print('Loading model from', ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Wrapper for performing inference with anchored models
    net = AnchoringWrapperInference(model)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(checkpoint['net'])
    print(f'Loaded {dataset} Checkpoint')
    
    net.to(device)
    best_acc = checkpoint['acc']
    logger.info(f'Checkpoint loaded from {ckpt_path} with test accuracy (with 1 anchor) --- {best_acc:.4f}')
    mcce = MulticlassCalibrationError(num_classes=nclass, n_bins=20, norm='l1')
    net.eval()
    correct = 0
    total = 0
    preds_cifar= []
    T = []
    n_ref = 1
    iterator = iter(anchorloader)
    anchors, _ = next(iterator)
    anchors = anchors.to(device)
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            mu = net(inputs, anchors=anchors, n_anchors=n_ref)
            T.append(targets)
            preds_cifar.append(mu)
            print(f'Processed {i}/{len(testloader)}')
        mean_preds = torch.cat(preds_cifar,0)
        all_targets = torch.cat(T,0)
        _, predicted = mean_preds.max(1)
        correct = predicted.eq(all_targets).sum().item()/all_targets.shape[0]
        acc = 100*correct
        ece_error = mcce(mean_preds, all_targets).item()
        conf, acc1 = relplot.multiclass_logits_to_confidences(mean_preds.cpu().data.numpy(), all_targets.cpu().data.numpy())
        smoothed_ece = relplot.smECE(f=conf, y=acc1)
    if args.eval_dataset == 'cifar10c':
        logger.info(f'CIFAR10-C - Corruption {cifar10c_corruption}, Severity {severity} with Anchoring - Test accuracy with {n_ref} anchors --- {acc:.4f}')
        logger.info(f'CIFAR10-C - Corruption {cifar10c_corruption}, Severity {severity} with Anchoring - RMS ECE with {n_ref} anchors --- {ece_error:.4f}')
        logger.info(f'CIFAR10-C - Corruption {cifar10c_corruption}, Severity {severity} with Anchoring - Smoothed ECE with {n_ref} anchors --- {smoothed_ece:.4f}')
    elif args.eval_dataset == 'cifar10cbar':
        logger.info(f'CIFAR10-CBar - Corruption {cifar10c_corruption}, Severity {severity} with Anchoring - Test accuracy with {n_ref} anchors --- {acc:.4f}')
        logger.info(f'CIFAR10-CBar - Corruption {cifar10c_corruption}, Severity {severity} with Anchoring - RMS ECE with {n_ref} anchors --- {ece_error:.4f}')
        logger.info(f'CIFAR10-CBar - Corruption {cifar10c_corruption}, Severity {severity} with Anchoring - Smoothed ECE with {n_ref} anchors --- {smoothed_ece:.4f}')
    elif args.eval_dataset == 'cifar100c':
        logger.info(f'CIFAR100-C - Corruption {cifar10c_corruption}, Severity {severity} with Anchoring - Test accuracy with {n_ref} anchors --- {acc:.4f}')
        logger.info(f'CIFAR100-C - Corruption {cifar10c_corruption}, Severity {severity} with Anchoring - RMS ECE with {n_ref} anchors --- {ece_error:.4f}')
        logger.info(f'CIFAR100-C - Corruption {cifar10c_corruption}, Severity {severity} with Anchoring - Smoothed ECE with {n_ref} anchors --- {smoothed_ece:.4f}')
    elif args.eval_dataset == 'cifar100cbar':
        logger.info(f'CIFAR100-Cbar - Corruption {cifar10c_corruption}, Severity {severity} with Anchoring - Test accuracy with {n_ref} anchors --- {acc:.4f}')
        logger.info(f'CIFAR100-Cbar - Corruption {cifar10c_corruption}, Severity {severity} with Anchoring - RMS ECE with {n_ref} anchors --- {ece_error:.4f}')
        logger.info(f'CIFAR100-Cbar - Corruption {cifar10c_corruption}, Severity {severity} with Anchoring - Smoothed ECE with {n_ref} anchors --- {smoothed_ece:.4f}')
    elif args.eval_dataset == 'clean' and dataset == 'cifar10':
        logger.info(f'CIFAR10 Clean with with Anchoring - Test accuracy with {n_ref} anchors --- {acc:.4f}')
        logger.info(f'CIFAR10 Clean with with Anchoring - RMS ECE with {n_ref} anchors --- {ece_error:.4f}')
        logger.info(f'CIFAR10 Clean with with Anchoring - Smoothed ECE with {n_ref} anchors --- {smoothed_ece:.4f}')
    elif args.eval_dataset == 'clean' and dataset == 'cifar100':
        logger.info(f'CIFAR100 Clean with with Anchoring - Test accuracy with {n_ref} anchors --- {acc:.4f}')
        logger.info(f'CIFAR100 Clean with with Anchoring - RMS ECE with {n_ref} anchors --- {ece_error:.4f}')
        logger.info(f'CIFAR100 Clean with with Anchoring - Smoothed ECE with {n_ref} anchors --- {smoothed_ece:.4f}')
    
    # Saving 
    if (args.dataset == 'cifar10' and args.eval_dataset == 'cifar10c') | (args.dataset == 'cifar100' and args.eval_dataset == 'cifar100c') | (args.dataset == 'cifar10' and args.eval_dataset == 'cifar10cbar') | (args.dataset == 'cifar100' and args.eval_dataset == 'cifar100cbar'):
        results_dict[args.cifar10c_corruption][args.severity]['logits'] = mean_preds.cpu().data.numpy()
        results_dict[args.cifar10c_corruption][args.severity]['targets'] = all_targets.cpu().data.numpy()


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Anchoring Inference')
    parser.add_argument('--modeltype', default="resnet18", type=str,
                        help='neural network name and training set')
    parser.add_argument('--seed', default=1, type=int,
                        help='model training seed')
    parser.add_argument('--dataset', default="cifar10", type=str,
                        help='in-distribution dataset')
    parser.add_argument('--eval_dataset', default="clean", type=str,
                        help='Eval dataset')
    parser.add_argument('--cifar10c_corruption', default="gaussian_blur", type=str,
                        help='Corruption type')
    parser.add_argument('--cifar10c_data_path', default="./data/CIFAR-10C/", type=str,
                        help='CIFAR10W data path')
    parser.add_argument('--severity', default=5, type=int,
                        help='Severity of corruption')
    parser.add_argument('--filename', default="blt", type=str,
                        help='CIFAR10C_with_n_train_anchors_1_ckpt.log')
    parser.add_argument('--log_path', default="./logs", type=str,
                        help='Absolute path of logs')
    parser.add_argument('--ckpt_path', default="blt", type=str,
                        help='Absolute path of the checkpoint')
    parser.add_argument('--method', default="anchoring", type=str,
                        help='Choice [anchoring]')
    parser.add_argument('--subselect_classes', default=1, type=int,help='Sub-selecting classes for anchor distribution')

    parser.set_defaults(argument=True)
    args = parser.parse_args()

    results_dict = {}
    corruption_list = ['brightness', 'defocus_blur', 'fog', 'gaussian_blur', 'glass_blur', 'jpeg_compression', 'motion_blur', 'saturate','snow','speckle_noise', 'contrast', 'elastic_transform', 'frost', 'gaussian_noise', 'impulse_noise', 'pixelate','shot_noise', 'spatter','zoom_blur']
    severity = [0,1,2,3,4]
    
    if args.method == 'anchoring':
        if (args.dataset == 'cifar10' and args.eval_dataset == 'cifar10c') | (args.dataset == 'cifar100' and args.eval_dataset == 'cifar100c'):
            for c in corruption_list:
                results_dict[c]={}
                for s in severity:
                    args.cifar10c_corruption = c
                    args.severity = s
                    print(f'Corruption = {args.cifar10c_corruption}, Severity = {args.severity}')
                    results_dict[c][s]={}
                    results_dict[c][s]['logits'] = []
                    results_dict[c][s]['targets'] = []
                    compute_anchoring_accuracy(args)
        elif (args.dataset == 'cifar10' and args.eval_dataset == 'cifar10cbar') | (args.dataset == 'cifar100' and args.eval_dataset == 'cifar100cbar'):
            corruption_list = ['blue_noise_sample', 'checkerboard_cutout', 'inverse_sparkles', 'pinch_and_twirl', 'ripple', 'brownish_noise', 'circular_motion_blur', 'lines', 'sparkles', 'transverse_chromatic_abberation']
            severity = [0,1,2,3,4]
            for c in corruption_list:
                for s in severity:
                    args.cifar10c_corruption = c
                    args.severity = s
                    print(f'Corruption = {args.cifar10c_corruption}, Severity = {args.severity}')
                    compute_anchoring_accuracy(args)

        elif (args.dataset == 'cifar10' and args.eval_dataset == 'clean') | (args.dataset == 'cifar100' and args.eval_dataset == 'clean'):
            compute_anchoring_accuracy(args)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


    # Saving the logits and true targets
    # Saving 
    if (args.dataset == 'cifar10' and args.eval_dataset == 'cifar10c') | (args.dataset == 'cifar100' and args.eval_dataset == 'cifar100c')| (args.dataset == 'cifar10' and args.eval_dataset == 'cifar10cbar') | (args.dataset == 'cifar100' and args.eval_dataset == 'cifar100cbar'):
        import pickle
        import os
        os.makedirs(f'{args.log_path}/{args.dataset}/{args.modeltype}/{args.filename}', exist_ok=True)
        with open(f'{args.log_path}/{args.dataset}/{args.modeltype}/{args.filename}/logits_y.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
        f.close()








    











        


    

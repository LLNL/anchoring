import datetime
import os
import time
import warnings
import sys
sys.path.append('../')

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from torchvision.models import resnet18, ResNet18_Weights, RegNet_Y_800MF_Weights
from torchmetrics.classification import MulticlassCalibrationError
from imagenet_r_ids import imagenet_r_mask, imagenet_a_mask
from wrapper import AnchoringWrapperInference
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def auroc_ood(values_in: np.ndarray, values_out: np.ndarray) -> float:
    """
    Implementation of Area-under-Curve metric for out-of-distribution detection.
    The higher the value the better.

    Args:
        values_in: Maximal confidences (i.e. maximum probability per each sample)
            for in-domain data.
        values_out: Maximal confidences (i.e. maximum probability per each sample)
            for out-of-domain data.

    Returns:
        Area-under-curve score.
    """
    if len(values_in) * len(values_out) == 0:
        return np.NAN
    y_true = len(values_in) * [1] + len(values_out) * [0]
    y_score = np.nan_to_num(np.concatenate([values_in, values_out]).flatten())
    return roc_auc_score(y_true, y_score)


def fpr_at_tpr(values_in: np.ndarray, values_out: np.ndarray, tpr: float) -> float:
    """
    Calculates the FPR at a particular TRP for out-of-distribution detection.
    The lower the value the better.

    Args:
        values_in: Maximal confidences (i.e. maximum probability per each sample)
            for in-domain data.
        values_out: Maximal confidences (i.e. maximum probability per each sample)
            for out-of-domain data.
        tpr: (1 - true positive rate), for which probability threshold is calculated for
            in-domain data.

    Returns:
        False positive rate on out-of-domain data at (1 - tpr) threshold.
    """
    if len(values_in) * len(values_out) == 0:
        return np.NAN
    t = np.quantile(values_in, (1 - tpr))
    fpr = (values_out >= t).mean()
    return fpr


def evaluate(model, criterion, data_loader, mcce, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    final_logit_list = []
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if args.dset == 'imagenet-r':
                output = model(image)[:, imagenet_r_mask]
            elif args.dset == 'imagenet-a':
                output = model(image)[:, imagenet_a_mask]
            else:
                output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            ece = mcce(output, target)
            metric_logger.meters["ece"].update(ece.item(), n=batch_size)
            smoothed_ece = utils.compute_smoothed_ece(output, target)
            metric_logger.meters["smoothed_ece"].update(smoothed_ece.item(), n=batch_size)

            all_output = [torch.zeros_like(output) for _ in range(args.world_size)]
            torch.distributed.all_gather(all_output, output)
            final_logit_list.append(torch.cat(all_output))

            num_processed_samples += batch_size
    

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    print(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} ECE {metric_logger.ece.global_avg:.3f} Smoothed ECE {metric_logger.smoothed_ece.global_avg:.3f}")
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, metric_logger.ece.global_avg, metric_logger.smoothed_ece.global_avg, torch.cat(final_logit_list,0)

def evaluate_anchoring(model, criterion, data_loader_test, data_loader_train, mcce, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"
    anchors, _ = next(iter(data_loader_train))
    anchors = anchors.to(device, non_blocking=True)
    num_processed_samples = 0

    final_logit_list = []
    final_std_list = []
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader_test, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if args.dset == 'imagenet-r':
                output, std = model(image, anchors=anchors, n_anchors=5, return_std=True)
                output, std = output[:, imagenet_r_mask], std[:, imagenet_r_mask]
            elif args.dset == 'imagenet-a':
                output, std = model(image, anchors=anchors, n_anchors=5, return_std=True)
                output, std = output[:, imagenet_a_mask], std[:, imagenet_a_mask]
            else:
                output, std = model(image, anchors=anchors, n_anchors=5, return_std=True)

            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            ece = mcce(output, target)
            metric_logger.meters["ece"].update(ece.item(), n=batch_size)
            smoothed_ece = utils.compute_smoothed_ece(output, target)
            metric_logger.meters["smoothed_ece"].update(smoothed_ece.item(), n=batch_size)

            all_output = [torch.zeros_like(output) for _ in range(args.world_size)]
            torch.distributed.all_gather(all_output, output)
            final_logit_list.append(torch.cat(all_output))

            all_std = [torch.zeros_like(std) for _ in range(args.world_size)]
            torch.distributed.all_gather(all_std, std)
            final_std_list.append(torch.cat(all_std))
            
            num_processed_samples += batch_size
    
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    print(num_processed_samples)
    if (
        hasattr(data_loader_test.dataset, "__len__")
        and len(data_loader_test.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader_test.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} ECE {metric_logger.ece.global_avg:.3f} Smoothed ECE {metric_logger.smoothed_ece.global_avg:.3f}")
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, metric_logger.ece.global_avg, metric_logger.smoothed_ece.global_avg, torch.cat(final_logit_list,0), torch.cat(final_std_list,0)



def load_data(valdir, args):
    # Data loading code
    print("Loading Validation data")
    val_resize_size, val_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    preprocessing = presets.ClassificationPresetEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
        interpolation=interpolation,
        backend=args.backend,
        use_v2=args.use_v2,
    )

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        preprocessing,
    )

    print("Creating data loaders")
    if args.distributed:
        print('Distributed data loaders')
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_test, test_sampler


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    args.filename = args.ckpt_name+'.log'

    device = torch.device(args.device)
    log_path = f'{args.output_dir}/{args.dset}/{args.model}/'
    logfile = f'{log_path}/{args.filename}'

    if not os.path.exists(log_path):
        os.makedirs(log_path,exist_ok = True)
    
    loglevel = logging.INFO
    logging.basicConfig(level=loglevel,filename=logfile, filemode='a', format='%(levelname)s - %(message)s')
    logger = logging.getLogger()
    
    print("Loading model")
    
    if args.model == 'anchored_vit_b_16':
        model = torchvision.models.vit_b_16()
        model.conv_proj= torch.nn.Conv2d(6, 768, kernel_size=(16, 16), stride=(16, 16))
        model = AnchoringWrapperInference(model)
        state_dict = torch.load(f'../ckpts/{args.model}/{args.ckpt_name}.pth', map_location=device)['model_ema']
        del state_dict['n_averaged']
        new_state_dict = {}
        for key, val in state_dict.items():
            new_state_dict[key[7:]] = val
        model.load_state_dict(new_state_dict)
        logger.info(f'Anchored model loaded from ../ckpts/{args.model}/{args.ckpt_name}.pth')
    elif args.model == 'anchored_swin_v2_t':
        model = torchvision.models.swin_v2_t()
        model.features[0][0]= torch.nn.Conv2d(6, 96, kernel_size=(4, 4), stride=(4, 4))
        model = AnchoringWrapperInference(model)
        state_dict = torch.load(f'../ckpts/{args.model}/{args.ckpt_name}.pth', map_location=device)['model_ema']
        del state_dict['n_averaged']
        new_state_dict = {}
        for key, val in state_dict.items():
            new_state_dict[key[7:]] = val
        model.load_state_dict(new_state_dict)
        logger.info(f'Anchored model loaded from ../ckpts/{args.model}/{args.ckpt_name}.pth')
    elif args.model == 'anchored_swin_v2_s':
        model = torchvision.models.swin_v2_s()
        model.features[0][0]= torch.nn.Conv2d(6, 96, kernel_size=(4, 4), stride=(4, 4))
        model = AnchoringWrapperInference(model)
        state_dict = torch.load(f'../ckpts/{args.model}/{args.ckpt_name}.pth', map_location=device)['model_ema']
        del state_dict['n_averaged']
        new_state_dict = {}
        for key, val in state_dict.items():
            new_state_dict[key[7:]] = val
        model.load_state_dict(new_state_dict)
        logger.info(f'Anchored model loaded from ../ckpts/{args.model}/{args.ckpt_name}.pth')
    elif args.model == 'anchored_swin_v2_b':
        model = torchvision.models.swin_v2_b()
        model.features[0][0]= torch.nn.Conv2d(6, 128, kernel_size=(4, 4), stride=(4, 4))
        model = AnchoringWrapperInference(model)
        state_dict = torch.load(f'../ckpts/{args.model}/{args.ckpt_name}.pth', map_location=device)['model_ema']
        del state_dict['n_averaged']
        new_state_dict = {}
        for key, val in state_dict.items():
            new_state_dict[key[7:]] = val
        model.load_state_dict(new_state_dict)
        logger.info(f'Anchored model loaded from ../ckpts/{args.model}/{args.ckpt_name}.pth')

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.dset == 'imagenet-c' or args.dset == 'imagenet-cbar':
        if args.dset == 'imagenet-c':
            corruption_list = ['brightness', 'defocus_blur', 'fog', 'gaussian_blur', 'glass_blur', 'jpeg_compression', 'motion_blur', 'saturate','snow','speckle_noise', 'contrast', 'elastic_transform', 'frost', 'gaussian_noise', 'impulse_noise', 'pixelate','shot_noise', 'spatter','zoom_blur']
        else:
            corruption_list = ['blue_noise_sample', 'brownish_noise', 'caustic_refraction', 'checkerboard_cutout', 'cocentric_sine_waves',  'inverse_sparkles',  'perlin_noise',  'plasma_noise',  'single_frequency_greyscale',  'sparkles']
            #corruption_list = ['cocentric_sine_waves',  'inverse_sparkles',  'perlin_noise',  'plasma_noise',  'single_frequency_greyscale',  'sparkles']
        severity = [1,2,3,4,5]
        for c in corruption_list:
            for s in severity:
                print(f'Corruption = {c}, Severity = {s}')
                val_dir = os.path.join(args.data_path,c,str(s))
                dataset_test, test_sampler = load_data(val_dir, args)
                num_classes = len(dataset_test.classes)
                print(f'Num class = {num_classes}')
                #print(dataset_test.classes)
                mcce = MulticlassCalibrationError(num_classes=num_classes, n_bins=20, norm='l2')   # Root-mean square calibration error

                data_loader_test = torch.utils.data.DataLoader(
                    dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
                )
                if 'anchored' not in args.model:
                    acc1, acc5, ece, smoothed_ece, _ = evaluate(model, criterion, data_loader_test, mcce, device=device)
                    if torch.distributed.get_rank() == 0:
                        logger.info(f'{args.dset} - Corruption {c}, Severity {s} - Top1 accuracy --- {acc1:.4f}')
                        logger.info(f'{args.dset} - Corruption {c}, Severity {s} - Top5 accuracy --- {acc5:.4f}')
                        logger.info(f'{args.dset} - Corruption {c}, Severity {s} - ECE --- {ece:.4f}')
                        logger.info(f'{args.dset} - Corruption {c}, Severity {s} - Smoothed ECE --- {smoothed_ece:.4f}')
                elif 'anchored' in args.model:
                    dataset_train, train_sampler = load_data('../data/imagenet_train_examples', args)
                    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, sampler=train_sampler, num_workers=args.workers, pin_memory=True)
                    acc1, acc5, ece, smoothed_ece, _, _ = evaluate_anchoring(model, criterion, data_loader_test, data_loader_train, mcce, device=device)
                    #acc1, acc5, ece, smoothed_ece, _, _ = evaluate_anchoring_duq(model, net, criterion, data_loader_test, data_loader_train, mcce, device=device)
                    if torch.distributed.get_rank() == 0:
                        logger.info(f'{args.dset} - Corruption {c}, Severity {s} - Top1 accuracy with {5} anchors --- {acc1:.4f}')
                        logger.info(f'{args.dset} - Corruption {c}, Severity {s} - Top5 accuracy with {5} anchors --- {acc5:.4f}')
                        logger.info(f'{args.dset} - Corruption {c}, Severity {s} - ECE with {5} anchors --- {ece:.4f}')
                        logger.info(f'{args.dset} - Corruption {c}, Severity {s} - Smoothed ECE with {5} anchors --- {smoothed_ece:.4f}')
    

    elif args.dset == 'imagenet-r' or args.dset == 'imagenet-a' or args.dset == 'imagenet-sketch' or args.dset == 'imagenetv2-matched-frequency-format-val' or args.dset == 'imagenetv2-threshold0.7-format-val' or args.dset == 'imagenetv2-top-images-format-val':
        val_dir = os.path.join(args.data_path)
        dataset_test, test_sampler = load_data(val_dir, args)
        num_classes = len(dataset_test.classes)
        print(f'Num class = {num_classes}')
        mcce = MulticlassCalibrationError(num_classes=num_classes, n_bins=20, norm='l2')   # Root-mean square calibration error

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
        )
        if 'anchored' not in args.model:
            acc1, acc5, ece, smoothed_ece, _ = evaluate(model, criterion, data_loader_test, mcce, device=device)
            if torch.distributed.get_rank() == 0:
                logger.info(f'{args.dset} - Top1 accuracy --- {acc1:.4f}')
                logger.info(f'{args.dset} - Top5 accuracy --- {acc5:.4f}')
                logger.info(f'{args.dset} - ECE --- {ece:.4f}')
                logger.info(f'{args.dset} - Smoothed ECE --- {smoothed_ece:.4f}')
        elif 'anchored' in args.model:
            dataset_train, train_sampler = load_data('../data/imagenet_train_examples', args)
            data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, sampler=train_sampler, num_workers=args.workers, pin_memory=True)
            acc1, acc5, ece, smoothed_ece, _, _ = evaluate_anchoring(model, criterion, data_loader_test, data_loader_train, mcce, device=device)
            if torch.distributed.get_rank() == 0:
                logger.info(f'{args.dset} - Top1 accuracy with {5} anchors --- {acc1:.4f}')
                logger.info(f'{args.dset} - Top5 accuracy with {5} anchors --- {acc5:.4f}')
                logger.info(f'{args.dset} - ECE with {5} anchors --- {ece:.4f}')
                logger.info(f'{args.dset} - Smoothed ECE with {5} anchors --- {smoothed_ece:.4f}')
    
    elif args.dset == 'imagenet-clean-val':
        val_dir = os.path.join(args.data_path)
        dataset_test, test_sampler = load_data(val_dir, args)

        num_classes = len(dataset_test.classes)
        print(f'Num class = {num_classes}')
        #print(dataset_test.classes)
        mcce = MulticlassCalibrationError(num_classes=num_classes, n_bins=20, norm='l2')   # Root-mean square calibration error

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
        )
        if 'anchored' not in args.model:
            acc1, acc5, ece, smoothed_ece, _ = evaluate(model, criterion, data_loader_test, mcce, device=device)
            if torch.distributed.get_rank() == 0:
                logger.info(f'{args.dset} - Top1 accuracy --- {acc1:.4f}')
                logger.info(f'{args.dset} - Top5 accuracy --- {acc5:.4f}')
                logger.info(f'{args.dset} - ECE --- {ece:.4f}')
                logger.info(f'{args.dset} - Smoothed ECE --- {smoothed_ece:.4f}')
        elif 'anchored' in args.model:
            dataset_train, train_sampler = load_data('../data/imagenet_train_examples', args)
            data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, sampler=train_sampler, num_workers=args.workers, pin_memory=True)
            acc1, acc5, ece, smoothed_ece, _, _ = evaluate_anchoring(model, criterion, data_loader_test, data_loader_train, mcce, device=device)
            if torch.distributed.get_rank() == 0:
                logger.info(f'{args.dset} - Top1 accuracy with {5} anchors --- {acc1:.4f}')
                logger.info(f'{args.dset} - Top5 accuracy with {5} anchors --- {acc5:.4f}')
                logger.info(f'{args.dset} - ECE with {5} anchors --- {ece:.4f}')
                logger.info(f'{args.dset} - Smoothed ECE with {5} anchors --- {smoothed_ece:.4f}')
            
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Corruptions Evaluation", add_help=add_help)
    parser.add_argument("--dset", default="imagenet-c", type=str, help="dataset name")
    parser.add_argument("--data-path", default="../datasets/imagenet-c", type=str, help="dataset path")
    parser.add_argument("--ckpt_name", default="vanilla_p_0.0_duq_top1_66.736", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
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
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", dest="output_dir", default="./logs", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
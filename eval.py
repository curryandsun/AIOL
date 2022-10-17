import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS, t_v_split
from dataset import ood_data
from utils import AverageMeter, accuracy, get_score, get_roc_sklearn, get_pr_sklearn, get_fpr

best_acc = 0
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser(description='PyTorch AIOL Evaluation')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--root', default='/data0/OOD_Data', type=str,
                        help='root of the dataset path')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=1000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet'],
                        help='dataset name')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--T', default=1., type=float,
                        help='pseudo label temperature')
    parser.add_argument('--out', default='./results/temp',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--seen-ood', type=str, default='imagenet32', 
                        choices=['cifar10', 'cifar100', 'imagenet32', 'svhn', 'split'], 
                        help="choose seen OOD data")             

    args = parser.parse_args()
    global best_acc

    def create_model(args):
        import models.wideresnet as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    print(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)
        logger.addHandler(
            logging.FileHandler(os.path.join(os.path.dirname(args.resume), "eval.log"), "a")
        )

    # find the temperature of last epoch during training via train.log
    try:
        train_log_path = os.path.join(os.path.dirname(args.resume), "train.log")
        with open(train_log_path, 'r') as f:
            lines = f.readlines()
        for line in lines[::-1]:
            line = line.strip()
            if 'temperature' in line:
                import re
                r = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
                args.T = float(r[0])
                break
    except:
        args.T = 1
    logger.info('T:%.4f' % args.T)

    args.model_depth = 28
    args.model_width = 2
    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.seen_ood == 'split':
            args.num_classes = 6
            args.num_labeled = 600

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.seen_ood == 'split':
            args.num_classes = 65
            args.num_labeled = 6500

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    
    test_id_dataset, test_ood_dataset = DATASET_GETTERS[args.dataset].get_cifar(args, mode='val')

    if args.local_rank == 0:
        torch.distributed.barrier()

    # test loader
    test_indices, _ = t_v_split(args, test_id_dataset.targets)

    test_loader = DataLoader(
        test_id_dataset,
        sampler=SubsetRandomSampler(test_indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        )
    
    # seen ood loader
    if args.seen_ood == 'split':
        seen_ood_loader = DataLoader(
            test_ood_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True)
    else:
        seen_ood_loader = ood_data.__dict__[args.seen_ood](args, mode='seen')

    # unseen ood loader
    unseen_ood_loaders_dict = {}
    ds = ['cifar10', 'cifar100', 'svhn', 'blobs', 'texture', 'imagenet32', 'isun', 'lsun', 'places365']
    ds.remove(args.dataset)
    if args.seen_ood != 'split':
        ds.remove(args.seen_ood)

    for d in ds:
        ood_loader = ood_data.__dict__[d](args)
        unseen_ood_loaders_dict[d] = ood_loader

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    ema_model = None
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    if args.resume:
        print("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    if args.use_ema:
        test_model = ema_model.ema
    else:
        test_model = model

    test(args, test_loader, test_model)
    ood_detect_seen(args, test_loader, seen_ood_loader, test_model)
    ood_detect_unseen(args, test_loader, unseen_ood_loaders_dict, test_model)
    logger.info('\n')
    

def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info(args.seen_ood)
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


def ood_detect_seen(args, id_loader, ood_loader, model):
    score_id = get_score(model, id_loader, args)
    score_ood = get_score(model, ood_loader, args)
    auroc = get_roc_sklearn(score_id, score_ood)
    fpr95 = get_fpr(score_id, score_ood)
    aupr = get_pr_sklearn(score_id, score_ood)
    logger.info("auc: %.1f\tfpr: %.1f\taupr: %.1f\tseen: %s" % (auroc * 100, fpr95 * 100, aupr * 100, args.seen_ood))


def ood_detect_unseen(args, test_loader, ood_loaders_dict, model):
    score_id = get_score(model, test_loader, args)
    auc_unseen_sum = 0
    fpr_unseen_sum = 0
    aupr_unseen_sum = 0
    unseen_num = 0
    
    for ood_name, ood_loader in ood_loaders_dict.items():
        score_ood = get_score(model, ood_loader, args)
        auroc = get_roc_sklearn(score_id, score_ood)
        fpr95 = get_fpr(score_id, score_ood)
        aupr = get_pr_sklearn(score_id, score_ood)

        logger.info("auc: %.1f\tfpr: %.1f\taupr: %.1f\t%s" % (auroc * 100, fpr95 * 100, aupr * 100, ood_name))

        auc_unseen_sum += auroc
        fpr_unseen_sum += fpr95
        aupr_unseen_sum += aupr
        unseen_num += 1
    
    logger.info("auc: %.1f\tfpr: %.1f\taupr: %.1f\tunseen" % (auc_unseen_sum / unseen_num * 100, fpr_unseen_sum / unseen_num * 100, aupr_unseen_sum / unseen_num * 100))


if __name__ == '__main__':
    main()

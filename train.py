import argparse
import logging
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS, t_v_split
from models.temperature_scaling import ModelWithTemperature
from utils import AverageMeter, accuracy, get_gmm_threshold, get_roc_sklearn, softXEnt, get_lam, mixup_data

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, args, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))
    if state['epoch'] == args.cr_warmup - 1:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_stage1.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch AIOL Training')
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
    parser.add_argument('--total-steps', default=2**17, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=512, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of CR loss')
    parser.add_argument('--lambda-o', default=1, type=float,
                        help='coefficient of Emin and Emax loss')
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
    parser.add_argument('--cr-warmup', default=205, type=float,
                        help='cr-warmup epochs (stage1)')
    parser.add_argument('--calibrate', action='store_true',
                        help="use temperature scaling to calibrate")
    parser.add_argument('--calibrate-start', type=int, default=40,
                        help="calibrate start epoch")
                        
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

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)
        logger.addHandler(
            logging.FileHandler(os.path.join(args.out, "train.log"), "a")
        )

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

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

    labeled_dataset, unlabeled_dataset, test_id_dataset = DATASET_GETTERS[args.dataset].get_cifar(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True)

    # for validation
    test_indices, val_indices = t_v_split(args, test_id_dataset.targets)

    test_loader = DataLoader(
        test_id_dataset,
        sampler=SubsetRandomSampler(test_indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True)
    
    valid_loader = DataLoader(
        test_id_dataset,
        sampler=SubsetRandomSampler(val_indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    ema_model = None
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  ID = {args.dataset}@{args.num_labeled}")
    logger.info(f"  OOD = {args.seen_ood}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Num Iters = {args.eval_step}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader, valid_loader, model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, valid_loader, model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_o = AverageMeter()
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
    
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()

    for epoch in range(args.start_epoch, args.epochs):
        # two-stage learning
        lam_u, lam_o = get_lam(epoch, args)
        logger.info('lam_u, lam_o: %.2f, %.2f' % (lam_u, lam_o))

        # use ema for stable results
        teacher_model = ema_model.ema
        
        # temperature scaling
        if args.calibrate and epoch >= args.calibrate_start - 1:
            scaled_model = ModelWithTemperature(args, teacher_model)
            scaled_model.set_temperature(valid_loader)
            args.T = scaled_model.temperature.detach()
        logger.info('temperature: %.4f' % args.T)

        # get thresholds by gmm
        id_th, ood_th = get_gmm_threshold(teacher_model, unlabeled_trainloader, args, logger)
        
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]

            if lam_o > 0:
                inputs_u_s = mixup_data(inputs_u_s, args)

            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs) 
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits
            
            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, pseudo_u = torch.max(pseudo_label, dim=-1)

            # selection
            mask_id = max_probs.ge(id_th).float()
            mask_ood = max_probs.le(ood_th).float()
            
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            Lu = softXEnt(logits_u_s, pseudo_label.detach(), reduction='mean')

            entropy = -(F.softmax(logits_u_s, dim=-1) * F.log_softmax(logits_u_s, dim=-1)).sum(dim=-1)
            Lo = (F.cross_entropy(logits_u_s, pseudo_u, reduction='none') * mask_id).mean() - (entropy * mask_ood).mean()

            loss = Lx + lam_u * Lu + lam_o * Lo

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_o.update(Lo.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Loss: {loss:.4f}. Lx: {loss_x:.4f}. Lu: {loss_u:.4f}. Lo: {loss_o:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_o=losses_o.avg,))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()
            
        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)
            auroc = ood_detect(args, unlabeled_trainloader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.train_loss_o', losses_o.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            args.writer.add_scalar('test/3.auroc', auroc, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, args, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
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

    logger.info("epoch: {:d}".format(epoch))
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


def ood_detect(args, unlabeled_trainloader, model, epoch):
    '''
        Return transductive AUROC on unlabeled data
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    if not args.no_progress:
        unlabeled_trainloader = tqdm(unlabeled_trainloader,
                           disable=args.local_rank not in [-1, 0])

    scores_id = []
    scores_ood = []
    with torch.no_grad():
        for batch_idx, ((inputs, _), targets) in enumerate(unlabeled_trainloader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)

            prob = torch.softmax(outputs / args.T, dim=1)
            batch_msp = torch.max(prob, dim=1)[0]

            scores_cur = -batch_msp
            scores_id += list(scores_cur[targets != -1].data.cpu().numpy())
            scores_ood += list(scores_cur[targets == -1].data.cpu().numpy())

            auroc = get_roc_sklearn(scores_id, scores_ood)

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                unlabeled_trainloader.set_description("Detect Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Transductive AUROC: {auroc:.4f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(unlabeled_trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    auroc=auroc * 100,
                ))
        if not args.no_progress:
            unlabeled_trainloader.close()

    logger.info("transductive AUROC: {:.2f}".format(auroc * 100))

    return auroc


if __name__ == '__main__':
    main()

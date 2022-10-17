import logging
import math

import os
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import ConcatDataset

from .ood_data import imagenet32_dataset, cifar10_dataset, cifar100_dataset, svhn_dataset

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

cifar10_id_labels = [2, 3, 4, 5, 6, 7]  # animals
cifar100_id_labels = [0, 1, 2, 3, 4, 6, 7, 11, 14, 15, 18, 19, 21, 24, 26, 27, 29, 30, 31, 32, 34, 35, 36, 38, 42, 43, 44, 45, 46, 47, 50, 51, 52, 53, 54, 55, 56, 57, 59, 62, 63, 64, 65, 66, 67, 70, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 88, 91, 92, 93, 95, 96, 97, 98, 99]  # livings

def target_transform_cifar10(target):
    if target in cifar10_id_labels:
        return cifar10_id_labels.index(target)
    else:
        return -1

def target_transform_cifar100(target):
    if target in cifar100_id_labels:
        return cifar100_id_labels.index(target)
    else:
        return -1


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    id_idx = []

    if args.num_classes == len(cifar10_id_labels):
        id_labels = cifar10_id_labels
    elif args.num_classes == len(cifar100_id_labels):
        id_labels = cifar100_id_labels
    else:
        id_labels = range(args.num_classes)
    for i in id_labels:
        idx = np.where(labels == i)[0]
        id_idx.extend(idx)
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    id_idx = np.array(id_idx)
    assert len(labeled_idx) == args.num_labeled

    ood_idx = np.array(list((set(range(len(labels)))-set(id_idx))))

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    np.random.shuffle(id_idx)
    np.random.shuffle(ood_idx)

    return labeled_idx, id_idx, ood_idx


# split testset into testset and validation set
def t_v_split(args, labels):
    valid_per_class = int(0.1 * len(labels)) // args.num_classes
    labels = np.array(labels)
    valid_idx = []

    if args.num_classes == len(cifar10_id_labels):
        id_labels = cifar10_id_labels
    elif args.num_classes == len(cifar100_id_labels):
        id_labels = cifar100_id_labels
    else:
        id_labels = range(args.num_classes)
    for i in id_labels:
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, valid_per_class, False)
        valid_idx.extend(idx)
    valid_idx = np.array(valid_idx)
    assert len(valid_idx) == int(0.1 * len(labels))

    test_idx = np.array(list((set(range(len(labels)))-set(valid_idx))))
    np.random.shuffle(test_idx)
    np.random.shuffle(valid_idx)

    return test_idx, valid_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        return self.normalize(self.weak(x)), self.normalize(self.strong(x))


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None and len(indexs):
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None and len(indexs):
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class TrainingData:
    def __init__(self, config):
        self.config = config

    def __get_unlabled_ood(self, args, transform_unlabeled, target_transform, train_ood_idxs=None):
        if args.seen_ood == 'cifar10':
            return cifar10_dataset(os.path.join(args.root, "cifar10"), transform=transform_unlabeled)
        elif args.seen_ood == 'cifar100':
            return cifar100_dataset(os.path.join(args.root, "cifar100"), transform=transform_unlabeled)
        elif args.seen_ood == 'imagenet32':
            return imagenet32_dataset(os.path.join(args.root, "ImageNet32"), train=False, transform=transform_unlabeled)
        elif args.seen_ood == 'svhn':
            return svhn_dataset(os.path.join(args.root, "svhn"), transform=transform_unlabeled)
        elif args.seen_ood == 'split':
            if args.dataset == 'cifar10':
                return CIFAR10SSL(os.path.join(args.root, "cifar10"), train_ood_idxs, train=True,transform=transform_unlabeled, target_transform=target_transform)
            else:
                return CIFAR100SSL(os.path.join(args.root, "cifar100"), train_ood_idxs, train=True,transform=transform_unlabeled, target_transform=target_transform)
        else:
            raise Exception('Seen OOD data not found, check args.seen_ood')

    def get_cifar(self, args, mode='train'):
        mean, std = self.config['mean'], self.config['std']
        transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                padding=int(32*0.125),
                                padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ])
        transform_unlabeled = TransformFixMatch(mean=mean, std=std)

        root = os.path.join(args.root, self.config['file'])
        datasets_ori = self.config['datasets']
        datasets_SSL = self.config['datasets_SSL']
        base_dataset = datasets_ori(root, train=True, download=True)
        base_dataset_test = datasets_ori(root, train=False, download=True)

        target_transform = None
        if args.seen_ood == 'split':
            target_transform = self.config['target_transform']
            
        # train_ood_idxs and test_ood_idxs are empty except for split
        train_labeled_idxs, train_id_idxs, train_ood_idxs = x_u_split(
            args, base_dataset.targets)
        _, test_id_idxs, test_ood_idxs = x_u_split(
            args, base_dataset_test.targets)
        
        test_id_dataset = datasets_SSL(
            root, test_id_idxs, train=False, 
            transform=transform_val, target_transform=target_transform)
        if mode == 'val':
            test_ood_dataset = datasets_SSL(
                root, test_ood_idxs, train=False, 
                transform=transform_val)
            return test_id_dataset, test_ood_dataset

        labeled_dataset = datasets_SSL(
            root, train_labeled_idxs, train=True,
            transform=transform_labeled, target_transform=target_transform)
        unlabeled_id_dataset = datasets_SSL(
            root, train_id_idxs, train=True,
            transform=transform_unlabeled, target_transform=target_transform)
        
        unlabeld_ood_dataset = self.__get_unlabled_ood(args, transform_unlabeled, target_transform, train_ood_idxs)
        unlabeled_dataset = ConcatDataset([unlabeled_id_dataset, unlabeld_ood_dataset])

        return labeled_dataset, unlabeled_dataset, test_id_dataset


CIFAR10_CONFIG = {
    'file': 'cifar10',
    'mean': cifar10_mean,
    'std': cifar10_std,
    'datasets': datasets.CIFAR10,
    'datasets_SSL': CIFAR10SSL,
    'target_transform': target_transform_cifar10
}

CIFAR100_CONFIG = {
    'file': 'cifar100',
    'mean': cifar100_mean,
    'std': cifar100_std,
    'datasets': datasets.CIFAR100,
    'datasets_SSL': CIFAR100SSL,
    'target_transform': target_transform_cifar100
}

DATASET_GETTERS = {
    'cifar10': TrainingData(CIFAR10_CONFIG),
    'cifar100': TrainingData(CIFAR100_CONFIG)
}

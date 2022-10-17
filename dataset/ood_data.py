import os
import numpy as np
from skimage.filters import gaussian as gblur
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.utils.data as data

import pickle
from PIL import Image


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def get_norm_layer(dataset):
    if dataset == 'cifar10':
        return transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    else:
        return transforms.Normalize(mean=cifar100_mean, std=cifar100_std)


# for ood test data
def cifar10(args, mode='unseen'):
    norm_layer = get_norm_layer(args.dataset)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        norm_layer
    ])
    
    is_train = False if mode == 'unseen' else True
    testset = datasets.CIFAR10(
                root=os.path.join(args.root, "cifar10"),
                train=is_train,
                download=True,
                transform=transform_val)

    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    return test_loader


def cifar100(args, mode='unseen'):
    norm_layer = get_norm_layer(args.dataset)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        norm_layer
    ])
    
    is_train = False if mode == 'unseen' else True
    testset = datasets.CIFAR100(
                root=os.path.join(args.root, "cifar100"),
                train=is_train,
                download=True,
                transform=transform_val)

    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    return test_loader


def svhn(args, mode='unseen'):
    norm_layer = get_norm_layer(args.dataset)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        norm_layer
    ])
    
    split_mode = 'test' if mode == 'unseen' else 'train'
    testset = datasets.SVHN(
                root=os.path.join(args.root, "svhn"),
                split=split_mode,
                download=True,
                transform=transform_val)

    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    return test_loader


def imagenet32(args, mode='unseen'):
    norm_layer = get_norm_layer(args.dataset)
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), norm_layer])
    if mode == 'unseen':
        dataset = imagenet32_dataset(os.path.join(args.root, "ImageNet32"), transform=transform, train=False)
    elif mode == 'seen':
        dataset = imagenet32_dataset(os.path.join(args.root, "ImageNet32"), transform=transform, train=True, data_nums=50000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return loader


def get_ood_loader(args, filename):
    """
        Minimal version since we use this data only for OOD evaluation.
    """
    norm_layer = get_norm_layer(args.dataset)
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), norm_layer])
    dataset = datasets.ImageFolder(root=os.path.join(args.root, filename), transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return loader

def texture(args):
    return get_ood_loader(args, "dtd/images")

def isun(args):
    return get_ood_loader(args, "iSUN")

def lsun(args):
    return get_ood_loader(args, "LSUN_resize")

def places365(args):
    return get_ood_loader(args, "places365")


# ref: https://github.com/hendrycks/outlier-exposure
def blobs(args):
    """
        Minimal version since we use this data only for OOD evaluation.
    """
    norm_layer = get_norm_layer(args.dataset)
    data = np.float32(np.random.binomial(n=1, p=0.7, size=(10000, 32, 32, 3)))
    for i in range(10000):
        data[i] = gblur(data[i], sigma=1.5, multichannel=False)
        data[i][data[i] < 0.75] = 0.0

    dummy_targets = torch.ones(10000)
    data = torch.cat([norm_layer(x).unsqueeze(0) for x in torch.from_numpy(data.transpose((0, 3, 1, 2)))])
    dataset = torch.utils.data.TensorDataset(data, dummy_targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return loader


# for unlabeled ood data
def transform2negone(y):
    return -1

def cifar10_dataset(root, transform):
    dataset = datasets.CIFAR10(root, train=False, transform=transform, target_transform=transform2negone)
    return dataset

def cifar100_dataset(root, transform):
    dataset = datasets.CIFAR100(root, train=False, transform=transform, target_transform=transform2negone)
    return dataset

def svhn_dataset(root, transform):
    dataset = datasets.SVHN(root, split='test', transform=transform, target_transform=transform2negone)
    return dataset


class imagenet32_dataset(data.Dataset):
    base_folder = 'imagenet32-batches'
    filename = "Imagenet32_train.zip"
    train_list = ['train_data_batch_' + str(i) for i in range(1, 11)]
    test_list = ['val_data']

    def __init__(self, root, transform=None, train=True, data_nums=None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        self.data_nums = data_nums
        # now load the picked numpy arrays
        self.data = []
        self.targets = []
        data_list = self.train_list if train else self.test_list
        for f in data_list:
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            fo.close()

        self.data = np.concatenate(self.data)
        self.data = self.data.reshape((-1, 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.data_nums is not None:
            if self.data_nums > 0:
                self.data = self.data[:self.data_nums]
            else:
                raise Exception('data_nums must be bigger than 0!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            image
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # fake label
        return img, -1

    def __len__(self):
        return len(self.data)

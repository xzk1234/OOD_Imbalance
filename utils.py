import torch
import os
import numpy.random as nr
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from ImageFolder import ImageFolder
import torch.nn as nn
import shutil
import sys
from datetime import datetime
import torch.nn.functional as F


num_test_samples_cifar10 = [1000] * 10
num_test_samples_cifar100 = [100] * 100
DATA_ROOT = os.path.expanduser('/home/intern3/.advertorch/data/CIFAR10/')


def get_imbalanced_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: imbalanced_list
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    selected_list = []
    indices = list(range(0, length))

    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(index)
            num_sample_per_class[label] -= 1

    return selected_list


def get_val_test_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    """
    Return a list of indices for validation and test from a dataset.
    Input: A test dataset (e.g., CIFAR-10)
    Output: validation_list and test_list
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = num_sample_per_class[0] # Suppose that all classes have the same number of test samples

    val_list = []
    test_list = []
    indices = list(range(0, length))
    if shuffle:
        nr.shuffle(indices)
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > (9 * num_samples / 10):
            val_list.append(index)
            num_sample_per_class[label] -= 1
        else:
            test_list.append(index)
            num_sample_per_class[label] -= 1

    return val_list, test_list


def get_imbalanced(dataset, num_sample_per_class, batch_size, TF_train, TF_test):
    print("Building CV {} data loader with {} workers".format(dataset, 8))
    ds = []

    if dataset == 'cifar10':
        dataset_ = datasets.CIFAR10
        num_test_samples = num_test_samples_cifar10
    elif dataset == 'cifar100':
        dataset_ = datasets.CIFAR100
        num_test_samples = num_test_samples_cifar100
    else:
        raise NotImplementedError()

    train_cifar = dataset_(root=DATA_ROOT, train=True, download=False, transform=TF_train)
    train_in_idx = get_imbalanced_data(train_cifar, num_sample_per_class)
    train_in_loader = torch.utils.data.DataLoader(train_cifar, batch_size=batch_size,
                                                  sampler=SubsetRandomSampler(train_in_idx), num_workers=8)
    ds.append(train_in_loader)

    test_cifar = dataset_(root=DATA_ROOT, train=False, download=False, transform=TF_test)
    val_idx, test_idx = get_val_test_data(test_cifar, num_test_samples)
    val_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100, sampler=SubsetRandomSampler(val_idx),
                                             num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100, sampler=SubsetRandomSampler(test_idx),
                                              num_workers=8)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds


def get_ood(data_path, batch_size, transform):
    if data_path == 'SVHN':
        def target_transform(target):
            return int(target)
        data_root = os.path.expanduser(os.path.join('~/.advertorch/data/', 'svhn-data'))
        ood_dataset = datasets.SVHN(
                    root=data_root, split='train', download=True,
                    transform=transform,
                    target_transform=target_transform,
                )
    else:
        ood_dataset = ImageFolder(data_path, transform=transform)
    ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    return ood_loader


def make_longtailed_imb(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num - 1))
    print(mu)
    class_num_list = []
    for i in range(class_num):
        class_num_list.append(int(max_num * np.power(mu, i)))

    return list(class_num_list)


def adjust_learning_rate(optimizer, lr_init, epoch):
    """decrease the learning rate at 160 and 180 epoch ( from LDAM-DRW, NeurIPS19 )"""
    lr = lr_init

    if epoch < 5:
        lr = (epoch + 1) * lr_init / 5
    else:
        if epoch >= 160:
            lr /= 100
        if epoch >= 180:
            lr /= 100

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def sum_t(tensor):
    return tensor.float().sum().item()


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""
    def __init__(self, fn, log_root="./logs/"):
        if not os.path.exists(log_root):
            os.mkdir(log_root)

        logdir = log_root + fn
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        if len(os.listdir(logdir)) != 0:
            ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                            "Will you proceed [y/N]? ")
            if ans in ['y', 'Y']:
                shutil.rmtree(logdir)
            else:
                exit(1)
        self.set_dir(logdir)

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        log_soft = F.log_softmax(pred)
        output = -log_soft * target
        output = torch.sum(output, dim=1)
        return output

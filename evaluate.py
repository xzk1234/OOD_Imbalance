import csv
import os

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import models
import argparse
from torchvision import transforms
from utils import *
import torch.backends.cudnn as cudnn
import time
import pickle
from ImageFolder import TinyImageNet


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--ood', action='store_true')
parser.add_argument('--ood_path', type=str, default='/data/intern3/openimages/')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--ood_rebalance', type=str, default=None)
parser.add_argument('--imbalance_type', type=str, default='longtail')
parser.add_argument('--imbalance_ratio', type=float, default=10.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seed', default=123456)
parser.add_argument('--model', type=str, default='resnet32')
parser.add_argument('--init_lr', type=float, default=0.1)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--augment', type=bool, default=True)
parser.add_argument('--decay', default=2e-4, type=float, help='weight decay')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.seed is not None:
    SEED = args.seed
else:
    SEED = np.random.randint(10000)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
if torch.cuda.is_available():
    N_GPUS = torch.cuda.device_count()
else:
    N_GPUS = 0

if args.ood:
    LOGNAME = 'Imbalance_' + args.dataset + '_' + args.imbalance_type + str(args.imbalance_ratio) + '_ood' + str(args.alpha)
else:
    LOGNAME = 'Imbalance_' + args.dataset + '_' + args.imbalance_type + str(args.imbalance_ratio)
if args.ood_path == 'SVHN':
    LOGNAME += '_SVHN'
elif args.ood_path == 'TinyImageNet':
    LOGNAME += '_TinyImageNet'
if args.ood_rebalance is not None:
    LOGNAME = LOGNAME + '_' + args.ood_rebalance
if args.test:
    LOGNAME = 'test'
logger = Logger(LOGNAME, log_root='./logs_all_data/')
LOGDIR = logger.logdir

if args.dataset == 'cifar100':
    N_CLASSES = 100
    N_SAMPLES = 500
elif args.dataset == 'cifar10':
    N_CLASSES = 10
    N_SAMPLES = 5000
else:
    raise NotImplementedError()

if 'cifar' in args.dataset:
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_ood = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
else:
    raise NotImplementedError()

N_SAMPLES_PER_CLASS_BASE = [int(N_SAMPLES)] * N_CLASSES
if args.imbalance_type == 'longtail':
    N_SAMPLES_PER_CLASS_BASE = make_longtailed_imb(N_SAMPLES, N_CLASSES, args.imbalance_ratio)
elif args.imbalance_type == 'part':
    N_SAMPLES_PER_CLASS_BASE = [int(N_SAMPLES / args.imbalance_ratio)] * N_CLASSES
if args.ood:
    train_loader, val_loader, test_loader = get_imbalanced(args.dataset, N_SAMPLES_PER_CLASS_BASE, args.batch_size // 2,
                                                           transform_train, transform_test)
    if args.ood_path == 'TinyImageNet':
        kwargs = {'num_workers': 8, 'pin_memory': True}
        with open(os.path.join(os.getcwd(), './ti_1M_lowconfidence_unlabeled.pickle'), 'rb') as f:
            aug_file = pickle.load(f)
        trainset_aug_file = aug_file['data']
        print('dataset size : ', trainset_aug_file.shape)
        trainset_aug = TinyImageNet(trainset_aug_file, transform_train)
        ood_loader = torch.utils.data.DataLoader(trainset_aug, batch_size=args.batch_size // 2, shuffle=True, **kwargs)
    else:
        ood_loader = get_ood(args.ood_path, batch_size=args.batch_size // 2, transform=transform_ood)


else:
    train_loader, val_loader, test_loader = get_imbalanced(args.dataset, N_SAMPLES_PER_CLASS_BASE, args.batch_size,
                                                           transform_train, transform_test)

if __name__ == '__main__':
    net = models.__dict__[args.model](N_CLASSES)
    state_dict = torch.load(args.model_path)
    net.load_state_dict(state_dict)

    net = net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
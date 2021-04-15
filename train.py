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
import random


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


def evaluate(net, dataloader, logger=None):
    is_training = net.training
    net.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct, total = 0.0, 0.0
    major_correct, neutral_correct, minor_correct = 0.0, 0.0, 0.0
    major_total, neutral_total, minor_total = 0.0, 0.0, 0.0

    class_correct = torch.zeros(N_CLASSES)
    class_total = torch.zeros(N_CLASSES)

    for inputs, targets in dataloader:
        batch_size = inputs.size(0)
        inputs, targets = inputs.to(device), targets.to(device)

        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * batch_size
        predicted = outputs[:, :N_CLASSES].max(1)[1]
        total += batch_size
        correct_mask = (predicted == targets)
        correct += sum_t(correct_mask)

        # For accuracy of minority / majority classes.
        major_mask = targets < (N_CLASSES // 3)
        major_total += sum_t(major_mask)
        major_correct += sum_t(correct_mask * major_mask)

        minor_mask = targets >= (N_CLASSES - (N_CLASSES // 3))
        minor_total += sum_t(minor_mask)
        minor_correct += sum_t(correct_mask * minor_mask)

        neutral_mask = ~(major_mask + minor_mask)
        neutral_total += sum_t(neutral_mask)
        neutral_correct += sum_t(correct_mask * neutral_mask)

        for i in range(N_CLASSES):
            class_mask = (targets == i)
            class_total[i] += sum_t(class_mask)
            class_correct[i] += sum_t(correct_mask * class_mask)

    results = {
        'loss': total_loss / total,
        'acc': 100. * correct / total,
        'major_acc': 100. * major_correct / major_total,
        'neutral_acc': 100. * neutral_correct / neutral_total,
        'minor_acc': 100. * minor_correct / minor_total,
        'class_acc': 100. * class_correct / class_total,
    }

    msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Major_ACC: %.3f%% | Neutral_ACC: %.3f%% | Minor ACC: %.3f%% ' % \
          (
              results['loss'], results['acc'], correct, total,
              results['major_acc'], results['neutral_acc'], results['minor_acc']
          )
    if logger:
        logger.log(msg)
    else:
        print(msg)

    net.train(is_training)
    return results


def save_checkpoint(acc, model, optim, epoch, index=False):
    # Save checkpoint.
    print('Saving..')

    if isinstance(model, nn.DataParallel):
        model = model.module

    state = {
        'net': model.state_dict(),
        'optimizer': optim.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    if index:
        ckpt_name = 'ckpt_epoch' + str(epoch) + '_' + str(SEED) + '.t7'
    else:
        ckpt_name = 'ckpt_' + str(SEED) + '.t7'

    ckpt_path = os.path.join(LOGDIR, ckpt_name)
    torch.save(state, ckpt_path)


def train_epoch_ood(model, criterion, optimizer, data_loader, ood_data_loader, alpha=1.0, logger=None):
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    for (inputs, targets), ood_inputs in tqdm(zip(data_loader, ood_data_loader)):
        if args.ood_path == 'SVHN':
            ood_inputs, _ = ood_inputs
        inputs, ood_inputs = inputs.to(device), ood_inputs.to(device)
        targets = np.eye(N_CLASSES)[targets]
        if len(inputs) != len(ood_inputs):
            ood_inputs = ood_inputs[:len(inputs)]
        if args.ood_rebalance is None:
            ood_targets_np = np.ones([len(ood_inputs), N_CLASSES]) / N_CLASSES
        elif args.ood_rebalance == 'exp':
            num_per_class = np.array(N_SAMPLES_PER_CLASS_BASE)
            ratio_per_class = num_per_class / np.sum(num_per_class)
            num_class_ratio = np.exp(-ratio_per_class) / np.sum(np.exp(-ratio_per_class))
            num_class_ratio = np.expand_dims(num_class_ratio, axis=0)
            ood_targets_np = np.repeat(num_class_ratio, len(ood_inputs), axis=0)
            '''
            print(N_SAMPLES_PER_CLASS_BASE)
            print(num_per_class)
            print(ratio_per_class)
            print(num_class_ratio)
            print(ood_targets_np)
            time.sleep(10000)
            '''
        elif args.ood_rebalance == 'exp2':
            num_per_class = np.array(N_SAMPLES_PER_CLASS_BASE)
            ratio_per_class = num_per_class / np.sum(num_per_class)
            num_class_ratio = np.exp(-2 * ratio_per_class) / np.sum(np.exp(-2 * ratio_per_class))
            num_class_ratio = np.expand_dims(num_class_ratio, axis=0)
            ood_targets_np = np.repeat(num_class_ratio, len(ood_inputs), axis=0)
        elif args.ood_rebalance == 'ratio':
            num_per_class = np.array(N_SAMPLES_PER_CLASS_BASE)
            ratio_per_class = num_per_class / np.sum(num_per_class)
            num_class_ratio = (1 / ratio_per_class) / np.sum(1 / ratio_per_class)
            num_class_ratio = np.expand_dims(num_class_ratio, axis=0)
            ood_targets_np = np.repeat(num_class_ratio, len(ood_inputs), axis=0)
        elif args.ood_rebalance == 'WeightNorm':
            angleW = model.linear.weight
            mode0 = torch.sum(angleW.mul(angleW), dim=1, keepdim=True)
            num_class_ratio = 1 / mode0.t().data.cpu().numpy()
            num_class_ratio = num_class_ratio / np.sum(num_class_ratio)
            ood_targets_np = np.repeat(num_class_ratio, len(ood_inputs), axis=0)
        elif args.ood_rebalance == 'WeightNorm2':
            angleW = model.linear.weight
            mode0 = torch.norm(angleW, dim=1, keepdim=True)
            num_class_ratio = 1 / mode0.t().data.cpu().numpy()
            num_class_ratio = num_class_ratio / np.sum(num_class_ratio)
            ood_targets_np = np.repeat(num_class_ratio, len(ood_inputs), axis=0)
        inputs = torch.cat([inputs, ood_inputs], dim=0)
        targets = np.concatenate([targets, ood_targets_np], axis=0)
        targets = torch.from_numpy(targets).to(device)

        batch_size = inputs.size(0)
        scalar = np.ones(batch_size)
        scalar[batch_size // 2:] = alpha
        scalar = torch.from_numpy(scalar).to(device)

        outputs, _ = model(inputs)
        loss = criterion(outputs, targets) * scalar
        loss = loss.mean()

        train_loss += loss.item() * batch_size
        predicted = outputs.max(1)[1][:batch_size // 2]
        total += batch_size
        correct += sum_t(predicted.eq(targets.max(1)[1][:batch_size // 2]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    msg = 'Loss: %.3f| Acc: %.3f%% (%d/%d)' % \
          (train_loss / total, 100. * correct / total * 2, correct, total / 2)
    if logger:
        logger.log(msg)
    else:
        print(msg)

    return train_loss / total, 100. * correct / total


def train_epoch(model, criterion, optimizer, data_loader, logger=None):
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in tqdm(data_loader):

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        outputs, _ = model(inputs)
        loss = criterion(outputs, targets).mean()

        train_loss += loss.item() * batch_size
        predicted = outputs.max(1)[1]
        total += batch_size
        correct += sum_t(predicted.eq(targets))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    msg = 'Loss: %.3f| Acc: %.3f%% (%d/%d)' % \
          (train_loss / total, 100. * correct / total, correct, total)
    if logger:
        logger.log(msg)
    else:
        print(msg)

    return train_loss / total, 100. * correct / total


def train_epoch_ood_O2m(model, model_seed, criterion, optimizer, data_loader, ood_data_loader, alpha=1.0, logger=None):
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    for (inputs, targets), ood_inputs in tqdm(zip(data_loader, ood_data_loader)):
        if args.ood_path == 'SVHN':
            ood_inputs, _ = ood_inputs
        inputs, ood_inputs = inputs.to(device), ood_inputs.to(device)
        if len(inputs) != len(ood_inputs):
            ood_inputs = ood_inputs[:len(inputs)]

        def number_of_certain_probability(sequence, probability):
            x = random.uniform(0, 1)
            cumulative_probability = 0.0
            for item, item_probability in zip(sequence, probability):
                cumulative_probability += item_probability
                if x < cumulative_probability:
                    break
            return item

        ood_targets = []
        num_per_class = np.array(N_SAMPLES_PER_CLASS_BASE)
        ratio_per_class = num_per_class / np.sum(num_per_class)
        num_class_ratio = (1 / ratio_per_class) / np.sum(1 / ratio_per_class)
        sequence = list(range(N_CLASSES))
        for _ in range(len(ood_inputs)):
            ood_targets.append(number_of_certain_probability(sequence, num_class_ratio))
        ood_targets = np.array(ood_targets)
        ood_targets = torch.from_numpy(ood_targets).to(device)

        def random_perturb(inputs, attack, eps):
            if attack == 'inf':
                r_inputs = 2 * (torch.rand_like(inputs) - 0.5) * eps
            else:
                r_inputs = (torch.rand_like(inputs) - 0.5).renorm(p=2, dim=1, maxnorm=eps)
            return r_inputs

        random_noise = random_perturb(ood_inputs, 'l2', 0.5)
        ood_inputs = torch.clamp(ood_inputs + random_noise, 0, 1)

        def make_step(grad, attack, step_size):
            if attack == 'l2':
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_grad = grad / (grad_norm + 1e-10)
                step = step_size * scaled_grad
            elif attack == 'inf':
                step = step_size * torch.sign(grad)
            else:
                step = step_size * grad
            return step

        adversarial_criterion = nn.CrossEntropyLoss()
        for _ in range(10):
            ood_inputs = ood_inputs.clone().detach().requires_grad_(True)
            outputs_g, _ = model_seed(ood_inputs)
            loss = adversarial_criterion(outputs_g, ood_targets)
            grad, = torch.autograd.grad(loss, [ood_inputs])

            ood_inputs = ood_inputs - make_step(grad, 'l2', 0.1)
            ood_inputs = torch.clamp(ood_inputs, 0, 1)

        ood_inputs = ood_inputs.detach()

        targets = np.eye(N_CLASSES)[targets]
        ood_targets_np = np.eye(N_CLASSES)[ood_targets.cpu()]

        inputs = torch.cat([inputs, ood_inputs], dim=0)
        targets = np.concatenate([targets, ood_targets_np], axis=0)
        targets = torch.from_numpy(targets).to(device)

        batch_size = inputs.size(0)
        scalar = np.ones(batch_size)
        scalar[batch_size // 2:] = alpha
        scalar = torch.from_numpy(scalar).to(device)

        outputs, _ = model(inputs)
        loss = criterion(outputs, targets) * scalar
        loss = loss.mean()

        train_loss += loss.item() * batch_size
        predicted = outputs.max(1)[1][:batch_size // 2]
        total += batch_size
        correct += sum_t(predicted.eq(targets.max(1)[1][:batch_size // 2]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    msg = 'Loss: %.3f| Acc: %.3f%% (%d/%d)' % \
          (train_loss / total, 100. * correct / total * 2, correct, total / 2)
    if logger:
        logger.log(msg)
    else:
        print(msg)

    return train_loss / total, 100. * correct / total

if __name__ == '__main__':
    TEST_ACC = 0  # best test accuracy
    BEST_VAL = 0  # best validation accuracy
    net = models.__dict__[args.model](N_CLASSES)
    net_seed = models.__dict__[args.model](N_CLASSES)

    net, net_seed = net.to(device), net_seed.to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=args.decay)
    if N_GPUS > 1:
        net = nn.DataParallel(net)

    if args.ood_rebalance == 'O2m':
        # Load checkpoint.
        logger.log('==> Resuming from checkpoint..')
        ckpt_g = 'logs_all_data/' + 'Imbalance_' + args.dataset + '_' + args.imbalance_type + str(args.imbalance_ratio) \
                 + '/ckpt_123456.t7'
        ckpt_g = torch.load(ckpt_g)
        net_seed.load_state_dict(ckpt_g['net'])

    for epoch in range(args.epoch):
        logger.log('Epoch: {}'.format(epoch))
        adjust_learning_rate(optimizer, args.init_lr, epoch)
        per_cls_weights = torch.ones(N_CLASSES).to(device)
        if args.ood:
            criterion = CrossEntropy().to(device)
            if args.ood_rebalance == 'O2m':
                train_loss, train_acc = train_epoch_ood_O2m(net, net_seed, criterion, optimizer, train_loader,
                                                            ood_loader, alpha=args.alpha, logger=logger)
            else:
                train_loss, train_acc = train_epoch_ood(net, criterion, optimizer, train_loader, ood_loader,
                                                        alpha=args.alpha, logger=logger)
        else:
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights, reduction='none').to(device)
            train_loss, train_acc = train_epoch(net, criterion, optimizer, train_loader, logger=logger)

        val_eval = evaluate(net, val_loader, logger=logger)
        val_acc = val_eval['acc']
        if val_acc >= BEST_VAL:
            BEST_VAL = val_acc

            test_stats = evaluate(net, test_loader, logger=logger)
            TEST_ACC = test_stats['acc']
            TEST_ACC_CLASS = test_stats['class_acc']

            save_checkpoint(TEST_ACC, net, optimizer, epoch)
            logger.log("========== Class-wise test performance ( avg : {} ) ==========".format(TEST_ACC_CLASS.mean()))

    logger.log("========== Final test performance ( avg : {} ) ==========".format(TEST_ACC_CLASS.mean()))
    logger.log("========== Final test performance ( major : {} ) ==========".format(test_stats['major_acc']))
    logger.log("========== Final test performance ( neutral : {} ) ==========".format(test_stats['neutral_acc']))
    logger.log("========== Final test performance ( minor : {} ) ==========".format(test_stats['minor_acc']))

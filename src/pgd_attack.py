from __future__ import print_function
import os
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from model import HandcraftTuNASLikeNetworkCIFAR
import genotypes
import logging
import sys

CUR = os.getcwd()
PARENT = Path(CUR).parent
ROOT = os.path.join(PARENT, 'best_params')
files = os.listdir(ROOT)
dir_names = [f for f in files if os.path.isdir(os.path.join(ROOT, f))]
arch_dirs = [s for s in dir_names if 'arch' in s]

CIFAR_CLASSES = 10
CELL_KINDS = 4

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=25, metavar='N', help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031, help='perturbation')
parser.add_argument('--num-steps', type=int, default=20, help='perturb number of steps')
parser.add_argument('--step-size', default=0.01, help='perturb step size')
parser.add_argument('--random', default=True, help='random initialization for PGD')
parser.add_argument('--white-box-attack', default=False, help='whether perform white-box attack')
parser.add_argument('--arch', type=str, default='000', help='which architecture to use')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--target_arch', type=str, default='ADVRUSH', help='which architecture to use')
parser.add_argument('--source_arch', type=str, default='ADVRUSH', help='which architecture to use')
parser.add_argument('--target_checkpoint', type=str, default='./', help='which architecture to use')
parser.add_argument('--source_checkpoint', type=str, default='./', help='which architecture to use')
parser.add_argument('--log_path', type=str, default='./', help='path to store log file')
parser.add_argument('--checkpoint', type=str, default='./', help='which architecture to use')
parser.add_argument('--data_type', type=str, default='cifar10', help='which dataset to use')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('../', 'best_params', 'test_acc.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# set up data loader
if args.data_type == 'cifar10':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
elif args.data_type == 'cifar100':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
elif args.data_type == 'svhn':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)

#mini_size = int(len(testset) * 0.1)
#val_size = len(testset) - mini_size
#testset, _ = torch.utils.data.random_split(testset, [mini_size, val_size])

print(f"The number of test data: {len(testset)}")
# logging.info("The number of valid data: %d", len(valid_data))

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    pred = torch.max(out[0],dim=1)[1]
    
    # Original
    # err = (out.data.max(1)[1] != y.data).float().sum()
    cln_acc = (pred == y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            # Original
            # loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            loss = nn.CrossEntropyLoss()(model(X_pgd)[0], y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    # Original
    # err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    pgd_acc = (torch.max(model(X_pgd)[0], dim=1)[1] == y.data).float().sum()
    # Original
    # print('err pgd (white-box): ', err_pgd)
    return cln_acc, pgd_acc

def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    cln_acc_total = 0
    pgd_acc_total = 0

    for step, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        cln_acc, pgd_acc = _pgd_whitebox(model, X, y)
        cln_acc_total += cln_acc
        pgd_acc_total += pgd_acc
        if step % args.report_freq == 0:
            print(f'valid step:{step} acc pgd (white-box): {pgd_acc_total}')

    cln_acc_100 = 100 * cln_acc_total / len(testset)
    pgd_acc_100 = 100 * pgd_acc_total / len(testset)
    
    return cln_acc_100, pgd_acc_100


def main():
    for ad in arch_dirs:
        arch = ad[5:8]
        dir_path = os.path.join(ROOT, ad)
        if os.path.exists(dir_path):
            cells = []
            for c in arch:
                assert c >= '0' and c <= str(CELL_KINDS-1)
                cell = eval("genotypes.%s" % f"CELL{c}")
                cells.append(cell)
            
            model = HandcraftTuNASLikeNetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, cells)
            data_path = os.path.join(ROOT, ad, 'model_best_ae.pth.tar')
            checkpoint = torch.load(data_path, map_location='cuda:0')        
            model.load_state_dict(checkpoint['state_dict'])
            model.drop_path_prob = args.drop_path_prob
            model.cuda()
            cln_acc, adv_acc = eval_adv_test_whitebox(model, device, test_loader)
            print(f"{ad}: test-acc:{cln_acc} test-ae-acc:{adv_acc}")
            logging.info("%s: test-acc:%f test-ae-acc:%f", ad, cln_acc, adv_acc)


if __name__ == '__main__':
    main()
import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import random
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import HandcraftTuNASLikeNetworkCIFAR
from trades import trades_loss, madry_loss

CELL_KINDS = 4
CIFAR_CLASSES = 10

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size') #128
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation')
parser.add_argument('--num_steps', type=int, default=7, help='perturb number of steps')
parser.add_argument('--step_size', type=float, default=0.01, help='perturb step size')
parser.add_argument('--beta', type=float, default=6.0, help='regularization in TRADES')
parser.add_argument('--adv_loss', type=str, default='pgd', help='experiment name')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='000', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--root', type=str, default='../result', help='random seed')

args = parser.parse_args()

args.save = f'arch-{args.arch}-{time.strftime("%Y%m%d-%H%M%S")}'
args.save = os.path.join(args.root, args.save)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('../src/*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)


  cells = []
  for c in args.arch:
    assert c >= '0' and c <= str(CELL_KINDS-1)
    cell = eval("genotypes.%s" % f"CELL{c}")
    cells.append(cell)

  model = HandcraftTuNASLikeNetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, cells)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, _ = utils._data_transforms_cifar10(args)
  trainval_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)  

  train_size = int(len(trainval_data) * 0.8)
  val_size = len(trainval_data) - train_size
  train_data, valid_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

  logging.info("The number of train data: %d", len(train_data))
  logging.info("The number of valid data: %d", len(valid_data))
  logging.info("The number of total data: %d", len(train_data)+len(valid_data))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  best_acc = 0.0
  best_ae_acc = 0.0
  for epoch in range(args.epochs):
    adjust_learning_rate(optimizer, epoch)
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_ae_acc, train_loss = train(train_queue, model, criterion, optimizer)
    logging.info('epoch %d train_acc %f train_ae_acc %f train_loss %f', epoch, train_acc, train_ae_acc,train_loss)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    valid_ae_acc, _ = adv_infer(valid_queue, model, criterion)

    if valid_ae_acc > best_ae_acc:
      best_ae_acc = valid_ae_acc
      utils.save_checkpoint({
        'best_cln': 'best_cln',
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, is_best=True, save=args.save, epoch=epoch, is_cln=False)

    if valid_acc > best_acc:
      best_acc = valid_acc
      utils.save_checkpoint({
        'best_ae': 'best_ae',
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, is_best=True, save=args.save, epoch=epoch, is_cln=True)
    logging.info('epoch %d valid_acc %f, valid_ae_acc %f, best_acc %f, best_ae_acc %f', epoch, valid_acc, valid_ae_acc, best_acc, best_ae_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    utils.save_checkpoint({
        'epoch': epoch + 1, 
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_ae_acc': train_ae_acc,
        'valid_acc': valid_acc,
        'valid_ae_acc': valid_ae_acc,
        }, is_best=False, save=args.save, epoch=epoch)

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  ae_top1 = utils.AvgrageMeter()
  ae_top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda(non_blocking=True)
    target = Variable(target).cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    if args.adv_loss == 'pgd':
      loss, ae_logits = madry_loss(
            model,
            input, 
            target, 
            optimizer,
            step_size = args.step_size,
            epsilon = args.epsilon, 
            perturb_steps = args.num_steps)
    elif args.adv_loss == 'trades':
      loss, loss_natural, loss_robust = trades_loss(model,
                input,
                target,
                optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps,
                beta=args.beta,
                distance='l_inf')
    #loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    ae_prec1, ae_prec5 = utils.accuracy(ae_logits, target, topk=(1, 5))

    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)
    ae_top1.update(ae_prec1.data.item(), n)
    ae_top5.update(ae_prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train step:%03d loss:%e top1-acc:%f top5-acc:%f top1-ae-acc:%f top5-ae-acc:%f', step, objs.avg, top1.avg, top5.avg, ae_top1.avg, ae_top5.avg)

  return top1.avg, ae_top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, requires_grad=False).cuda(non_blocking=True)
      target = Variable(target, requires_grad=False).cuda(non_blocking=True)

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid step:%03d loss:%e top1 acc:%f top5 acc%f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def adv_infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, requires_grad=False).cuda(non_blocking=True)
      target = Variable(target, requires_grad=False).cuda(non_blocking=True)

      # generate adversarial example
      x_adv = input.detach() + 0.001 * torch.randn(input.shape).cuda().detach()
      step_size=0.003
      epsilon=0.031
      perturb_steps=10
      criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')

      for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
          logits, _ = model(x_adv)
          loss_ce = criterion_ce(logits, target).mean()
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, input - epsilon), input + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
     
      x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
      logits, _ = model(x_adv)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid step:%03d loss:%e top1 ae acc:%f top5 ae acc:%f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def adjust_learning_rate(optimizer, epoch):
  """decrease the learning rate"""
  lr = args.learning_rate
  if epoch >= 99:
    lr = args.learning_rate * 0.1
  if epoch >= 149:
    lr = args.learning_rate * 0.01
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

if __name__ == '__main__':
  main() 
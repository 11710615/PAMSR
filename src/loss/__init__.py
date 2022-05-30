import os
from importlib import import_module
from turtle import forward

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# revisit L1 loss
class RL1(nn.Module):
    def __init__(self):
        super(RL1, self).__init__()
        self.l1 = nn.L1Loss()
    def forward(self,m,hr): # sr: tuple
        sr = m[0] + torch.abs(hr - m[0]) * torch.randn_like(hr)
        out = self.l1(sr,hr)
        return out

class AuxLoss(nn.Module):
    def __init__(self):
        super(AuxLoss, self).__init__()
        self.l1 = nn.L1Loss()
    def forward(self,m,hr): # sr: tuple
        sr_ = m[0].detach()
        sigma = m[1]
        out = self.l1(torch.abs(hr-sr_),sigma)
        return out


# torch.autograd.set_detect_anomaly(True)
class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        # self.n_GPUs = args.n_GPUs
        self.gpu_ids = args.gpu_ids
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'RL1':
                loss_function = RL1()
            elif loss_type == 'AuxLoss':
                loss_function = AuxLoss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range,
                    n_colors=args.n_colors,
                    RL1=(args.model == 'swinir_sigblock')
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type,
                    RL1=(args.model == 'swinir_sigblock')
                )
            elif loss_type.find('FMAE') >=0:
                module = import_module('loss.fmae')
                loss_function = getattr(module,'FMAE')()  # 无需参数的实例化

            elif loss_type.find('Style') >= 0:
                module = import_module('loss.style')
                loss_function = getattr(module, 'Style')(
                    rgb_range=args.rgb_range,
                    n_colors=args.n_colors,
                    RL1=(args.model == 'swinir_sigblock'))

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        # if not args.cpu and args.n_GPUs > 1:
        #     self.loss_module = nn.DataParallel(
        #         self.loss_module, range(args.n_GPUs)
        #     )
        if not args.cpu and len(args.gpu_ids) > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, args.gpu_ids
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)  # mask_weighted L1 loss
                # print('loss', loss)
                # print('sr.shape:',sr.shape)
                # print('hr.shape:',hr[0,0,:,:])
                # y1
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        # if self.n_GPUs == 1:
        #     return self.loss_module
        if len(self.gpu_ids) == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()


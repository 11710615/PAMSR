from ast import operator
from cmath import nan
from operator import mod
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
from model import common
from kornia.losses import ssim_loss as ssim_m
import wandb

class Gradient_L1(nn.Module):
    def __init__(self, ksize=3):
        super(Gradient_L1, self).__init__()
        self.ksize = ksize
        self.l1 = nn.L1Loss()

    def get_local_gradient(self, x):
        kernel_v = [[-1, -2, -1], 
                    [0, 0, 0], 
                    [1, 2, 1]]
        kernel_h = [[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()
        
        x_list = []
        # x = x.to(torch.float32)
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)
        x = torch.cat(x_list, dim = 1)
        
        # # normalize
        # x = (x-torch.min(x))/torch.max(x)
        return x   

    def forward(self,output,hr):
        if isinstance(output,list):
            sr = output[0]
            sr_ema = output[1]
        else:
            sr = output[0]
            sr_ema = output[2]
        residual_ema = torch.sum(torch.abs(hr - sr_ema), 1, keepdim=True) / 255
        residual_SR = torch.sum(torch.abs(hr - sr), 1, keepdim=True) / 255

        pixel_level_gradient = self.get_local_gradient(hr.clone().detach())
        overall_weight = pixel_level_gradient
        # normalize
        eps = 1e-6
        overall_weight = (overall_weight-torch.min(overall_weight))/(torch.max(overall_weight)+eps)
        overall_weight[residual_SR < residual_ema] = 0
        out = self.l1(overall_weight*sr, overall_weight*hr)
        return out


class Gradient_WL1(nn.Module):
    def __init__(self, ksize=3):
        super(Gradient_WL1, self).__init__()
        self.ksize = ksize
        self.l1 = nn.L1Loss()
    def get_local_weights(self,residual, ksize=7):
        ksize = self.ksize
        pad = (ksize - 1) // 2
        residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')
        unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
        pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)
        return pixel_level_weight

    def get_local_gradient(self, x):
        kernel_v = [[-1, -2, -1], 
                    [0, 0, 0], 
                    [1, 2, 1]]
        kernel_h = [[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()
        
        x_list = []
        # x = x.to(torch.float32)
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)
        x = torch.cat(x_list, dim = 1)
        
        # # normalize
        # x = (x-torch.min(x))/torch.max(x)
        
        return x

    def forward(self,output,hr):
        if isinstance(output,list):
            sr = output[0]
            sr_ema = output[1]
        else:
            sr = output[0]
            sr_ema = output[2]
        residual_ema = torch.sum(torch.abs(hr - sr_ema), 1, keepdim=True) / 255
        residual_SR = torch.sum(torch.abs(hr - sr), 1, keepdim=True) / 255

        # need to detach, else nan will appear
        patch_level_weight = torch.var(residual_SR.clone().detach(), dim=(-1, -2, -3), keepdim=True) ** (1/5)
        pixel_level_weight = self.get_local_weights(residual_SR.clone().detach(), self.ksize)
        pixel_level_gradient = self.get_local_gradient(hr.clone().detach())
        overall_weight = patch_level_weight * pixel_level_weight * pixel_level_gradient

        # normalize
        eps = 1e-6
        overall_weight = (overall_weight-torch.min(overall_weight))/(torch.max(overall_weight)+eps)

        overall_weight[residual_SR < residual_ema] = 0
        out = self.l1(overall_weight*sr, overall_weight*hr)
        return out

class WL1(nn.Module):
    def __init__(self, ksize=3):
        super(WL1, self).__init__()
        self.ksize = ksize
        self.l1 = nn.L1Loss()
    def get_local_weights(self,residual, ksize=7):
        ksize = self.ksize
        pad = (ksize - 1) // 2
        residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')
        unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
        pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)
        return pixel_level_weight

    def forward(self,output,hr):
        if isinstance(output,list):
            sr = output[0]
            sr_ema = output[1]
        else:
            sr = output[0]
            sr_ema = output[2]
        residual_ema = torch.sum(torch.abs(hr - sr_ema), 1, keepdim=True) / 255
        residual_SR = torch.sum(torch.abs(hr - sr), 1, keepdim=True) / 255

        # need to detach, else nan will appear
        patch_level_weight = torch.var(residual_SR.clone().detach(), dim=(-1, -2, -3), keepdim=True) ** (1/5)
        pixel_level_weight = self.get_local_weights(residual_SR.clone().detach(), self.ksize)
        overall_weight = patch_level_weight * pixel_level_weight

        # # normalize
        # eps = 1e-6
        # overall_weight = (overall_weight-torch.min(overall_weight))/(torch.max(overall_weight)+eps)
        # # inverse
        
        overall_weight[residual_SR < residual_ema] = 0
        out = self.l1(overall_weight*sr, overall_weight*hr)

        # print('output',torch.max(overall_weight),out,torch.max(patch_level_weight),torch.max(pixel_level_weight))
        # r
        return out


class InvWL1(nn.Module):
    def __init__(self, ksize=3):
        super(InvWL1, self).__init__()
        self.ksize = ksize
        self.l1 = nn.L1Loss()
    def get_local_weights(self,residual, ksize=7):
        ksize = self.ksize
        pad = (ksize - 1) // 2
        residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')
        unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
        pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)
        return pixel_level_weight

    def forward(self,output,hr):
        if isinstance(output,list):
            sr = output[0]
            sr_ema = output[1]
        else:
            sr = output[0]
            sr_ema = output[2]
        residual_ema = torch.sum(torch.abs(hr - sr_ema), 1, keepdim=True) / 255
        residual_SR = torch.sum(torch.abs(hr - sr), 1, keepdim=True) / 255

        # need to detach, else nan will appear
        patch_level_weight = torch.var(residual_SR.clone().detach(), dim=(-1, -2, -3), keepdim=True) ** (1/5)
        pixel_level_weight = self.get_local_weights(residual_SR.clone().detach(), self.ksize)
        overall_weight = patch_level_weight * pixel_level_weight

        # normalize
        eps = 1e-6
        overall_weight = (overall_weight-torch.min(overall_weight))/(torch.max(overall_weight)+eps)
        # inverse
        overall_weight = 1 - (overall_weight > 0.01) * overall_weight + (overall_weight <= 0.01) * (overall_weight - 1)

        overall_weight[residual_SR < residual_ema] = 0
        out = self.l1(overall_weight*sr, overall_weight*hr)

        # print('output',torch.max(overall_weight),out,torch.max(patch_level_weight),torch.max(pixel_level_weight))
        # r
        return out
    
# rewrite L1 loss
class RL1(nn.Module):
    def __init__(self):
        super(RL1, self).__init__()
        self.l1 = nn.L1Loss()
    def forward(self,sr,hr): # sr: tuple
        if isinstance(sr, list):
            sr = sr[0]
        elif isinstance(sr, tuple):
            sr = sr[0]
        else:
            sr = sr
        out = self.l1(sr,hr)
        return out
    
# ssim loss before rec
class ssimloss(nn.Module):
    def __init__(self):
        super(ssimloss, self).__init__()
    def forward(self,sr,hr): # sr: tuple
        if isinstance(sr, list):
            sr = sr[0]
        elif isinstance(sr, tuple):
            sr = sr[0]
        else:
            sr = sr
        out = ssim_m(sr, hr, 5)
        return out

class L1GM(nn.Module):
    def __init__(self):
        super(L1GM,self).__init__()
        self.gm = common.Get_gradient()
        self.l1 = nn.L1Loss()
    def forward(self,out,hr):
        sr_gm = self.gm(out[0])
        hr_gm = self.gm(hr)
        return self.l1(sr_gm,hr_gm)

class L1RG(nn.Module):
    def __init__(self):
        super(L1RG, self).__init__()
        self.gm = common.Get_gradient()
        self.l1 = nn.L1Loss()
    def forward(self,out,hr):
        rec_gm=out[1]
        hr_gm = self.gm(hr)
        return self.l1(rec_gm,hr_gm)


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
            elif loss_type == 'WL1':
                loss_function = WL1()
            elif loss_type == 'InvWL1':
                loss_function = InvWL1()
            elif loss_type == 'Gradient_WL1':
                loss_function = Gradient_WL1()
            elif loss_type == 'Gradient_L1':
                loss_function = Gradient_L1()
            elif loss_type == 'L1GM':
                loss_function = L1GM()
            elif loss_type == 'L1RG':
                loss_function = L1RG()
            elif loss_type == 'ssim_rec':
                module = import_module('loss.ssim')
                loss_function = getattr(module, 'SSIM')()
            elif loss_type == 'ssim':
                loss_function = ssimloss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range,
                    n_colors=args.n_colors,
                    RL1=(args.model in ['swinir_sigblock','swinir_sp'])
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type,
                    RL1=(args.model in ['swinir_sigblock','swinir_sp'])
                )
            elif loss_type.find('ganGM') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type,
                    RL1=(args.model in ['swinir_sigblock','swinir_sp']),
                    GM_input=True
                )
            elif loss_type.find('FMAE') >=0:
                module = import_module('loss.fmae')
                loss_function = getattr(module,'FMAE')()  # 无需参数的实例
            elif loss_type.find('Style') >= 0:
                module = import_module('loss.style')
                loss_function = getattr(module, 'Style')(
                    rgb_range=args.rgb_range,
                    n_colors=args.n_colors,
                    RL1=(args.model in ['swinir_sigblock','swinir_sp']))       
            elif loss_type.find('topology') >= 0:
                module = import_module('loss.topology')
                loss_function = getattr(module, 'topology')()
            elif loss_type == 'rec':  # rec_gradloss
                module = import_module('loss.rec')
                loss_function = getattr(module, 'rec')(whether_rec=True)
            elif loss_type == 'rec_l1':  # rec_gradloss
                module = import_module('loss.rec_l1')
                loss_function = getattr(module, 'rec')(whether_rec=True)
            elif loss_type == 'gradloss':
                module = import_module('loss.gradloss')
                loss_function = getattr(module, 'gradloss')(whether_rec=False)
            elif loss_type == 'GradientLoss':
                module = import_module('loss.GradientLoss')
                loss_function = getattr(module, 'GradientLoss')(rec=args.rec, operator=args.operator)
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
        # if not args.cpu and len(args.gpu_ids) > 0:
        #     self.loss_module = nn.DataParallel(
        #         self.loss_module)
        if not args.cpu and len(args.gpu_ids) > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, args.gpu_ids
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):

        losses = []
        loss_wandb = {}
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()

                loss_wandb[l['type']] = effective_loss

            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss
                loss_wandb['DIS'] = self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()
    
        return loss_sum, loss_wandb

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


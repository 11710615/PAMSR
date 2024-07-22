import math
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import copy

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

# use gradient intensity. input:[b,h,w,c]->output:[b,c,h,c]
class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[-1, -2, -1], 
                    [0, 0, 0], 
                    [1, 2, 1]]
        kernel_h = [[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim = 1)
        return x

class grad_info(nn.Module):
    def __init__(self):
        super(grad_info, self).__init__()
        kernel_v = [[-1, -2, -1], 
                    [0, 0, 0], 
                    [1, 2, 1]]
        kernel_h = [[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)
    def forward(self, x):
        x_list = []
        x_h_list = []
        x_v_list = []
        # x = x.to(torch.float32)
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)
            x_h_list.append(x_i_h)
            x_v_list.append(x_i_v)
        x = torch.cat(x_list, dim = 1)
        x_h = torch.cat(x_h_list, dim=1)
        x_v = torch.cat(x_v_list, dim=1)
        out = torch.cat([x, x_h, x_v], dim=1)
        
        # # normalize
        # x = (x-torch.min(x))/torch.max(x)
        return out 


def ema(model, model_ema, decay=0.999):
    model_params = dict(model.named_parameters())
    model_ema_params = dict(model_ema.named_parameters())
    for k in model_params:
        model_ema_params[k].data.mul_(decay).add_(model_params[k].data, alpha=1 - decay)
    return model_ema
# class EMA():
#     """
#     from https://zhuanlan.zhihu.com/p/68748778
#     """
#     def __init__(self, model, decay=0.999):
#         self.model = model
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}
# 
    # def register(self):
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             self.shadow[name] = param.data.clone()

    # def update(self):
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             assert name in self.shadow
    #             new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
    #             self.shadow[name] = new_average.clone()

    # def apply_shadow(self):
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             assert name in self.shadow
    #             self.backup[name] = param.data
    #             param.data = self.shadow[name]

    # def restore(self):
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             assert name in self.backup
    #             param.data = self.backup[name]
    #     self.backup = {}
from turtle import forward
from typing import Mapping
import numpy as np
import torch.nn as nn
import torch
import cv2
from tqdm import tqdm
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet

class rdice(nn.Module):
    def __init__(self, whether_rec=True):
        super(rdice, self).__init__()
        self.L1 = nn.L1Loss()
        rec_map = np.load('./loss/rec_map.npy', allow_pickle=True).item()
        self.h_idx = torch.from_numpy(rec_map['h_idx']).type(torch.long)
        self.w_idx = torch.from_numpy(rec_map['w_idx']).type(torch.long)
        self.rec = rec
        arch_kwargs = {
            "n_stages": 9,
            "features_per_stage": [
                32,
                64,
                128,
                256,
                512,
                512,
                512,
                512,
                512
            ],
            "conv_op": torch.nn.modules.conv.Conv2d,
            "kernel_sizes": [
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ]
            ],
            "strides": [
                [
                    1,
                    1
                ],
                [
                    2,
                    2
                ],
                [
                    2,
                    2
                ],
                [
                    2,
                    2
                ],
                [
                    2,
                    2
                ],
                [
                    2,
                    2
                ],
                [
                    2,
                    2
                ],
                [
                    2,
                    2
                ],
                [
                    2,
                    2
                ]
            ],
            "n_conv_per_stage": [
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2
            ],
            "n_conv_per_stage_decoder": [
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2
            ],
            "conv_bias": True,
            "norm_op": torch.nn.modules.instancenorm.InstanceNorm2d,
            "norm_op_kwargs": {
                "eps": 1e-05,
                "affine": True
            },
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": torch.nn.LeakyReLU,
            "nonlin_kwargs": {
                "inplace": True
            }
        }
        self.preTrainUnet = PlainConvUNet(
                 input_channels=1,
                 n_stages=arch_kwargs['n_stages'],
                 features_per_stage=arch_kwargs['features_per_stage'],
                 conv_op=arch_kwargs['conv_op'],
                 kernel_sizes=arch_kwargs['kernel_sizes'],
                 strides=arch_kwargs['strides'],
                 n_conv_per_stage=arch_kwargs['n_conv_per_stage'],
                 num_classes=2,
                 n_conv_per_stage_decoder=arch_kwargs['n_conv_per_stage_decoder'],
                 conv_bias=arch_kwargs['conv_bias'],
                 norm_op=arch_kwargs['norm_op'],
                 norm_op_kwargs=arch_kwargs['norm_op_kwargs'],
                 dropout_op=arch_kwargs['dropout_op'],
                 dropout_op_kwargs=arch_kwargs['dropout_op_kwargs'],
                 nonlin=arch_kwargs['nonlin'],
                 nonlin_kwargs=arch_kwargs['nonlin_kwargs'],
                 deep_supervision=False,
                 nonlin_first=False
)
        params = torch.load(r'/mnt/pank/nnUNet/DATASET/nnUNet_trained_models/Dataset999_PAMSeg/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth')
        self.preTrainUnet.load_state_dict(params['network_weights'])
        for p in self.parameters():
            p.requires_grad = False
    
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
    
    def rec_img(self, img, patch_cord):

        # patch_cord = [h,w,h_size, w_size, start_row][b]
        b,c,*_ = img.shape
        imgReconstruction = torch.zeros([b,c,1024,1024]).type_as(img)  # [b,1,h,w]
      
        for i in range(b):
            h_ii = self.h_idx[patch_cord[0][i]:patch_cord[2][i], patch_cord[1][i]:patch_cord[3][i]].flatten()
            w_ii = self.w_idx[patch_cord[0][i]:patch_cord[2][i], patch_cord[1][i]:patch_cord[3][i]].flatten()
            # print('***',h_ii.shape, len(w_ii), img.shape,patch_cord[0][i],patch_cord[2][i], patch_cord[1][i],patch_cord[3][i])
            imgReconstruction[i:i+1,:,h_ii, w_ii] = img[i:i+1,:,:,:].flatten()
            
        return imgReconstruction
 
    def forward(self, sr, hr):
        patch_cord = sr[-1]
        if isinstance(sr, list):
            sr = sr[0]
        # print('**sr', sr.shape)
        *_, h, w = sr.shape
        if self.rec:
            sr_rec = self.rec_img(sr, patch_cord)
            sr_rec_grad = self.get_local_gradient(sr_rec)
            hr_rec = self.rec_img(hr, patch_cord)
            hr_rec_grad = self.get_local_gradient(hr_rec)
            out = self.L1(sr_rec_grad, hr_rec_grad)
            out = out * 1024 * 1024 / h / w
        else:
            sr_grad = self.get_local_gradient(sr)
            hr_grad = self.get_local_gradient(hr)
            out = self.L1(sr_grad, hr_grad)
        return out



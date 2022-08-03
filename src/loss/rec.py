from turtle import forward
from typing import Mapping
import numpy as np
import torch.nn as nn
import torch
import cv2
from tqdm import tqdm

class rec(nn.Module):
    def __init__(self, downsample_gt=False):
        super(rec, self).__init__()
        self.downsample_gt = downsample_gt
        self.L1 = nn.L1Loss()
        rec_map = np.load('./loss/rec_map.npy', allow_pickle=True).item()
        self.h_idx = torch.from_numpy(rec_map['h_idx']).type(torch.long)
        self.w_idx = torch.from_numpy(rec_map['w_idx']).type(torch.long)

    # def pad_img(self, patch, patch_cord):
    #     batch, channel, h, w = patch.shape
    #     img_pad = torch.zeros((batch, channel, 1024, 1024))
    #     for b in range(batch):
    #         if not self.downsample_gt:
    #             img_pad[b:b+1, :, patch_cord[0][b]:patch_cord[2][b], patch_cord[1][b]:patch_cord[3][b]] = patch[b:b+1,:,:,:]
    #         else:
    #             start_row = patch_cord[4]
    #             for i in range(patch_cord[0][b], h + patch_cord[0][b]):
    #                 img_pad[b:b+1, :, 2*i+start_row[b]:2*i+start_row[b]+1, patch_cord[1][b]:patch_cord[3][b]] = patch[b:b+1,: ,i-patch_cord[0][b]:i+1-patch_cord[0][b],:]
    #     return img_pad
    
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
        sr_rec = self.rec_img(sr, patch_cord)
        
        hr_rec = self.rec_img(hr, patch_cord)
        # print('sr***', sr.shape, sr_rec.shape, hr_rec.shape)
        out = self.L1(sr_rec, hr_rec)
        # print('loss**', out)
        return out



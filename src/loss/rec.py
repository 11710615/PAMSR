from turtle import forward
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
    def pad_img(self, patch, patch_cord):
        batch, channel, h, w = patch.shape
        img_pad = torch.zeros((batch, channel, 1024, 1024))
        for b in range(batch):
            if not self.downsample_gt:
                img_pad[b:b+1, :, patch_cord[0][b]:patch_cord[2][b], patch_cord[1][b]:patch_cord[3][b]] = patch[b:b+1,:,:,:]
            else:
                start_row = patch_cord[4]
                for i in range(patch_cord[0][b], h + patch_cord[0][b]):
                    img_pad[b:b+1, :, 2*i+start_row[b]:2*i+start_row[b]+1, patch_cord[1][b]:patch_cord[3][b]] = patch[b:b+1,: ,i-patch_cord[0][b]:i+1-patch_cord[0][b],:]
        return img_pad
    def rec_img(self, img_pad, patch_cord):
        assert(img_pad.shape[-2:] == (1024, 1024))
        # img = img_pad[...,12:1012, 12:1012]
        img = img_pad
        amp = 1
        bScanSumOrigin = 1024
        bScanNumOrigin = 1024
        batch, channel, bScanSum, bScanNum = img.shape

        imgReconstruction = torch.zeros_like(img)

        dR = bScanNumOrigin / bScanNum
        dAngle = np.pi / bScanSum
        dRReconstruction = dR / amp
        dAngleReconstruction = dAngle / amp
        offset = (bScanNum / 2) * amp
        for b in range(batch):
            if self.downsample_gt:
                start_row = patch_cord[4][b]
                row_range = range(2*patch_cord[0][b]+start_row, 2*patch_cord[2][b]+start_row, 2)
                column_range = range(patch_cord[1][b], patch_cord[3][b], 1)
                # print('**',2*patch_cord[0][b]+start_row, 2*patch_cord[2][b]+start_row)
            else:
                row_range = range(patch_cord[0][b], patch_cord[2][b])
                column_range = range(patch_cord[1][b], patch_cord[3][b])
            for rowIndex in row_range:
                for columnIndex in column_range:
                    r = (offset - columnIndex) * dRReconstruction
                    # angle = (rowIndex - 1) * dAngleReconstruction
                    angle = rowIndex * dAngleReconstruction
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    i = round(x + bScanNumOrigin/2)-1
                    j = round(y + bScanNumOrigin/2)-1
                    if j > bScanSumOrigin:
                        j = bScanSumOrigin

                    if i > bScanNumOrigin:
                        i = bScanNumOrigin
                    if i < 0:
                        i = 0;
                    if j < 0:
                        j = 0;
                    if imgReconstruction[b:b+1, :, j, i]  < img[b:b+1, :, rowIndex, columnIndex]:
                        imgReconstruction[b:b+1, :, j, i] = img[b:b+1, :, rowIndex, columnIndex]        
        return imgReconstruction
 
    def forward(self, sr, hr):
        patch_cord = sr[-1]
        if isinstance(sr, list):
            sr = sr[0]
        sr = self.pad_img(sr, patch_cord)
        # print('**sr', sr.shape)
        sr_rec = self.rec_img(sr, patch_cord)
        
        hr = self.pad_img(hr, patch_cord)
        hr_rec = self.rec_img(hr, patch_cord)
        # print('sr***', sr.shape, sr_rec.shape, hr_rec.shape)
        out = self.L1(sr_rec, hr_rec)
        # print('loss**', out)
        return out



import enum
from posixpath import split
from turtle import forward
from typing import Mapping
import numpy as np
import torch.nn as nn
import torch
import cv2
from tqdm import tqdm
import torch.nn.functional as F

class entrop(nn.Module):
    def __init__(self):
        super(entrop, self).__init__()
        self.L1 = nn.L1Loss()

    def cal_entrop(self, patch):
        result = 0
        h,w = patch.shape[-2:]
        patch_255 = (patch * 255).int
        hist, bin_edges = torch.histogram(patch_255, bins=torch.arange(257), density=True)
        for i in np.arange(256):
            if hist[i] == 0:
                result = result
            else:
                result = result - hist[i] * (torch.log(hist[i]) / torch.log(2.0))
        return result

    def split_patch(self, img, patch_num=[8,8]):
        patch_list = []
        h,w = img.shape[-2:]
        n_patch_h = patch_num[0]
        n_patch_w = patch_num[1]
        patch_size_h = h // patch_num[0]
        patch_size_w = w // patch_num[1]
        for i in range(n_patch_h):
            for j in range(n_patch_w):
                patch_list.append(img[...,i*patch_size_h:(i+1)*patch_size_h,j*patch_size_w:(j+1)*patch_size_w])
        return patch_list

    def change_patch_val(self, val_list, img):
        out = torch.ones_like(img)
        n_patch_h = int(torch.sqrt(len(val_list)))
        patch_size = img.shape[0] // n_patch_h
        k=0
        for i in range(n_patch_h):
            for j in range(n_patch_h):
                out[..., i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = val_list[k]
                k += 1
        out = (out-out.min()) / out.max()
        return out

    def forward(self, sr, hr):
        if isinstance(sr, list):
            sr = sr[0]
        hr_patch = self.split_patch(hr)
        entrop_list = []
        for i, patch in enumerate(hr_patch):
            entrop_list.append(self.cal_entrop(patch))
        entrop_weight = self.change_patch_val(entrop_list, hr)
        out = self.L1(entrop_weight*sr, entrop_weight*hr)
        # print('**sr', sr.shape)

        return out



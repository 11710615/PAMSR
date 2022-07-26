from turtle import forward
import numpy as np
import torch.nn as nn
import torch
import cv2

class rec_loss(nn.Module):
    def __init__(self):
        super(rec_loss, self).__init__()
    def pad_img(self, patch, patch_cord):
        batch, channel, h, w = patch.shape
        img_pad = torch.zeros((batch, channel, 1024, 1024))
        img_pad[...,patch_cord[0]:patch_cord[2], patch_cord[1]:patch_cord[3]] = patch
        return img_pad
    def rec_img(self, img_pad):
        assert(img_pad.shape[-2:] == (1024, 1024))
        img_center = img_pad[...,12:1012, 12:1012]

    
    def forward(self, x):

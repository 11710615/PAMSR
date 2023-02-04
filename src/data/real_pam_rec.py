import os
from tkinter import E
from unittest import result
from cv2 import split
import torch
import cv2
import numpy as np
import pickle as pkl
import torch.nn.functional as F
import random
import math
from bisect import bisect_left

def padding_zero(img, scale=2):
    """down sample with scale and padding zeros"""
    img_down = img.copy()
    h,w = img.shape[-2:]
    for i in range(1,scale):
        img_down[..., range(i, h, scale),:] = 0
        for j in range(1,scale):
            img_down[..., :, range(j, w, scale)] = 0
    # img_down[...,range(0, h, scale), :] = 0
    # img_down[..., :, range(0, w, scale)] = 0
    return img_down

def padding_zero_lr(lr,scale=2):
    h,w=lr.shape
    out = np.zeros([h*scale,w*scale])
    for i in range(h):
        for j in range(w):
            out[i*scale, j*scale]=lr[i,j]
    return out
    

class BurstSRDataset(torch.utils.data.Dataset):
    """ used for burst dataset
        hr: [b, 1, h/2, w]

        lr is generated from hr with mask"""
    def __init__(self, args, data_id, name='BurstSRDataset', center_crop=False, split='train'):
        """
        args:
            root : path of the root directory
            burst_size : Burst size. Maximum allowed burst size is 14.
            crop_sz: Size of the extracted crop. Maximum allowed crop size is 80
            center_crop: Whether to extract a random crop, or a centered crop.
            random_flip: Whether to apply random horizontal and vertical flip
            split: Can be 'train' or 'val'
        """
        self.burst_size = args.burst_size
        assert self.burst_size <= 25, 'burst_sz must be less than or equal to 14'
        assert split in ['train', 'val', 'val_divided']
        root = args.data_train[0]  # list
        # root = args.dir_data + '/' + root + '/' 
        root = args.dir_data + '/' + root
        super().__init__()
        self.args = args
        # self.burst_size = burst_size
        self.split = split
        self.center_crop = center_crop
        self.name = name
        self.root = root
        self.downsample_gt = args.downsample_gt

        self.substract_black_level = True
        self.white_balance = False

        self.burst_list = self._get_burst_list(data_id)
        if self.split == 'train':

            n_patch = args.batch_size * args.test_every
            n_image = len(self.burst_list)
            if n_image == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patch // n_image, 1)

    def _get_burst_list(self, data_id):
        burst_list = sorted(os.listdir(self.root))  # 'busrst_data/train'
        #print(burst_list)
        # if self.split == 'train':
        #     data_id = data_id[0]
        # else:
        #     data_id = data_id[1]
        # print(data_id,'**')
        out = [burst_list[i] for i in data_id]
        # print('out**', out)
        # k
        return out  # 'brain_1','brain_3',...,'ear_8'

    def get_burst_info(self, burst_id):
        burst_info = {'burst_size': 25, 'burst_name': self.burst_list[burst_id]}
        return burst_info

    def _get_lr_image(self, burst_id, im_id, start_row=0):  # directly read image and to tensor
        ################## get burst_lr ##################
        scale = self.args.scale[0]
        # im_id: 0-24, burst_id:0-56
            # hr_image = cv2.imread(self.root + '/' + self.burst_list[burst_id] + '/' + str(im_id+1) + '_X1_X1.png', 0)
        hr_image = cv2.imread(self.root + '/' + self.burst_list[burst_id], 0)
        
        # median filter to remove noise
        # hr_image = cv2.medianBlur(hr_image, ksize=3)
        
        # padding [1000,1000] -> [1024, 1024]
        hr_image = cv2.copyMakeBorder(hr_image, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0)
        
        if hr_image is None:
            # hr_image = cv2.imread(self.root + '/' + self.burst_list[burst_id] + '/' + '1_X1_X1.png', 0)
            print('path is not exist: ', self.root + '/' + self.burst_list[burst_id] + '/' + str(im_id) + '_X1_X1.png')
        
        lr_image = padding_zero(hr_image, scale=scale)
        lr_image = (lr_image - lr_image.min()) / (lr_image.max() - lr_image.min())
        lr_image = torch.from_numpy(lr_image)
        return lr_image

    def _get_gt_image(self, burst_id, start_row):
        if self.split != 'val_divided':
            gt_img = cv2.imread(self.root + '/' + self.burst_list[burst_id],0)
            # gt_img = cv2.imread(self.root + '/' + self.burst_list[burst_id] + '/1_X1_X1.png',0)
            # median filter to remove noise
            # gt_img = cv2.medianBlur(gt_img, ksize=3)
            gt_img = cv2.copyMakeBorder(gt_img, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0)
        else:
            gt_img = cv2.imread(self.root + '/' + self.burst_list[burst_id], 0)
            # gt_img = cv2.medianBlur(gt_img, ksize=3)
            gt_img = cv2.copyMakeBorder(gt_img, 6, 6, 12, 12, cv2.BORDER_CONSTANT, value=0)
            
        gt_img = (gt_img- gt_img.min()) / (gt_img.max()-gt_img.min())  # norm
        gt_img = torch.from_numpy(gt_img)
        return gt_img

    def get_burst(self, burst_id, im_ids, info=None):
        start_row = np.random.randint(2)  # 0/1
        frames = [self._get_lr_image(burst_id, i, start_row) for i in im_ids]
        gt = self._get_gt_image(burst_id, start_row)
        if info is None:
            info = self.get_burst_info(burst_id)

        return frames, gt, info, start_row

    def _sample_images(self):  # keep ids=0 as the base frame
        # burst_size_max = self.args.burst_size_max
        ids = list(range(self.burst_size))  # select burst img regularly  [0,1,2,3,...]   
        # ids = random.sample(range(1, burst_size_max), k=self.burst_size - 1)
        # ids = [0, ] + ids
        return ids
    
    def _get_crop(self, frames, gt, patch_size=(64,64), scale=2, center_crop=False, patch_select='random'):
        ih, iw = frames[0].shape[:2]  # 256,
        p = scale
        
        tp = [num for num in patch_size]
        ip = [num for num in tp]  # 64

        if center_crop:
            ix = (iw - patch_size[1]) // 2
            iy = (ih - patch_size[0]) // 2
        else:
            stride = 8
            if patch_select=='entrop_fixed':
                iy = random.randrange(0, ih - ip[1] + 1)
                ix_start = random.randrange(0, iw - ip[1] - ip[1] + 1)
                img_entrop = 0
                for i_x in range(ix_start, ix_start + ip[1] + 1, stride):  # select 8 adjent pathces
                    img = frames[0][iy:iy + ip[0], i_x:i_x + ip[1]]
                    entrop_temp = image_entrop_gray(img)
                    if img_entrop < entrop_temp:
                        img_entrop = entrop_temp
                        ix = i_x
            elif patch_select=='entrop_random':
                iy = random.randrange(0, ih - ip[1] + 1)
                ix_start = random.randrange(0, iw - ip[1] - ip[1] + 1)
                entrop_list = []
                for i_x in range(ix_start, ix_start + ip[1] + 1, stride):  # select 8 adjent pathces
                    img = frames[0][iy:iy + ip[0], i_x:i_x + ip[1]]
                    entrop_temp = image_entrop_gray(img)
                    entrop_list.append(entrop_temp)
                entrop_array = np.array(entrop_list)
                entrop_array = (entrop_array - entrop_array.min()) / (entrop_array.max()- entrop_array.min())
                entrop_array = entrop_array * np.random.rand(1, len(entrop_list))
                ix = np.argmax(entrop_array) * stride + ix_start
            elif patch_select=='grad_random':
                iy = random.randrange(0, ih - ip[1] + 1)
                ix_start = random.randrange(0, iw - ip[1] - ip[1] + 1)
                grad_list = []
                for i_x in range(ix_start, ix_start + ip[1] + 1, stride):  # select 8 adjent pathces
                    img = frames[0][iy:iy + ip[0], i_x:i_x + ip[1]]
                    grad_temp = image_grad_gray(img)
                    grad_list.append(grad_temp)
                grad_array = np.array(grad_list)
                grad_array = (grad_array - grad_array.min()) / (grad_array.max()- grad_array.min())
                grad_array = grad_array * np.random.rand(1, len(grad_list))
                ix = np.argmax(grad_array) * stride + ix_start

            elif patch_select == 'entrop_window':
                img = frames[0].numpy()
                window_num = [ih//ip[0], iw//ip[1]]
                img_window = split_patch(img, window_num)
                entrop_list = [image_entrop_gray(window) for window in img_window]
                entrop_cprob = cal_cprob(entrop_list)
                window_idx = bisect_left(entrop_cprob, random.uniform(0,1)) - 1
                window_idx_h = window_idx // window_num[1] 
                window_idx_w = window_idx % window_num[1]
                flag_h = 1 if window_idx_h == (window_num[0]-1) else 0
                flag_w = 1 if window_idx_w == (window_num[1]-1) else 0
                iy = random.randrange(0, ih // window_num[0] - flag_h * ip[0] + 1) + window_idx_h * (ih // window_num[0])
                ix = random.randrange(0, iw // window_num[1] - flag_w * ip[1] + 1) + window_idx_w * (iw // window_num[1])

            elif patch_select=='grad_window':
                img = frames[0].numpy()
                window_num = [ih//ip[0], iw//ip[1]]
                img_window = split_patch(img, window_num)
                entrop_list = [image_grad_gray(window) for window in img_window]
                entrop_cprob = cal_cprob(entrop_list)
                window_idx = bisect_left(entrop_cprob, random.uniform(0,1)) - 1
                window_idx_h = window_idx // window_num[1] 
                window_idx_w = window_idx % window_num[1]
                flag_h = 1 if window_idx_h == (window_num[0]-1) else 0
                flag_w = 1 if window_idx_w == (window_num[1]-1) else 0
                iy = random.randrange(0, ih // window_num[0] - flag_h * ip[0] + 1) + window_idx_h * (ih // window_num[0])
                ix = random.randrange(0, iw // window_num[1] - flag_w * ip[1] + 1) + window_idx_w * (iw // window_num[1])

            elif patch_select=='random':
                ix = random.randrange(0, iw - ip[1] + 1)
                iy = random.randrange(0, ih - ip[0] + 1)
                
        tx, ty = ix, iy
        ret = [
            [img[iy:iy + ip[0], ix:ix + ip[1]] for img in frames],
            gt[ty:(ty+tp[0]), tx:(tx+tp[1])]
        ]
        # cord information
        cord = [ty, tx, ty + tp[0], tx + tp[1]]
        return ret, cord  # ret[0]: burst list, ret[1]: gt
    
    def _augment(self, frames, gt, hflip=True, rot=True): 
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5
        
        def _augment(img):
            if hflip: img = img.flip(-1)
            if vflip: img = img.flip(-2)
            if rot90: img = img.transpose(1, 0)
            return img
        return [[_augment(img) for img in frames],
                _augment(gt)]        
    
    def __len__(self):
        if self.split == 'train':
            return len(self.burst_list) * self.repeat
        else:
            return len(self.burst_list) 

    def _get_index(self, idx):
        if self.split == 'train':
            return idx % len(self.burst_list)
        else:
            return idx

    def __getitem__(self, index):
        index = self._get_index(index)
        # Sample the images in the burst, in case a burst_size_max < len(self.burst_list) is used.
        im_ids = self._sample_images()

        # Read the burst images along with HR ground truth
        frames, gt, meta_info, start_row = self.get_burst(index, im_ids)
        # Extract crop if needed
        if self.split == 'train':
            ret, patch_cord = self._get_crop(frames, gt, patch_size=self.args.patch_size, scale=self.args.scale[0], patch_select=self.args.patch_select)
            burst_list, gt = ret
            if not self.args.no_augment:
                burst_list, gt = self._augment(burst_list, gt)
        else:
            # burst_list, gt = self._get_crop(frames, gt, patch_size=512//self.args.scale[0], scale=self.args.scale[0], center_crop=True)
            ret, patch_cord = self._get_crop(frames, gt, patch_size=self.args.test_patch_size, scale=self.args.scale[0], center_crop=True)
            burst_list, gt = ret

        patch_cord.append(start_row)
        # unsqueence
        burst_list = [img.unsqueeze(0) for img in burst_list]
        gt = gt.unsqueeze(0).float()
        burst = torch.stack(burst_list, dim=0).float()  # [5,1,h,w]，封装后，添加batch
        if self.burst_size == 1:
            burst = burst.squeeze(0)  # [1,1,h,w]->[1,h,w]
        
        # print('start_row, ***', patch_cord[4], gt.shape, burst.shape)
        return burst, gt, meta_info, patch_cord  # dict

def image_entrop_gray(img):  # input:[0,1]
    result = 0
    if isinstance(img, torch.Tensor):
        img = img.detach().numpy() * 255
    hist, bin_edges = np.histogram(img, bins=np.arange(257), density=True)
    for i in np.arange(256):
        if hist[i] == 0:
            result = result
        else:
            result = result - hist[i] * (math.log(hist[i]) / math.log(2.0))
    return result

def image_grad_gray(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().numpy() * 255
    img_x = cv2.Sobel(img, -1, 1, 0)
    img_y = cv2.Sobel(img, -1, 0, 1)
    grad=cv2.addWeighted(img_x,0.5,img_y,0.5,0)
    grad_sum = grad.sum()

    return grad_sum

def cal_cprob(entrop_list):
    # normalize    
    entrop_array = np.array(entrop_list)
    entrop_array = (entrop_array - entrop_array.min()) / (entrop_array.max()-entrop_array.min())
    entrop_array = entrop_array / np.sum(entrop_array)
    entrop_prob = entrop_array
    entrop_cprob = [0]
    for i in range(len(entrop_list)):
        entrop_cprob.append(entrop_prob[i]+entrop_cprob[i])
    entrop_cprob[-1] = 1.0
    return entrop_cprob

def split_patch(img, patch_num=[4,4]):
    if isinstance(img, torch.Tensor):
        img = img.detach().numpy()
    patch_list = []
    [h,w] = img.shape[-2:]
    n_patch_h = patch_num[0]
    n_patch_w = patch_num[1]
    patch_size_h = h // patch_num[0]
    patch_size_w = w // patch_num[1]
    for i in range(n_patch_h):
        for j in range(n_patch_w):
            patch_list.append(img[...,i*patch_size_h:(i+1)*patch_size_h,j*patch_size_w:(j+1)*patch_size_w])
    return patch_list
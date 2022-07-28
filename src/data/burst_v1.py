import os
from cv2 import split
import torch
import cv2
import numpy as np
import pickle as pkl
import torch.nn.functional as F
import random


def down_sample(img, scale):
    """down sample with scale"""
    h,w = img.shape[-2:]
    img_down = img[...,range(0, h, scale), :]
    img_down = img_down[..., :, range(0, w, scale)]

    return img_down

class BurstSRDataset(torch.utils.data.Dataset):
    """ used for burst dataset
        hr: [b, 1, h/2, w]

        lr is generated from hr with mask"""
    def __init__(self, args, name='BurstSRDataset', center_crop=False, split='train'):
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
        root = args.dir_data + '/' + root + '/' + split
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

        self.burst_list = self._get_burst_list()
        if self.split == 'train':

            n_patch = args.batch_size * args.test_every
            n_image = len(self.burst_list)
            if n_image == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patch // n_image, 1)

    def _get_burst_list(self):
        burst_list = sorted(os.listdir(self.root))  # 'busrst_data/train'
        #print(burst_list)
        return burst_list  # 'brain_1','brain_3',...,'ear_8'

    def get_burst_info(self, burst_id):
        burst_info = {'burst_size': 25, 'burst_name': self.burst_list[burst_id]}
        return burst_info

    def _get_lr_image(self, burst_id, im_id, start_row=0):  # directly read image and to tensor
        ################## get burst_lr ##################
        scale = self.args.scale[0]
        if self.split != 'val_divided':
            # im_id: 0-24, burst_id:0-56
            if not os.path.exists(self.root + '/' + self.burst_list[burst_id] + '/' + str(im_id+1) + '_X1_X1.png'):
                hr_image = cv2.imread(self.root + '/' + self.burst_list[burst_id] + '/' + '1_X1_X1.png', 0)
            else:
                hr_image = cv2.imread(self.root + '/' + self.burst_list[burst_id] + '/' + str(im_id+1) + '_X1_X1.png', 0)
            
            # median filter to remove noise
            hr_image = cv2.medianBlur(hr_image, ksize=3)
            
            # padding [1000,1000] -> [1024, 1024]
            hr_image = cv2.copyMakeBorder(hr_image, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0)
            
            if hr_image is None:
                # hr_image = cv2.imread(self.root + '/' + self.burst_list[burst_id] + '/' + '1_X1_X1.png', 0)
                print('path is not exist: ', self.root + '/' + self.burst_list[burst_id] + '/' + str(im_id) + '_X1_X1.png')
            
            if self.downsample_gt:
                h,_ = hr_image.shape[-2:]
                hr_image = hr_image[...,range(start_row,h,2),:]

        else:
            hr_image = cv2.imread(self.root + '/' + self.burst_list[burst_id], 0)

            # median filter to remove noise
            hr_image = cv2.medianBlur(hr_image, ksize=3)

            hr_image = cv2.copyMakeBorder(hr_image, 6, 6, 12, 12, cv2.BORDER_CONSTANT, value=0)  # [512,1024]
            if hr_image is None:
                print('path is not exist: ', self.root + '/' + self.burst_list[burst_id])
        lr_image = down_sample(hr_image, scale=scale)
        lr_image = (lr_image - lr_image.min()) / lr_image.max()
        lr_image = torch.from_numpy(lr_image)
        return lr_image

    def _get_gt_image(self, burst_id, start_row):
        if self.split != 'val_divided':
            gt_img = cv2.imread(self.root + '/' + self.burst_list[burst_id] + '/1_X1_X1.png',0)
            # median filter to remove noise
            gt_img = cv2.medianBlur(gt_img, ksize=3)
            gt_img = cv2.copyMakeBorder(gt_img, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0)
            if self.downsample_gt:
                h,_ = gt_img.shape[-2:]
                gt_img = gt_img[...,range(start_row,h,2),:]
        else:
            gt_img = cv2.imread(self.root + '/' + self.burst_list[burst_id], 0)
            gt_img = cv2.medianBlur(gt_img, ksize=3)
            gt_img = cv2.copyMakeBorder(gt_img, 6, 6, 12, 12, cv2.BORDER_CONSTANT, value=0)
            
        gt_img = (gt_img- gt_img.min()) / gt_img.max()  # norm
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
        ids = list(range(self.burst_size))  # select burst img regularly     
        # ids = random.sample(range(1, burst_size_max), k=self.burst_size - 1)
        # ids = [0, ] + ids
        return ids
    
    def _get_crop(self, frames, gt, patch_size=(64,64), scale=2, center_crop=False):
        ih, iw = frames[0].shape[:2]  # 256,512
        p = scale
        
        tp = [num * scale for num in patch_size]
        ip = [num // scale for num in tp]  # 64

        if center_crop:
            ix = (iw - patch_size[1]) // 2
            iy = (ih - patch_size[0]) // 2
        else:
            ix = random.randrange(0, iw - ip[1] + 1)
            iy = random.randrange(0, ih - ip[0] + 1)
        
        tx, ty = scale * ix, scale * iy
        ret = [
            [img[iy:iy + ip[0], ix:ix + ip[1]] for img in frames],
            gt[ty:ty + tp[0], tx:tx + tp[1]]
        ]
        # print('tp**',)
        # cord information
        cord = [ty, tx, ty + tp[0], tx + tp[1]]
        # print('***', ty,tx, ip[0], ip[1])
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

        # print('***', frames[0].shape, gt.shape)
        # k
        # Extract crop if needed
        if self.split == 'train':
            ret, patch_cord = self._get_crop(frames, gt, patch_size=self.args.patch_size, scale=self.args.scale[0])
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


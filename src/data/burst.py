import os
from cv2 import split
import torch
import cv2
import numpy as np
import pickle as pkl
import torch.nn.functional as F
import random

class BurstSRDataset(torch.utils.data.Dataset):
    """ Real-world burst super-resolution dataset. """
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
        assert self.burst_size <= 50, 'burst_sz must be less than or equal to 14'
        assert split in ['train', 'val']
        root = args.data_train[0]  # list
        root = args.dir_data + '/' + root + '/' + split
        super().__init__()
        self.args = args
        # self.burst_size = burst_size
        self.split = split
        self.center_crop = center_crop
        self.name = name
        self.root = root

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
        burst_list = sorted(os.listdir('{}/x1'.format(self.root)))  # 'busrst_data/train'
        #print(burst_list)
        return burst_list  # 'brain_1','brain_3',...,'ear_8'

    def get_burst_info(self, burst_id):
        burst_info = {'burst_size': 50, 'burst_name': self.burst_list[burst_id]}
        return burst_info

    def _get_raw_image(self, burst_id, im_id):  # directly read image and to tensor
        # im_id: 0-49, burst_id:0-30
        idx_temp = self.burst_list[burst_id].find('_') + 1
        raw_image = cv2.imread('{}/x{}/{}/{}_{}x{}.png'.format(self.root, str(self.args.scale[0]), self.burst_list[burst_id], self.burst_list[burst_id][idx_temp:], im_id+1, self.args.scale[0]),0)
        if raw_image is None:
            print('path', '{}/x{}/{}/{}_{}x{}.png'.format(self.root, str(self.args.scale[0]), self.burst_list[burst_id], self.burst_list[burst_id][idx_temp:], im_id+1, self.args.scale[0]))
        raw_image = raw_image / raw_image.max()
        raw_image = torch.from_numpy(raw_image)
        
        # raw_image = SamsungRAWImage.load('{}/{}/samsung_{:02d}'.format(self.root, self.burst_list[burst_id], im_id))
        return raw_image

    def _get_gt_image(self, burst_id):
        idx_temp = self.burst_list[burst_id].find('_') + 1
        gt_img = cv2.imread('{}/x1/{}/{}_1.png'.format(self.root, self.burst_list[burst_id], self.burst_list[burst_id][idx_temp:]),0)
        gt_img = gt_img / gt_img.max()  # norm
        gt_img = torch.from_numpy(gt_img)
        return gt_img

    def get_burst(self, burst_id, im_ids, info=None):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]

        gt = self._get_gt_image(burst_id)
        if info is None:
            info = self.get_burst_info(burst_id)

        return frames, gt, info

    def _sample_images(self):  # keep ids=0 as the base frame
        burst_size_max = 50
        ids = random.sample(range(1, burst_size_max), k=self.burst_size - 1)
        ids = [0, ] + ids
        return ids
    
    def _get_crop(self, frames, gt, patch_size=64, scale=4, center_crop=False):

        ih, iw = frames[0].shape[:2]  # 250,250
        p = scale
        tp = p * patch_size  # 64*2=128
        ip = tp // scale  # 64
        
        if center_crop:
            ix = (iw - patch_size) // 2
            iy = (ih - patch_size) // 2
        else:
            ix = random.randrange(0, iw - ip + 1)
            iy = random.randrange(0, ih - ip + 1)
        
        tx, ty = scale * ix, scale * iy
        
        ret = [
            [img[iy:iy + ip, ix:ix + ip] for img in frames],
            gt[ty:ty + tp, tx:tx + tp]
        ]
        return ret  # ret[0]: burst list, ret[1]: gt
    
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
        frames, gt, meta_info = self.get_burst(index, im_ids)

        # Extract crop if needed
        if self.split == 'train':
            burst_list, gt = self._get_crop(frames, gt, patch_size=self.args.patch_size, scale=self.args.scale[0])
            if not self.args.no_augment:
                burst_list, gt = self._augment(burst_list, gt)
        else:
            # burst_list, gt = self._get_crop(frames, gt, patch_size=512//self.args.scale[0], scale=self.args.scale[0], center_crop=True)
            burst_list, gt = self._get_crop(frames, gt, patch_size=1000//self.args.scale[0], scale=self.args.scale[0], center_crop=False)
        # unsqueence
        burst_list = [img.unsqueeze(0) for img in burst_list]
        gt = gt.unsqueeze(0).float()
        burst = torch.stack(burst_list, dim=0).float()

        return burst, gt, meta_info  # dict


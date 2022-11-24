import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from kornia.losses import ssim_loss as ssim_m


# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#     return gauss/gauss.sum()

# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window

# def _ssim(img1, img2, window, window_size, channel, reduction = 'mean'):
#     mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
#     mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1*mu2

#     sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
#     sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

#     C1 = 0.01**2
#     C2 = 0.03**2

#     ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
#     # print('ssim:', ssim_map.shape, ssim_map.max().item(), ssim_map.min().item())

#     if reduction == 'mean':
#         return ssim_map.mean()
#     elif reduction == 'none':
#         return ssim_map
#     elif reduction == 'navg':
#         return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        rec_map = np.load('./loss/rec_map.npy', allow_pickle=True).item()
        self.h_idx = torch.from_numpy(rec_map['h_idx']).type(torch.long)
        self.w_idx = torch.from_numpy(rec_map['w_idx']).type(torch.long)
        
    def rec_img(self, img, patch_cord):
        # patch_cord = [h,w,h_size, w_size, start_row][b]
        b,c,*_ = img.shape
        imgReconstruction = torch.zeros([b,c,1024,1024]).type_as(img)  # [b,1,h,w]
        batch_idx_list = []
        for i in range(b):
            h_ii = self.h_idx[patch_cord[0][i]:patch_cord[2][i], patch_cord[1][i]:patch_cord[3][i]].flatten()
            w_ii = self.w_idx[patch_cord[0][i]:patch_cord[2][i], patch_cord[1][i]:patch_cord[3][i]].flatten()
            # print('***',h_ii.shape, len(w_ii), img.shape,patch_cord[0][i],patch_cord[2][i], patch_cord[1][i],patch_cord[3][i])
            imgReconstruction[i:i+1,:,h_ii, w_ii] = img[i:i+1,:,:,:].flatten()
            batch_idx_list.append([np.array(h_ii).min(),np.array(w_ii).min(), np.array(h_ii).max(), np.array(w_ii).max()])
        return imgReconstruction, batch_idx_list
    
    def forward(self, sr, hr):
        patch_cord = sr[-1]
        if isinstance(sr, list):
            sr = sr[0]
        sr_rec, batch_idx_list = self.rec_img(sr, patch_cord)
        hr_rec, _ = self.rec_img(hr, patch_cord)
        (batch, _, _, _) = sr.size()
        ssim = 0
        for b in range(batch):
            idx = batch_idx_list[b]
            out = ssim_m(sr_rec[b:b+1,:, idx[0]:idx[2], idx[1]:idx[3]], hr_rec[b:b+1,:, idx[0]:idx[2], idx[1]:idx[3]], 5)
            ssim = ssim + out
        ssim_loss_value = ssim.div(b)
        # print(ssim_loss_value, '***')
        return ssim_loss_value

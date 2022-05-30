import torch
import torch.nn as nn
import torch.nn.functional as F

class FMAE(nn.Module):
    def __init__(self):
        super(FMAE, self).__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, sr, hr):
        sr_fft = torch.fft.fft2(sr, norm='forward')
        hr_fft = torch.fft.fft2(hr, norm='forward')
        sr_fft_abs = sr_fft.abs()
        hr_fft_abs = hr_fft.abs()
        loss = self.l1_loss(sr_fft_abs, hr_fft_abs)
        return loss

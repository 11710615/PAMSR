from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from loss.UNetFamily import U_Net

class topology(nn.Module):
    def __init__(self):
        super(topology, self).__init__()

        self.net = U_Net(img_ch=1, output_ch=2)
        state_dict = torch.load('loss/unet_model.pth')
        self.net.load_state_dict(state_dict['net'])

        # stop grad
        for p in self.parameters():
            p.requires_grad = False

        self.L1 = nn.L1Loss()

    def forward(self, sr, hr):
        if hr.shape[1]==3:  # input must have ONE channel
            hr = hr[:,0,:]

        if isinstance(sr,list):
            sr = sr[0]
        elif isinstance(sr, tuple):
            sr = sr[0]

        if sr.shape[1]==3:  # vgg input must have 3 channels
            sr = sr[:,0,:]

        with torch.no_grad():
            hr_topology = self.net(hr.detach())
            hr_topology = hr_topology[:,0:1,:]

        # print('hr_topology', torch.max(hr_topology * sr), torch.min(hr_topology * sr))
        # print(torch.max(hr_topology ), torch.min(hr_topology))
        # print(torch.max(hr), torch.min(hr))
        # r
        loss = self.L1(hr_topology * sr, hr_topology * hr)

        return loss

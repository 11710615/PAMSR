from modulefinder import Module
from turtle import forward
import torch
import torch.nn as nn

# fd-unet for PAM Rec

def make_model(args):
    return FD_UNet(f_in=32)

class FD_UNet(nn.Module):
    def __init__(self, f_in=32):
        super(FD_UNet, self).__init__()

        self.conv1 = nn.Sequential(*[nn.Conv2d(1, 32, 3, 1, padding=1),nn.ELU(), nn.BatchNorm2d(32)])
        self.DB1 = FD_Block(f_in=32, f_out=64)

        self.down1 = DownBlock(64)
        self.down2 = DownBlock(128)
        self.down3 = DownBlock(256)
        self.down4 = DownBlock(512)

        self.bottle = UpSample(1024)

        self.up1 = UpBlock(1024)
        self.up2 = UpBlock(512)
        self.up3 = UpBlock(256)
        
        self.conv2 = nn.Sequential(*[nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0), nn.ELU(), nn.BatchNorm2d(32)])
        self.DB2 = FD_Block(32, 64)
        self.conv3 = nn.Sequential(*[nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(1)])

    def forward(self, img):
        shortcut1 = img.clone()
        out = self.conv1(img)

        out = self.DB1(out)     #[8,64,128,128]
        shortcut2 = out.clone()


        out = self.down1(out)  # [8,128,64,64]
        shortcut3 = out.clone()


        out = self.down2(out)
        shortcut4 = out.clone()


        out = self.down3(out)  # [8, 512, 16, 16]
        shortcut5 = out.clone()


        out = self.down4(out)  # [8,1024,8,8]
      
        out = self.bottle(out) # [8,512,16,16]
        out = torch.cat([out, shortcut5], 1)
        
        out = self.up1(out)
        out = torch.cat([out, shortcut4], dim=1)
    
        out = self.up2(out)
        out = torch.cat([out, shortcut3], dim=1)

        out = self.up3(out)
        out = torch.cat([out, shortcut2], dim=1)

        out = self.conv2(out)
        out = self.DB2(out)
        out = self.conv3(out) + shortcut1
        return out

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
                                   
class FD_Block(nn.Module):
    def __init__(self, f_in=32, f_out=64):
        super(FD_Block, self).__init__()
        self.model = []
        k = f_in // 4
        for i in range(f_in, f_out, k):
            model_list = []
            # 1x1 cov
            model_list += [nn.Conv2d(i, f_in, kernel_size=1, stride=1, padding='same'), nn.ELU(), nn.BatchNorm2d(f_in)]
            model_list += [nn.Conv2d(f_in, k, kernel_size=3, stride=1, padding=1), nn.ELU(), nn.BatchNorm2d(k)]
            self.model.append(nn.Sequential(*model_list))
        self.model = nn.ModuleList(self.model)


    def forward(self, img):
        out = img
        for model_temp in self.model:
            # model_temp = model_temp.to(img.device)
            out_temp = model_temp(out)
            out = torch.cat([out_temp, out], dim=1)
        return out

class DownSample(nn.Module):
    def __init__(self, f_in):
        super(DownSample, self).__init__()

        model = [nn.Conv2d(f_in, f_in, kernel_size=1, padding='same', stride=1), nn.ELU(), nn.BatchNorm2d(f_in)]
        model += [nn.Conv2d(f_in, f_in, kernel_size=3, padding=1, stride=2), nn.ELU(), nn.BatchNorm2d(f_in)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        out = self.model(input)
        return out

class UpSample(nn.Module):
    def __init__(self, f_in):
        super(UpSample, self).__init__()        

        model = [nn.Conv2d(f_in, f_in, kernel_size=1, stride=1, padding='same'), nn.ELU(), nn.BatchNorm2d(f_in)]
        model += [nn.ConvTranspose2d(f_in, f_in//2, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ELU(), nn.BatchNorm2d(f_in//2)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        out = self.model(input)
        return out

class UpBlock(nn.Module):
    def __init__(self, f_in):
        super(UpBlock, self).__init__()
        self.db = FD_Block(f_in//4, f_in//2)
        self.conv = nn.Sequential(*[nn.Conv2d(f_in, f_in//4, kernel_size=1, padding='same', stride=1), nn.ELU(), nn.BatchNorm2d(f_in//4)])
        self.up = UpSample(f_in//2)
    def forward(self, input):
        out = self.conv(input)
        out = self.db(out)
        out = self.up(out)
        return out

class DownBlock(nn.Module):
    def __init__(self, f_in):
        super(DownBlock, self).__init__()    
        self.down = DownSample(f_in)
        self.db = FD_Block(f_in, f_in*2)
    def forward(self, input):

        out = self.down(input)
        out = self.db(out)
        return out
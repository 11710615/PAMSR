from model import common
from model import attention
import torch.nn as nn

def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return PAEDSR(args, dilated.dilated_conv)
    else:
        return PAEDSR(args)

class PAEDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(PAEDSR, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.msa = attention.PyramidAttention(channel=256, reduction=8,res_scale=args.res_scale);         
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock//2)
        ]
        m_body.append(self.msa)
        for _ in range(n_resblock//2):
            m_body.append( common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ))
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, 1, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        
        return x 

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
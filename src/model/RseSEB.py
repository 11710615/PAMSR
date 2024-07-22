import torch
import torch.nn as nn

def make_model(args):
    return ResSEB(args)

class ResSEB(nn.Module):
    def __init__(self, args):
        super().__init__()
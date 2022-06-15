from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import copy

class Style(nn.Module):
    def __init__(self, rgb_range=1, n_colors=3, RL1=False):
        super(Style, self).__init__()
        self.vgg_features = models.vgg19(pretrained=True).features
        for p in self.parameters():
            p.requires_grad = False
        self.n_colors = n_colors

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = models.vgg19(pretrained=True).features.eval()
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.cnn_normalization_std = torch.tensor([0.229*rgb_range, 0.224*rgb_range, 0.225*rgb_range]).to(device)
        self.RL1 = RL1
    def forward(self, sr, hr):
        if self.RL1 is True:
            sr = sr[0]
        if isinstance(sr,list):
            sr = sr[0]
        if self.n_colors==1:
            sr = sr.repeat(1,3,1,1)
            hr = hr.repeat(1,3,1,1).detach()
        else:
            sr = sr
        with torch.no_grad():
            hr = hr.detach()
        # if torch.max(hr) > 1:
        #     sr = sr / 255
        #     hr = hr / 255   
        # hr_gm = cal_GramMatrix(hr, self.vgg_features)
        # sr_gm = cal_GramMatrix(sr, self.vgg_features)
        # loss = 0
        # for i in range(len(hr_gm)):
        #     loss = loss + F.mse_loss(hr_gm[i], sr_gm[i])

        temp_model, style_losses = get_style_model_and_losses(
        self.cnn, self.cnn_normalization_mean, self.cnn_normalization_std,hr)
        temp_model(sr)
        style_score = 0
        for sl in style_losses:
            style_score += sl.loss
        return style_score


class ContentLoss(nn.Module):
    
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        # self.std = torch.tensor(std).view(-1, 1, 1)
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)     

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img,
                               style_layers=['conv_2', 'conv_4', 'conv_7', 'conv_10']):
    # At runtime, CNN is a pretrained VGG19 CNN network.
    cnn = copy.deepcopy(cnn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    model = nn.Sequential(normalization)

    style_losses = []

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        # The first layer simply puts names to things and replaces ReLU inplace
        # (which is optimized) with ReLU reallocated. This is a small optimization
        # being removed, and hence a small performance penalty, necessitated by
        # ContentLoss and StyleLoss not working well when inplace=True.
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        # add_module is a setter that is pretty much a setattr equivalent, used for
        # registering the layer with PyTorch.
        model.add_module(name, layer)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses

# def cal_GramMatrix(input_img, vgg_features):
#     assert(input_img.shape[1]==3)
#     layer_chosed = [2,7,14,21]  # 2th, 4th, 7th, 10th conv layers
#     GM_list = []
#     feat = input_img
#     for i, layer in enumerate(vgg_features):
#         feat = layer(feat)
#         if i in layer_chosed:
#             b,d,h,w = feat.shape
#             GM_temp = torch.einsum('b d h w, b z h w -> b d z', feat, feat)
#             GM_list.append(GM_temp/(b*d*h*w))
#             if i > layer_chosed[-1]:
#                 break
#     return GM_list

# def cal_StyleLoss(input1, input2, vgg_features):
#     l2_crition = torch.nn.MSELoss()
#     GM_sr = cal_GramMatrix(input1, vgg_features)
#     GM_hr = cal_GramMatrix(input2, vgg_features)
#     L_style = 0
#     for i in range(len(GM_sr)):
#         L_style = L_style + l2_crition(GM_sr[i], GM_hr[i])
#     return L_style/4
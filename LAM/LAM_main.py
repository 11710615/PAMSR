#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2021-03-03 20:40:27
LastEditTime: 2021-03-03 23:00:43
Description: file content
'''
import torch, cv2, os, sys, numpy as np, matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append(os.path.abspath('..'))
from utils import load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one
from mode import get_model, load_model, print_network
from SaliencyModel.utils import vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, grad_norm, prepare_images, make_pil_grid, blend_input
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.attributes import attr_grad
from SaliencyModel.BackProp import I_gradient, attribution_objective, Path_gradient
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.BackProp import GaussianBlurPath
from SaliencyModel.utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel

import inspect
from gpu_mem_track import MemTracker

import model.swinir
import model.edsr
import model.rdn
import model.han
import model.swinir_ca

# frame = inspect.currentframe()
# gpu_tracker = MemTracker(frame)
k=1
mode = 'rdn'
mode_dict = {'swinir':'RGB','han':'L','swinir_ca':'RGB','edsr':'RGB','rdn':'L'}
state_dict_path = './state_dict/'+mode+'_best.pt'
state_dict = torch.load(state_dict_path, map_location='cuda:0')  
model = model.rdn.make_model()
model.load_state_dict(state_dict)

## The code needs the support of GPU. 
## method_name and checkpoint_path
# model = load_model('HAN', './best.pth')  # You can Change the model name to load different models

## Define windoes_size of D
window_size = 50

## input image
# img_lr, img_hr = prepare_images('./img/brain_1.png')  # [500,500], [1000,1000]
img_lr = Image.open('./img/brain_2x2.png')
img_hr = Image.open('./img/brain_2.png')

if mode in ['swinir','swinir_ca','edsr']:
    box = (200,200,400,400)
    img_lr = img_lr.crop(box)
    img_hr = img_hr.crop((400,400,800,800))
if mode in ['rdn','han']:
    img_lr = pil_to_cv2(img_lr)  # [h,w,3]
    img_hr = pil_to_cv2(img_hr)
    img_lr = img_lr[200:400,200:400,:]
    img_hr = img_hr[400:800,400:800,:]
    img_lr = Image.fromarray(cv2.cvtColor(img_lr.astype(np.uint8), cv2.COLOR_BGR2GRAY))
    img_hr = Image.fromarray(cv2.cvtColor(img_hr.astype(np.uint8), cv2.COLOR_BGR2GRAY))


w = 120  # The x coordinate of your select patch
h = 120# The y coordinate of your select patch [120,120],[120,170] 

tensor_lr = PIL2Tensor(img_lr)[:3] ; tensor_hr = PIL2Tensor(img_hr)[:3] # [1,500,500]
cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2) ; cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2) # [500,500,1]

draw_img = pil_to_cv2(img_hr)
cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
position_pil = cv2_to_pil(draw_img)

sigma = 1.2 ; fold = 50 ; l = 9 ; alpha = 0.5

attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)


# interpolated_grad_numpy: [50,3,500,500]
interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective, gaus_blur_path_func, cuda=True, device=3)

# mean
grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)

# abs normalize
abs_normed_grad_numpy = grad_abs_norm(grad_numpy)

# zoom to HR.size
saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=2)

saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
pil = make_pil_grid(
    [position_pil,  # hr
     saliency_image_abs, # 
     blend_abs_and_input,
     blend_kde_and_input,
     Tensor2PIL(torch.clamp(result, min=0., max=1.), mode=mode_dict[mode])]  # 截断到[0,1]
)

pil.save('./LAM_results/'+mode+'_'+str(k)+'_result.png')
# plt.imshow(pil)
# plt.show()
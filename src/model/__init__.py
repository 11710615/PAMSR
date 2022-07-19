from heapq import merge
import os
from importlib import import_module

from requests import patch

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo
import numpy as np

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = (args.model == 'VDSR')
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        # self.n_GPUs = args.n_GPUs
        self.gpu_ids = args.gpu_ids
        # print('***',self.gpu_ids)
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()

        self.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.training:
            # if self.n_GPUs > 1:
            #     return P.data_parallel(self.model, x, range(self.n_GPUs))
            if len(self.gpu_ids) > 1:
                return P.data_parallel(self.model, x, self.gpu_ids)
            else:
                return self.model(x)
        else:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                out = forward_function(x)
                return out

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train:
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    # def split_patch(self,img, shave, n_patch=4):
    #     h, w = img[0].size()[-2:]
    #     n_p = int(np.sqrt(n_patch))
    #     patch_h = h // n_p
    #     patch_w = w // n_p
    #     print('patch',patch_h,h,w)
    #     patch_list = []
    #     for i in range(n_p):
    #         if i == n_p-1:
    #             h0 = slice(i*patch_h-2*shave, (i+1)*patch_h)
    #         elif i==0:
    #             h0 = slice(i*patch_h,(i+1)*patch_h+2*shave)
    #         else:
    #             h0 = slice(i*patch_h-shave, (i+1)*patch_h+shave)
    #         for j in range(n_p):
    #             if j == n_p-1:
    #                 w0 = slice(j*patch_w-2*shave, (j+1)*patch_w)
    #             elif j == 0:
    #                 w0 = slice(j*patch_w, (j+1)*patch_w+2*shave)
    #             else:
    #                 w0 = slice(j*patch_w-shave, (j+1)*patch_w+shave)
    #             patch_list.append(img[...,h0, w0])
            
    #     return torch.cat(patch_list)

    # def merge_patch(self, patch_list, shave):
    #     patch_list = patch_list[0]
    #     b, c, patch_h, patch_w = patch_list[0].size()
    #     n_p = int(np.sqrt(len(patch_list)))
    #     out_h = int((patch_h - 4*shave)*n_p)
    #     out_w = int((patch_w - 4*shave)*n_p)
    #     merge_out = torch.zeros([b, c, out_h, out_w])
    #     print('*****merge', merge_out.shape, out_h,patch_h,n_p)
    #     patch_idx = 0
    #     for i in range(n_p):
    #         patch_idx = i * n_p
    #         if i == n_p-1:
    #             h0 = slice(-(patch_h-4*shave), None)
    #         elif i == 0:
    #             h0 = slice(0, (patch_h-4*shave))
    #         else:
    #             h0 = slice(2*shave, -2*shave)
    #         h_merge = slice(i*(patch_h-4*shave),(i+1)*(patch_h-4*shave))
    #         for j in range(n_p):
    #             patch_idx_temp = patch_idx + j
    #             if j == n_p-1:
    #                 w0 = slice(-(patch_w-4*shave), None)
    #             elif j == 0:
    #                 w0 = slice(0, patch_w-4*shave)
    #             else:
    #                 w0 = slice(2*shave, -2*shave)
    #             w_merge = slice(j*(patch_w-4*shave), (j+1)*(patch_w-4*shave))
    #             merge_out[..., h_merge, w_merge] = patch_list[patch_idx_temp][...,h0, w0]
    #     return merge_out

    def forward_chop(self, *args, shave=10, min_size=160000):
        scale = 1 if self.input_large else self.scale[self.idx_scale]
        # n_GPUs = min(self.n_GPUs, 4)
        n_GPUs = min(len(self.gpu_ids), 4)
        # height, width
        h, w = args[0].size()[-2:]
        # shave = int(h % 4)
        top = slice(0, h//2 + shave)
        bottom = slice(h - h//2 - shave, h)
        left = slice(0, w//2 + shave)
        right = slice(w - w//2 - shave, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]
        # n_patch = 4
        # x_chops = [self.split_patch(a, shave, n_patch) for a in args]

        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                # y = P.data_parallel(self.model, *x, range(n_GPUs))
                y = P.data_parallel(self.model, *x, self.gpu_ids)  #[[3,1,540,540],[3,1,540,540]]=[[sr_out],[gm_out]]
                if isinstance(y,tuple):
                    y = y[0]  # output tuple for spsr
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))

        else:
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if isinstance(y,tuple): 
                    y = y[0]  # output tuple for spsr
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        
        # y = self.merge_patch(y_chops, shave=shave)

        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        # print('y_chop', len(y_chops[0]), y[0].shape)
        # k
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if isinstance(y,list):
             y = y[0]

        return y

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y

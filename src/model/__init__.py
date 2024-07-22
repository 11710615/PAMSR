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
        self.tile = args.tile
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.args = args
        # self.n_GPUs = args.n_GPUs
        self.gpu_ids = args.gpu_ids
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
            elif self.tile:
                forward_function = self.forward_tile
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
                # print(pre_train,'*************')
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    def forward_chop(self, *args, shave=32, min_size=16000):
        scale = 1 if self.input_large else self.scale[self.idx_scale]
        # n_GPUs = min(self.n_GPUs, 4)
        n_GPUs = min(len(self.gpu_ids), 4)
        # height, width
        h, w = args[0].size()[-2:]

        top = slice(0, h//2 + shave)
        bottom = slice(h - h//2 - shave, h)
        left = slice(0, w//2 + shave)
        right = slice(w - w//2 - shave, w)
        # print(args[0].shape,'*')
        if len(args[0].shape) == 3:
            x_chops = [torch.cat([
                a[..., top, left].unsqueeze(1),
                a[..., top, right].unsqueeze(1),
                a[..., bottom, left].unsqueeze(1),
                a[..., bottom, right].unsqueeze(1)
            ]) for a in args]
        else:
            x_chops = [torch.cat([
                a[..., top, left],
                a[..., top, right],
                a[..., bottom, left],
                a[..., bottom, right]
            ]) for a in args]
        # print(len(args), x_chops[0].shape, len(x_chops))
        # for a in args:
        #     print(a.shape,'+++++')
        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                # y = P.data_parallel(self.model, *x, range(n_GPUs))
                y = P.data_parallel(self.model, *x, self.gpu_ids)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)
        if self.args.model != 'fd_unet':
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
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if isinstance(y,list): y = y[0]

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

    def forward_tile(self, img_lq, tile=64, window_size=8):
        # test the image tile by tile
        # [1,1,256,512]
        # if len(img_lq.shape)==5:
        #     img_lq = img_lq.permute(2,0,1,3,4).squeeze(0)
        # batch, burst, c, h, w = img_lq.size()
        batch = img_lq.shape[0]
        _, h, w = img_lq.shape[-3:]
        tile = min(tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = self.args.tile_overlap  # overlap for tile  56
        sf = self.scale[0]
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(batch, 1, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = self.model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)
                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

        return output


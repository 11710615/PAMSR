import imp
import os
import math
from decimal import Decimal
from statistics import mode

import utility

import torch
import torch.nn as nn
import torch.nn.utils as utils
from tqdm import tqdm
from model.common import Get_gradient, ema
import copy
import wandb

class Trainer_burst_ema():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.downsample_gt = args.downsample_gt

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = my_model
        self.loss = my_loss

        # ema model
        self.model_ema = copy.deepcopy(self.model)  # the tensor's status is also copied
        # self.model_ema.eval() # batchsize to 1?
        for p in self.model_ema.parameters():
            p.requires_grad = False

        self.optimizer = utility.make_optimizer(args, self.model)
        self.output_channels = args.output_channels
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    # def forward_downsample_gt(self, burst, model, idx_scale=0):
    #     h, _ = burst.shape[-2:]
    #     burst_odd = burst[..., range(0, h, 2),:]    
    #     burst_even = burst[..., range(1, h, 2),:]
    #     # burst_odd, burst_even = self.prepare(burst_odd, burst_even, hr)
    #     sr_burst_odd = model(burst_odd, idx_scale)
    #     sr_burst_even = model(burst_even, idx_scale)
    #     # combine
    #     # h_hr, _ = hr.shape[-2:]
    #     # out = torch.zeros_like(hr)
    #     # out[..., range(0, h_hr, 2), :] = sr_burst_odd
    #     # out[..., range(1, h_hr, 2), :] = sr_burst_even
    #     return sr_burst_odd, sr_burst_even

    def train(self):
        # os.environ["WANDB_API_KEY"] = 
        # os.environ["WANDB_MODE"] = "offline"
        # wandb.init(project='burst_sr_ema', name=self.args.save, entity='p3kkk', config=self.args)

        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()  # learning rate

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)

        for batch, data in enumerate(self.loader_train):

            burst, hr, meta_info, patch_cord = data  # burst:[1,15,1,64,64]; hr: [1,1,256,256]
            if self.output_channels < hr.shape[1]:
                hr = hr[:,0:1,:]
            # if hr.shape[1]==3:
            #     hr = hr[:,0:1,:]
            # print('burst', burst.shape)
            # k
            burst, hr = self.prepare(burst, hr)  # to device
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(burst, 0)

            self.model_ema = ema(self.model, self.model_ema, decay=0.999)
            sr_ema = self.model_ema(burst,0)
            if isinstance(sr, tuple):
                model_out = sr + sr_ema + patch_cord
            else:
                model_out = [sr, sr_ema, patch_cord]
            
            loss, loss_wandb = self.loss(model_out, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            if len(burst.shape) == 5:
                base_input = torch.cat([burst[i][0:1] for i in range(burst.shape[0])], axis=0)
            else:
                base_input = burst
            # print('base_input', base_input.shape)
            # wandb.log({'total_loss': loss, 'loss': loss_wandb, 'epoch':epoch,
            # 'burst_output': wandb.Image(sr),
            # 'gt': wandb.Image(hr),
            # 'base_input': wandb.Image(base_input)})

            timer_data.tic()
                
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results:
            self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):

                # d.dataset.set_scale(idx_scale)

                for burst, hr, meta_info, patch_cord in tqdm(d, ncols=80): # hr: [1,1,500,250]
                    filename = meta_info['burst_name']
                    if self.output_channels < hr.shape[1]:
                        hr = hr[:,0:1,:]
                    # if hr.shape[1]==3:
                    #     hr = hr[:,0:1,:]
                    burst, hr = self.prepare(burst, hr)

                    with torch.no_grad():
                        # if self.downsample_gt:
                        #     sr_odd, sr_even = self.forward_downsample_gt(burst, self.model, idx_scale)
                        #     sr = torch.zeros_like(hr)
                        #     h, _ = sr.shape[-2:]
                        #     sr[..., range(0, h, 2), :] = sr_odd
                        #     sr[..., range(1, h, 2), :] = sr_even
                        # else:
                        sr = self.model(burst, idx_scale)

                    sr, hr = self.center_crop(sr, hr)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        if len(burst.shape) == 5:
                            save_list.extend([burst[0][0:1], hr])
                        else:
                            save_list.extend([burst, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                    

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

    def center_crop(self, sr, hr):
        assert(sr.shape==hr.shape)
        h, w = sr.shape[-2:]
        h_crop = int(h/100)*100
        w_crop = int(w/100)*100

        ih = (h - h_crop) // 2
        iw = (w - w_crop) // 2

        sr_crop = sr[...,ih:ih+h_crop, iw:iw+w_crop]
        hr_crop = hr[...,ih:ih+h_crop, iw:iw+w_crop]
        return sr_crop, hr_crop  
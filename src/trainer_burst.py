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
import cv2
from model.common import Get_gradient

import wandb

class Trainer_burst():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.output_channels = args.output_channels
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):

        wandb.init(project='burst_sr', name=self.args.save, entity='p3kkk', config=self.args)

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

            burst, hr, _ = data  # burst:[1,15,1,64,64]; hr: [1,1,256,256]
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
            # print('****sr*****', torch.max(sr), torch.min(sr), torch.max(hr), torch.max(burst))
            # print('**', sr.shape, hr.shape, burst.shape)
            loss = self.loss(sr, hr)
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

            wandb.log({'total_loss': loss, 'epoch':epoch,
            'burst_output': wandb.Image(sr),
            'gt': wandb.Image(hr),
            'base_input': wandb.Image(burst[0][0:1])})

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

                for burst, hr, meta_info in tqdm(d, ncols=80):
                    filename = meta_info['burst_name']
                    if self.output_channels < hr.shape[1]:
                        hr = hr[:,0:1,:]
                    # if hr.shape[1]==3:
                    #     hr = hr[:,0:1,:]
                    # print('*****',burst.shape, hr.shape)
                    burst, hr = self.prepare(burst, hr)

                    with torch.no_grad():
                        sr = self.model(burst, idx_scale)

                    sr = utility.quantize(sr, self.args.rgb_range)
                    # print('***sr***', sr.shape, hr.shape, burst.shape)
                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([burst[0][0:1], hr])

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


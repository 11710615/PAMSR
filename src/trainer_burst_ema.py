import imp
import os
import math
from decimal import Decimal
from statistics import mode
import numpy as np

import utility

import torch
import torch.nn as nn
import torch.nn.utils as utils
from tqdm import tqdm
from model.common import Get_gradient, ema
import copy
import wandb
from loss.GradientLoss import CannyFilter

class Trainer_burst_ema():
    def __init__(self, args, loader, my_model, my_loss, ckp, fold):
        self.args = args
        self.scale = args.scale
        self.downsample_gt = args.downsample_gt

        self.fold = str(fold)
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = my_model
        self.loss = my_loss
        # self.cannyfilrer = CannyFilter(use_cuda=False)
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


    def train(self):
        # os.environ["WANDB_API_KEY"] = 
        # os.environ["WANDB_MODE"] = "offline"
        # wandb.init(project='proposed', name='fold_'+self.fold+'_'+self.args.save, entity='p3kkk', config=self.args)

        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()  # learning rate

        self.ckp.write_log(
            '[Fold {} Epoch {}]\tLearning rate: {:.2e}'.format(self.fold, epoch, Decimal(lr))
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
            # print(sr.shape, burst.shape, hr.shape, 'train')

            self.model_ema = ema(self.model, self.model_ema, decay=0.999)
            sr_ema = self.model_ema(burst,0)
            if isinstance(sr, tuple):
                model_out = sr + sr_ema + patch_cord
            else:
                model_out = [sr, sr_ema, patch_cord]
            # print(model_out[0].shape, hr.shape, burst.shape, '111')
            # s
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

            # hr_sobel, hr_canny = self.cannyfilrer(hr)
            # sr_sobel, sr_canny = self.cannyfilrer(sr)
            # sr_rec, hr_rec = self.rec_from_polar(model_out, hr)
            # print('base_input', base_input.shape)
            # wandb.log({'total_loss': loss, 'loss': loss_wandb, 'epoch':epoch,
            # 'burst_output': wandb.Image(sr),
            # 'gt': wandb.Image(hr),
            # 'hr_sobel': wandb.Image(hr_sobel),
            # 'sr_sobel': wandb.Image(sr_sobel),
            # 'hr_canny': wandb.Image(hr_canny),
            # 'sr_canny': wandb.Image(sr_canny)})

            timer_data.tic()
                
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nFold: {}Evaluation:'.format(self.fold))
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results:
            self.ckp.begin_background()
        eval_value = np.zeros([4])
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
                        sr = self.model(burst, idx_scale)
                    sr, hr = self.center_crop(sr, hr)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    # print('**',utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d))
                    # print('***111', utility.evaluation(sr.cpu().numpy().squeeze(0).squeeze(0), hr.cpu().numpy().squeeze(0).squeeze(0), self.args.rgb_range)[0])
                    # k
                    if self.args.test_only:
                        psnr_, ssim_ = utility.evaluation(sr.cpu().numpy().squeeze(0).squeeze(0), hr.cpu().numpy().squeeze(0).squeeze(0), self.args.rgb_range)
                        # sr_rec = self.rec_img_test(sr)
                        # hr_rec = self.rec_img_test(hr)
                        sr_rec = sr.cpu().numpy().squeeze(0).squeeze(0)
                        hr_rec = hr.cpu().numpy().squeeze(0).squeeze(0)
                        psnr_rec, ssim_rec = utility.evaluation(sr_rec, hr_rec, 255)
                        eval_value += np.array([psnr_, ssim_, psnr_rec, ssim_rec])

                    if self.args.save_gt:
                        if len(burst.shape) == 5:
                            save_list.extend([burst[0][0:1], hr])
                        else:
                            save_list.extend([burst, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                    

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                eval_value /= len(d)
                best = self.ckp.log.max(0)
                if self.args.test_only:
                    self.ckp.write_log(
                        '[{} x{} Fold {}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})\nSSIM: {:.3f}\t Rec_PSNR: {:.3f} Rec_SSIM: {:.3f}, *{:.3f}'.format(
                            d.dataset.name,
                            scale,
                            self.fold,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1,
                            eval_value[1],
                            eval_value[2],
                            eval_value[3],
                            eval_value[0]
                        )
                    )
                else:
                    self.ckp.write_log(
                        '[{} x{} Fold {}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.fold,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
        # print('**',self.ckp.log[:, idx_data, idx_scale].numpy())

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        # early stop when val_psnr stops increasing
        if epoch==self.args.epochs-1 or (epoch-30 > best[1][idx_data, idx_scale] + 1):
            self.ckp.fold_best.append(np.round(best[0][idx_data, idx_scale].numpy(),5))
            if self.args.data_train[0].find('bsr') >= 0:
                self.ckp.early_stop = False
            else:
                self.ckp.early_stop = True

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
        # print(sr.shape, hr.shape, '*****************')
        assert(sr.shape==hr.shape)
        h, w = sr.shape[-2:]
        # h_crop = int(h/100)*100
        # w_crop = int(w/100)*100
        h_crop = self.args.test_patch_size[0]*self.args.scale[0]
        w_crop = self.args.test_patch_size[1]*self.args.scale[0]
        # print(w_crop, h_crop,'****3333333')
        ih = (h - h_crop) // 2
        iw = (w - w_crop) // 2
        sr_crop = sr[...,ih:ih+h_crop, iw:iw+w_crop]
        hr_crop = hr[...,ih:ih+h_crop, iw:iw+w_crop]
        return sr_crop, hr_crop  
    
    def rec_img(self, img, patch_cord, h_idx, w_idx):
        # patch_cord = [h,w,h_size, w_size, start_row][b]
        b,c,*_ = img.shape
        imgReconstruction = torch.zeros([b,c,1024,1024]).type_as(img)  # [b,1,h,w]
        for i in range(b):
            h_ii = h_idx[patch_cord[0][i]:patch_cord[2][i], patch_cord[1][i]:patch_cord[3][i]].flatten()
            w_ii = w_idx[patch_cord[0][i]:patch_cord[2][i], patch_cord[1][i]:patch_cord[3][i]].flatten()
            # print('***',h_ii.shape, len(w_ii), img.shape,patch_cord[0][i],patch_cord[2][i], patch_cord[1][i],patch_cord[3][i])
            imgReconstruction[i:i+1,:,h_ii, w_ii] = img[i:i+1,:,:,:].flatten()
        return imgReconstruction

    def rec_from_polar(self, model_out, hr):
        rec_map = np.load('./loss/rec_map.npy', allow_pickle=True).item()
        h_idx = torch.from_numpy(rec_map['h_idx']).type(torch.long)
        w_idx = torch.from_numpy(rec_map['w_idx']).type(torch.long)

        patch_cord = model_out[-1]
        if isinstance(model_out, list):
            sr = model_out[0]
        sr_rec = self.rec_img(sr, patch_cord, h_idx, w_idx)
        hr_rec = self.rec_img(hr, patch_cord, h_idx, w_idx)
        return sr_rec, hr_rec

    def rec_img_test(self, img):
        # assert(img.shape[-2:] == (1000, 1000))
        # img = img_pad[...,12:1012, 12:1012]
        # map_idx = np.zeros_like(img)
        img = img.cpu().numpy().squeeze(0).squeeze(0)
        # img = img * 255
        amp = 1
        bScanSumOrigin = 1000
        bScanNumOrigin = 1000
        bScanSum, bScanNum = img.shape[-2:]  # [1000,1000]

        mindata = img.min()
        maxdata = img.max()
        
        imgReconstruction = np.zeros_like(img)

        row, column = img.shape[-2:]
        dR = bScanNumOrigin / bScanNum  # 1
        dAngle = np.pi / bScanSum
        dRReconstruction = dR / amp  # 1
        dAngleReconstruction = dAngle / amp
        offset = (bScanNum / 2) * amp  # 500
        
        for rowIndex in range(1,row+1):
            for columnIndex in range(1, column+1):
                r = (offset - columnIndex) * dRReconstruction
                angle = (rowIndex - 1) * dAngleReconstruction
                # angle = rowIndex * dAngleReconstruction
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                i = round(x + bScanNumOrigin/2) + 1
                j = round(y + bScanNumOrigin/2) + 1
                if j > bScanSumOrigin:
                    j = bScanSumOrigin

                if i > bScanNumOrigin:
                    i = bScanNumOrigin
                if i < 1:
                    i = 1
                if j < 1:
                    j = 1
                if imgReconstruction[...,j-1,i-1] < img[..., rowIndex-1, columnIndex-1]:
                    imgReconstruction[...,j-1,i-1] = img[...,rowIndex-1, columnIndex-1]
        imgReconstruction = (imgReconstruction - imgReconstruction.min()) / (imgReconstruction.max()-imgReconstruction.min()) * 255
        imgReconstruction = imgReconstruction.astype('uint8')  

        return imgReconstruction
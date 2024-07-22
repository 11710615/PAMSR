import torch

import utility
import data
import model
import loss
from option import args
# from trainer import Trainer
# from trainer_ema import Trainer_ema
# from trainer_burst import Trainer_burst
from trainer_burst_ema import Trainer_burst_ema
from torch.utils.data.dataloader import DataLoader
import warnings
from sklearn.model_selection import KFold
import os
import numpy as np

warnings.filterwarnings('ignore')

torch.manual_seed(args.seed)
# checkpoint = utility.checkpoint(args, fold)

data_list = os.listdir(os.path.join(args.dir_data, args.data_train[0]))
KF = KFold(n_splits=5, shuffle=True, random_state=55)
fold_best = []
fold_for_train = [1,2,3,4] # expert
def main():
    global model
    for fold, data_id in enumerate(KF.split(data_list)):
        print(fold)
        if not args.test_only and fold in fold_for_train:  # flag to select fold for training
            continue
        elif not args.test_only and not fold in fold_for_train:
            if int(args.scale[0])==2:
                args.pre_train = ''
            else:
                pre_train = '../experiment/'+ args.save.replace('_x{}'.format(int(args.scale[0])), '_x2') + '/{}'+'/model/model_best.pt'
                args.pre_train = pre_train.format(fold)
            print(args.pre_train)
        elif args.test_only and not fold in args.fold:  # flag to select fold for testing
            # print(not fold in args.fold, fold, args.fold)
            continue
        else:
            pre_train = '../experiment/'+ args.save + '/{}'+'/model/model_best.pt'
            args.pre_train = pre_train.format(fold)
            # args.pre_train = ''

        checkpoint = utility.checkpoint(args, fold)
        if checkpoint.ok:
            loader = data.Data(args, data_id)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None

            t = Trainer_burst_ema(args, loader, _model, _loss, checkpoint, fold)

            while not t.terminate():
                t.train()
                t.test()
                if checkpoint.early_stop is True:

                    fold_best.append(checkpoint.fold_best[0])
                    break
            # fold_best.append(checkpoint.fold_best[0])
            checkpoint.done()
        print('{} fold_best: {} dB'.format(fold, fold_best))
    print('ave_psnr: {:.4f} dB'.format(np.mean(fold_best)))
if __name__ == '__main__':
    main()

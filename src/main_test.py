import torch

import utility
import data
import model
import loss
from option import args
from trainer_test import Trainer_burst_ema
from torch.utils.data.dataloader import DataLoader
import warnings
from sklearn.model_selection import KFold
import os
import numpy as np

warnings.filterwarnings('ignore')

torch.manual_seed(args.seed)
# checkpoint = utility.checkpoint(args, fold)

if args.data_train[0] in ['real_lr', 'unreg_real_lr']:
    data_list = os.listdir(os.path.join(args.dir_data, args.data_train[0], 'test'))
else:
    data_list = os.listdir(os.path.join(args.dir_data, args.data_train[0]))

fold_best = []
data_split = True
def main():
    global model
    data_id = range(len(data_list))
    checkpoint = utility.checkpoint(args)
    if checkpoint.ok:
        loader = data.Data(args, data_id)
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer_burst_ema(args, loader, _model, _loss, checkpoint, fold=0)
        while not t.terminate():
            t.train()
            t.test()
            if checkpoint.early_stop is True:
                fold_best.append(checkpoint.fold_best[0])
                break
            # fold_best.append(checkpoint.fold_best[0])
        checkpoint.done()
    print('{} fold_best: {} dB'.format(0, fold_best))
if __name__ == '__main__':
    main()

import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from trainer_ema import Trainer_ema
from trainer_burst import Trainer_burst
from trainer_burst_ema import Trainer_burst_ema
from torch.utils.data.dataloader import DataLoader
import warnings

warnings.filterwarnings('ignore')

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            # if args.use_ema:
            #     t = Trainer_ema(args, loader, _model, _loss, checkpoint)
            # elif args.template in ['bipnet_x2', 'bipnet_swinir_x2']:
            #     t = Trainer_burst(args, loader, _model, _loss, checkpoint)
            # elif args.template.find('polar_bipnet_swinir')>=0:

            t = Trainer_burst_ema(args, loader, _model, _loss, checkpoint)
            # else:
            #     t = Trainer(args, loader, _model, _loss, checkpoint)

            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()

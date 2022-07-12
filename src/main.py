import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from trainer_ema import Trainer_ema
from trainer_burst import Trainer_burst
from data.burst import BurstSRDataset
from torch.utils.data.dataloader import DataLoader

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
            if args.use_ema:
                t = Trainer_ema(args, loader, _model, _loss, checkpoint)
            elif args.model in ['bipnet', 'swinir_burst']:
                # train_dataset = BurstSRDataset(args, split='train')
                # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                # test_dataset = BurstSRDataset(args, split='val')
                # test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
                # t = Trainer_burst(args, train_loader,test_loader, _model, _loss, checkpoint)
                t = Trainer_burst(args, loader, _model, _loss, checkpoint)
            else:
                t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()

from ast import arg
from importlib import import_module
from operator import mod
from tkinter.tix import IMMEDIATE
from cv2 import split
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        if not datasets[0].name in ['BurstSRDataset', 'BurstSRDataset_new']:
            self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args, data_id=None):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:  # 'BVMedV4'
                if d in ['burst']:
                    module_name = d
                    m = import_module('data.'+module_name.lower())
                    datasets.append(getattr(m, 'BurstSRDataset')(args, split='train'))
                elif d in ['burst_v1']:
                    module_name = d
                    m = import_module('data.'+module_name.lower())
                    datasets.append(getattr(m, 'BurstSRDataset')(args, split='train'))
                elif d in ['test_reg', 'test_unreg','test_reg_mid', 'test_unreg_mid','test_mid_reg']:
                    module_name = 'syn_lr'
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, 'BurstSRDataset')(args, data_id, split='train'))
                elif d in ['mid_filter']:
                    if args.model in ['fd_unet','unet']:
                        module_name = 'pam_rec'
                        m = import_module('data.pam_rec')
                        datasets.append(getattr(m, 'BurstSRDataset')(args, data_id, split='train'))
                    else:
                        module_name = d
                        m = import_module('data.' + module_name.lower())
                        datasets.append(getattr(m, 'BurstSRDataset')(args, data_id, split='train'))
                elif d in ['burst_bsr']:
                    module_name = d
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, 'BurstSRDataset')(args, data_id, split='train'))
                elif d in ['real_lr', 'unreg_real_lr']:
                    module_name = 'real_lr'
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, 'BurstSRDataset')(args, data_id, split='train'))
                elif d in ['burst_v3', 'reg_mid']:
                    if args.model in ['fd_unet','unet']:
                        module_name = 'pam_rec'
                        m = import_module('data.pam_rec')
                        datasets.append(getattr(m, 'BurstSRDataset')(args, data_id, split='train'))
                    else:
                        print('loading the dataset: {}'.format(d))
                        module_name = 'burst_v3'
                        m = import_module('data.' + module_name.lower())
                        datasets.append(getattr(m, 'BurstSRDataset')(args, data_id, split='train'))      
                elif d in ['lly_x1','lly_x2']:
                    module_name = 'lly_x1'
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, 'BurstSRDataset')(args, data_id, split='train'))          
                else:
                    module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, module_name)(args, name=d))
             
            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            elif d in ['burst']:
                module_name = d
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, 'BurstSRDataset')(args, split='val')
            elif d in ['burst_v1']:
                module_name = d
                m = import_module('data.' + module_name.lower())
                if args.downsample_gt:
                    testset = getattr(m, 'BurstSRDataset')(args, split='val_divided')
                else:
                    testset = getattr(m, 'BurstSRDataset')(args, split='val')
            elif d in ['test_reg', 'test_unreg','test_reg_mid', 'test_unreg_mid','test_mid_reg']:
                module_name = 'syn_lr'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, 'BurstSRDataset')(args, data_id, split='val')
            elif d in ['mid_filter']:
                if args.model in ['fd_unet','unet']:
                    module_name = 'pam_rec'
                    m = import_module('data.pam_rec')
                    testset = getattr(m,'BurstSRDataset')(args, data_id, split='val') 
                else:
                    module_name = d
                    m = import_module('data.' + module_name.lower())
                    testset = getattr(m, 'BurstSRDataset')(args, data_id, split='val')
            elif d in ['burst_bsr']:
                module_name = d
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, 'BurstSRDataset')(args, data_id, split='val')
            elif d in ['real_lr', 'unreg_real_lr']:
                module_name = 'real_lr'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, 'BurstSRDataset')(args, data_id, split='val')
            elif d in ['burst_v3', 'reg_mid']:
                if args.model in ['fd_unet','unet']:
                    module_name = 'pam_rec'
                    m = import_module('data.pam_rec')
                    testset = getattr(m,'BurstSRDataset')(args, data_id, split='val') 
                else:
                    print('loading the dataset: {}'.format(d))
                    module_name = 'burst_v3'
                    m = import_module('data.' + module_name.lower())
                    testset = getattr(m, 'BurstSRDataset')(args, data_id, split='val') 
            elif d in ['lly_x1','lly_x2']:
                module_name = 'lly_x1'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, 'BurstSRDataset')(args, data_id, split='val')
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)
            print('Loading {} done'.format(module_name))
            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
            
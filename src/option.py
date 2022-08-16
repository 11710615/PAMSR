import argparse

from yaml import parse
import template
import torch

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
# parser.add_argument('--n_GPUs', type=int, default=1,
#                     help='number of GPUs')
parser.add_argument('--gpu_ids', type=str, default='0',
                    help='gpu_ids, default=0')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../data/dataset',
                    help='dataset directory')
                    
parser.add_argument('--mask_dir', type=str, default='../data/dataset/mask',
                    help='mask dir')
parser.add_argument('--use_mask', action='store_true',
                    help='use mask at the model output')
parser.add_argument('--use_ema', action='store_true',
                    help='use ema')

parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='burst_v3',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='burst_v3',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-88/1-32',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=64,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--output_channels', type=int, default=1,
                    help='output_channels')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--tile', action='store_true', 
                    help='test the model with tile-like input')
parser.add_argument('--tile_overlap', type=int, default=4,
                    help='overlap for tiles, stride = tile-tile_overlap')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

parser.add_argument('--test_patch_size', type=int, default=500,
                    help='the patch_size in test phase in case of out of memory')
parser.add_argument('--burst_size_max',type=int, default=25,
                    help='max size of burst')
parser.add_argument('--burst_size',type=int, default=1,
                    help='size of burst')
parser.add_argument('--downsample_gt', action='store_true',
                    help='whether downsample gt to eliminate malposition')
# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--entrop_select', action='store_true',
                    help='whether using entrop to select patch for training')
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,  #1000
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')
parser.add_argument('--use ema', action='store_true',
                    help='whether use ema')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')


# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')  # model- __init__.py
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gm', action='store_true',
                    help='save output gm')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.test_only:
    args.tile_overlap =64 -  32 // int(args.scale[0])  # scale=4: 56, scale=2: 48
    # print('**', args.tile_overlap)
# set gpu ids
str_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        args.gpu_ids.append(id)
if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False


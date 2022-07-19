from enum import Flag
from pickle import TRUE


def set_template(args):
    # Set the templates here

#########################################################################################
######################################### single ########################################
#########################################################################################
    if args.template.find('polar_swinir_GradientL1_x2') >= 0:
        args.model = 'Swinir'
        args.patch_size = 64
        args.num_features = 180
        args.burst_size = 1
        args.chop = True
        args.scale = '2'
        args.data_train = 'burst_v1'
        args.data_test = 'burst_v1'
        args.rgb_range = 1
        args.test_patch_size = 250
        args.loss = '200*Gradient_L1+0.5*VGG54'  # 10L1? 40:bury
        args.save = 'polar_swinir_GradientL1_x2_burst-' + args.scale + '_' + args.loss
        args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # args.pre_train = '../experiment/swinir_ema_x2_' + args.loss + '/model/model_best.pt'


#########################################################################################
######################################### burst #########################################
#########################################################################################
    if args.template.find('polar_bipnet_swinir_GradientL1_x2') >= 0:
        args.model = 'swinir_burst'
        args.patch_size = 64
        args.num_features = 180
        args.burst_size = 5
        args.scale = '2'
        args.loss = '200*Gradient_L1+0.5*VGG54'  # [10,4,0.1]->[100,1,0.1]
        args.output_channels = 1
        args.chop = True
        args.save = 'polar_bipnet_swinir_GradientL1_x2_burst-' + str(args.burst_size)+ '_' + args.loss
        args.data_train = 'burst_v1'
        args.data_test = 'burst_v1'
        args.batch_size = (len(args.gpu_ids) + 1) // 2
        args.rgb_range = 1
        args.test_patch_size = 250
        args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # args.pre_train = '../experiment/bipnet_swinir_GradientL1_x2_burst-'+ str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'


    if args.template.find('polar_bipnet_swinir_RL1_x2') >= 0:
        args.model = 'swinir_burst'
        args.patch_size = 64
        args.num_features = 180
        args.burst_size = 5
        args.scale = '2'
        args.loss = '1*RL1'  # [10,4,0.1]->[100,1,0.1]
        args.output_channels = 1
        args.chop = True
        args.save = 'polar_bipnet_swinir_RL1_x2_burst-' + str(args.burst_size)+ '_' + args.loss
        args.data_train = 'burst_v1'
        args.data_test = 'burst_v1'
        args.batch_size = (len(args.gpu_ids) + 1) // 2
        args.rgb_range = 1
        args.test_patch_size = 250
        args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # args.pre_train = '../experiment/bipnet_swinir_GradientL1_x2_burst-'+ str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'
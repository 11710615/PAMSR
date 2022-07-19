from enum import Flag
from pickle import TRUE


def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('DDBPN') >= 0:
        args.model = 'DDBPN'
        args.patch_size = 128
        args.scale = '4'

        args.data_test = 'Set5'

        args.batch_size = 20
        args.epochs = 1000
        args.decay = '500'
        args.gamma = 0.1
        args.weight_decay = 1e-4

        args.loss = '1*MSE'

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.decay = '150'

    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = True

    if args.template.find('VDSR') >= 0:
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 41
        args.lr = 1e-1

    if args.template.find('HAN') >= 0:
        args.model = 'HAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = True

    if args.template.find('edsr_1') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 16
        args.n_feats = 64
        args.chop = True

    if args.template.find('swinir') >= 0:
        args.model = 'SwinIR'
        args.patch_size = 64
        args.chop = True

    if args.template.find('swinir_ema') >= 0:
        args.model = 'SwinIR'
        args.patch_size = 64
        args.chop = True
        args.use_ema = True
        args.scale = '2'
        args.loss = '200*WL1+4*VGG54+0.1*GAN'
        args.save = 'swinir_ema_x' + args.scale + '_' + args.loss
        args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # args.pre_train = '../experiment/swinir_ema_x2_200*WL1+4*VGG54+0.1*GAN_2/model/model_best.pt'

    if args.template.find('swinir_ema_x4') >= 0:
        args.scale = '4'
        args.model = 'SwinIR'
        args.patch_size = 64
        args.chop = True
        args.use_ema = True
        args.save = 'swinir_ema_x4' + '_' + args.loss
        args.loss = '200*WL1+4*VGG54+0.1*GAN'
        args.pre_train = '../experiment/swinir_ema_x2_' + args.loss + '/model/model_best.pt'

    if args.template.find('swinir_inv_ema') >= 0:
        args.model = 'SwinIR'
        args.patch_size = 64
        args.chop = True
        args.use_ema = True
        args.scale = '2'
        args.loss = '2*InvWL1+4*VGG54+0.1*GAN' # 50? 10? 2.5? 2?
        args.save = 'swinir_inv_ema_x' + args.scale + '_' + args.loss
        args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # args.pre_train = '../experiment/swinir_inv_ema_x2_'+ args.loss +'/model/model_best.pt'

    if args.template.find('swinir_inv_ema_x4') >= 0:
        args.model = 'SwinIR'
        args.patch_size = 64
        args.chop = True
        args.use_ema = True
        args.scale = '4'
        args.loss = '2*InvWL1+4*VGG54+0.1*GAN' # 50? 10? 2.5? 2?
        args.save = 'swinir_inv_ema_x' + args.scale + '_' + args.loss
        # args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # args.pre_train = '../experiment/swinir_inv_ema_x2_'+ args.loss +'/model/model_best.pt'
        args.pre_train = '../experiment/swinir_inv_ema_x4_'+ args.loss +'/model/model_best.pt'

    if args.template.find('swinir_ema_topology') >= 0:
        args.model = 'SwinIR'
        args.patch_size = 64
        args.chop = True
        args.use_ema = True
        args.scale = '2'
        args.loss = '1*topology+5*VGG54+0.1*GAN'
        args.save = 'swinir_ema_topology_x' + args.scale + '_' + args.loss
        # args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        args.pre_train = '../experiment/swinir_ema_topology_x2_' + args.loss + '/model/model_best.pt'

    if args.template.find('swinir_ema_GradientWL1') >= 0:
        args.model = 'SwinIR'
        args.patch_size = 64
        args.chop = True
        args.use_ema = True
        args.scale = '2'
        args.loss = '40*Gradient_WL1+4*VGG54+0.1*GAN'  # 40L1?
        args.save = 'swinir_ema_x' + args.scale + '_' + args.loss
        # args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        args.pre_train = '../experiment/swinir_ema_x2_' + args.loss + '/model/model_best.pt'

    if args.template.find('swinir_ema_GradientWL1_x4') >= 0:
        args.model = 'SwinIR'
        args.patch_size = 64
        args.chop = True
        args.use_ema = True
        args.scale = '4'
        args.loss = '40*Gradient_WL1+4*VGG54+0.1*GAN'
        args.save = 'swinir_ema_x' + args.scale + '_' + args.loss
        args.pre_train = '../experiment/swinir_ema_x4_' + args.loss + '/model/model_best.pt'
        # args.pre_train = '../experiment/swinir_ema_x2_' + args.loss + '/model/model_best.pt'
    
    if args.template.find('swinir_ema_GradientL1') >= 0:
        args.model = 'SwinIR'
        args.patch_size = 64
        args.chop = True
        args.use_ema = True
        args.scale = '2'
        args.loss = '10*Gradient_L1+4*VGG54+0.1*GAN'  # 10L1? 40:bury
        args.save = 'swinir_ema_x' + args.scale + '_' + args.loss
        # args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        args.pre_train = '../experiment/swinir_ema_x2_' + args.loss + '/model/model_best.pt'

    if args.template.find('swinir_ema_GradientL1_x4') >= 0:
        args.model = 'SwinIR'
        args.patch_size = 64
        args.chop = True
        args.use_ema = True
        args.scale = '4'
        args.loss = '10*Gradient_L1+4*VGG54+0.1*GAN'
        args.save = 'swinir_ema_x' + args.scale + '_' + args.loss
        args.pre_train = '../experiment/swinir_ema_x4_' + args.loss + '/model/model_best.pt'
        # args.pre_train = '../experiment/swinir_ema_x2_' + args.loss + '/model/model_best.pt'

    if args.template.find('swinir_real') >= 0:
        args.model = 'SwinIR'
        args.patch_size = 64
        args.chop = True

    if args.template.find('swinir_ca') >= 0:
        args.model = 'SwinIR_CA'
        args.patch_size = 64
        args.chop = True

    if args.template.find('swinir_sigblock') >= 0:
        args.model = 'swinir_sigblock'
        args.patch_size = 64
        args.chop = True

    if args.template.find('swinir_sp') >= 0:
        args.model = 'swinir_sp'
        args.patch_size = 64
        args.chop = False
        args.output_channels = 1
        args.save = 'swinir_sp_'
        args.loss = '1*RL1+4*VGG54+0.5*GAN+1*L1GM+0.5*ganGM+1*L1RG'

    if args.template.find('swinir_hf_x4') >= 0:
        args.model = 'swinir_sp'
        args.patch_size = 64
        # args.chop = False
        args.output_channels = 1
        args.scale = '4'
        args.save = 'swinir_hf_x4'
        # args.reset = True

    # if args.template.find('swinir_sp_div2k') >= 0:
    #     args.model = 'swinir_sp'
    #     args.patch = 64
    #     args.chop = True
    #     args.data_train = 'DIV2K_train'
    #     args.data_test = 'DIV2K_valid'
    #     args.data_range = '1-800/1-100'
    #     args.output_channels = 3
    #     args.batch_size = 3

    if args.template.find('swinir_sp_ema_x2') >= 0:
        args.model = 'swinir_sp'
        args.patch_size = 64
        args.chop = False  # multi output does not use chop
        args.output_channels = 1
        args.use_ema = True
        # args.loss = '200*WL1+4*VGG54+0.1*GAN'
        # args.loss = '1*RL1+4*VGG54+0.5*GAN+1*L1GM+0.5*ganGM+1*L1RG'
        args.loss = '100*WL1+2*VGG54+0.1*GAN+0.5*L1GM+0.1*ganGM+0.5*L1RG'
        args.save = 'swinir_sp_ema_x2_' + args.loss
        # args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        args.pre_train = '../experiment/swinir_sp_ema_x2_'+ args.loss +'/model/model_best.pt'
        args.scale = '2'

    if args.template.find('swinir_sp_ema_x4') >= 0:
        args.model = 'swinir_sp'
        args.patch_size = 64
        args.chop = False
        args.output_channels = 1
        args.use_ema = True
        # args.loss = '200*WL1+4*VGG54+0.1*GAN'
        # args.loss = '1*RL1+4*VGG54+0.5*GAN+1*L1GM+0.5*ganGM+1*L1RG'
        args.loss = '100*WL1+2*VGG54+0.1*GAN+0.5*L1GM+0.1*ganGM+0.5*L1RG'
        args.save = 'swinir_sp_ema_x4_' + args.loss
        args.pre_train = '../experiment/swinir_sp_ema_x4_'+ args.loss+'/model/model_best.pt'
        # args.pre_train = '../experiment/swinir_sp_ema_x2_'+ args.loss+'/model/model_best.pt'
        args.scale = '4'

    if args.template.find('bipnet_x2') >= 0:
        args.model = 'bipnet'
        args.patch_size = 64
        args.num_features = 64
        args.burst_size = 5
        args.scale = '2'
        args.loss = '10*L1+1*VGG54'
        args.output_channels = 1
        args.chop = True
        args.save = 'bipnet_x2_burst-' + str(args.burst_size)+ '_' + args.loss
        args.data_train = 'burst'
        args.data_test = 'burst'
        args.test_patch_size = 500
        args.batch_size = 1
        args.rgb_range = 1
        # args.pre_train = '../experiment/bipnet_x2_burst-'+ str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'


    if args.template.find('bipnet_swinir_x2') >= 0:
        args.model = 'swinir_burst'
        args.patch_size = 64
        args.num_features = 180
        args.burst_size = 5
        args.scale = '2'
        args.loss = '10*L1+1*VGG54+0.1*GAN'
        args.output_channels = 1
        args.chop = True
        args.save = 'bipnet_swinir_x2_burst-' + str(args.burst_size)+ '_' + args.loss
        args.data_train = 'burst'
        args.data_test = 'burst'
        args.batch_size = 1
        args.test_patch_size = 500
        args.rgb_range = 1
        # args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        args.pre_train = '../experiment/bipnet_swinir_x2_burst-'+ str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'


    if args.template.find('bipnet_swinir_GradientL1_x2') >= 0:
        args.model = 'swinir_burst'
        args.patch_size = 64
        args.num_features = 180
        args.burst_size = 5
        args.scale = '2'
        args.loss = '200*Gradient_L1+0.5*VGG54+0.05*GAN'  # [10,4,0.1]->[100,1,0.1]
        args.output_channels = 1
        args.chop = True
        args.save = 'bipnet_swinir_GradientL1_x2_burst-' + str(args.burst_size)+ '_' + args.loss
        args.data_train = 'burst'
        args.data_test = 'burst'
        args.batch_size = (len(args.gpu_ids) + 1) // 2
        args.test_patch_size = 500
        args.rgb_range = 1
        # args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        args.pre_train = '../experiment/bipnet_swinir_GradientL1_x2_burst-'+ str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'


    if args.template.find('bipnet_swinir_GradientL1_x4') >= 0:
        args.model = 'swinir_burst'
        args.patch_size = 64
        args.num_features = 180
        args.burst_size = 5
        args.scale = '4'
        args.loss = '200*Gradient_L1+0.5*VGG54+0.05*GAN'  # [10,4,0.1]->[100,1,0.1]
        args.output_channels = 1
        args.chop = True
        args.save = 'bipnet_swinir_GradientL1_x2_burst-' + str(args.burst_size)+ '_' + args.loss
        args.data_train = 'burst'
        args.data_test = 'burst'
        args.batch_size = (len(args.gpu_ids) + 1) // 2
        args.test_patch_size = 250
        args.rgb_range = 1
        # args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        args.pre_train = '../experiment/bipnet_swinir_GradientL1_x2_burst-'+ str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'

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
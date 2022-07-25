from enum import Flag
from pickle import TRUE


def set_template(args):
    # Set the templates here

#########################################################################################
######################################### single ########################################
#########################################################################################
    if args.template.find('polar_swinir_GradientL1_x2') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.chop = True
        args.scale = '2'
        args.data_train = 'burst_v1'
        args.data_test = 'burst_v1'
        args.rgb_range = 1
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        if args.test_only:
            args.test_patch_size = (512 // down_scale, 512)
        else:
            args.test_patch_size = (256 // down_scale, 256)
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # 10L1? 40:bury
        args.loss = '1*Gradient_L1'
        args.save = str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_x2_burst-' +str(args.burst_size)+ '_' + args.loss
        args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_x2_burst-' + str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'
        args.no_augment = True

    if args.template.find('polar_swinir_GradientL1_x4') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.chop = True
        args.scale = '4'
        args.data_train = 'burst_v1'
        args.data_test = 'burst_v1'
        args.rgb_range = 1
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        if args.test_only:
            args.test_patch_size = (256 // down_scale, 256)
        else:
            args.test_patch_size = (128 // down_scale, 128)
        args.loss = '1*Gradient_L1'  # 10L1? 40:bury
        args.save = str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_x4_burst-' +str(args.burst_size)+ '_' + args.loss
        args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_x2_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_x4_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True

####################################################################################################
    if args.template.find('EDSR_paper_x2') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.patch_size = (64, 64)
        args.num_features = 180

        args.burst_size = 1
        args.chop = True
        args.scale = '2'
        args.data_train = 'burst_v1'
        args.data_test = 'burst_v1'
        args.rgb_range = 1
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        if args.test_only:
            args.test_patch_size = (512 // down_scale, 512)
        else:
            args.test_patch_size = (256 // down_scale, 256)
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # 10L1? 40:bury
        args.loss = '1*RL1'
        args.save = str(args.downsample_gt)+'_'+'EDSR_paper_x2_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss
        args.pre_train = '../experiment1/pretrain_model/edsr_x2-0edfb8a3.pt'
        args.no_augment = True
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'EDSR_paper_x2_NoAugu_burst-'+ str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'

    if args.template.find('EDSR_paper_x4') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.patch_size = (64, 64)
        args.num_features = 180

        args.burst_size = 1
        args.chop = True
        args.scale = '4'
        args.data_train = 'burst_v1'
        args.data_test = 'burst_v1'
        args.rgb_range = 1
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        if args.test_only:
            args.test_patch_size = (256 // down_scale, 256)
        else:
            args.test_patch_size = (128 // down_scale, 128)
        args.loss = '1*RL1'  
        args.save = str(args.downsample_gt)+'_'+'EDSR_paper_x4_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss
        args.pre_train = '../experiment/' + str(args.downsample_gt)+'_'+'EDSR_paper_x2_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True
        # args.pre_train = '../experiment/' + str(args.downsample_gt)+'_'+'EDSR_paper_x4_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'
########################################## swinir_L1 #################################################
    if args.template.find('polar_swinir_L1_x2') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.chop = True
        args.scale = '2'
        args.data_train = 'burst_v1'
        args.data_test = 'burst_v1'
        args.rgb_range = 1
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        if args.test_only:
            args.test_patch_size = (512 // down_scale, 512)
        else:
            args.test_patch_size = (256 // down_scale, 256)
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # 10L1? 40:bury
        args.loss = '1*RL1'
        args.save = str(args.downsample_gt)+'_'+'polar_swinir_L1_x2_burst-' +str(args.burst_size)+ '_' + args.loss
        args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_x2_burst-' + str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'
        args.no_augment = True

    if args.template.find('polar_swinir_L1_x4') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.chop = True
        args.scale = '4'
        args.data_train = 'burst_v1'
        args.data_test = 'burst_v1'
        args.rgb_range = 1
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        if args.test_only:
            args.test_patch_size = (256 // down_scale, 256)
        else:
            args.test_patch_size = (128 // down_scale, 128)
        args.loss = '1*RL1'  # 10L1? 40:bury
        args.save = str(args.downsample_gt)+'_'+'polar_swinir_L1_x4_burst-' +str(args.burst_size)+ '_' + args.loss
        args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_x2_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_x4_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True


#########################################################################################
######################################### burst #########################################
#########################################################################################
    if args.template.find('polar_bipnet_swinir_GradientL1_x2') >= 0:
        args.model = 'swinir_burst'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 5
        args.scale = '2'
        args.loss = '200*Gradient_L1+0.5*VGG54'  # [10,4,0.1]->[100,1,0.1]
        args.output_channels = 1
        args.chop = True
        args.data_train = 'burst_v1'
        args.data_test = 'burst_v1'
        args.batch_size = (len(args.gpu_ids) + 1) // 2
        args.rgb_range = 1
        # args.downsample_gt = True
        args.save = str(args.downsample_gt)+'_'+'polar_bipnet_swinir_GradientL1_x2_burst-' + str(args.burst_size)+ '_' + args.loss
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        if args.test_only:
            args.test_patch_size = (512 // down_scale, 512)
        else:
            args.test_patch_size = (256 // down_scale, 256)
        # args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_bipnet_swinir_GradientL1_x2_burst-'+ str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True

    if args.template.find('polar_bipnet_swinir_GradientL1_x4') >= 0:
        args.model = 'swinir_burst'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 5
        args.scale = '4'
        args.loss = '200*Gradient_L1+0.5*VGG54'  # [10,4,0.1]->[100,1,0.1]
        args.output_channels = 1
        args.chop = True
        args.data_train = 'burst_v1'
        args.data_test = 'burst_v1'
        args.batch_size = (len(args.gpu_ids) + 1) // 2
        args.rgb_range = 1
        # args.downsample_gt = True
        args.save = str(args.downsample_gt)+'_'+'polar_bipnet_swinir_GradientL1_x4_burst-' + str(args.burst_size)+ '_' + args.loss
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        if args.test_only:
            args.test_patch_size = (256 // down_scale, 256)
        else:
            args.test_patch_size = (128 // down_scale, 128)
        args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_bipnet_swinir_GradientL1_x2_burst-' + str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_bipnet_swinir_GradientL1_x4_burst-' + str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True
###################################################################################################

    if args.template.find('polar_bipnet_swinir_RL1_x2') >= 0:
        args.model = 'swinir_burst'
        args.patch_size = (64, 64)
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

        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        if args.test_only:
            args.test_patch_size = (512 // down_scale, 512)
        else:
            args.test_patch_size = (256 // down_scale, 256)

        # args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        args.pre_train = '../experiment/polar_bipnet_swinir_RL1_x2_burst-'+ str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True
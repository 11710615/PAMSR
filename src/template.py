from enum import Flag
from pickle import TRUE

def set_template(args):
    # Set the templates here

######################################### no_pre_train##################################
    if args.template.find('polar_swinir_L1_nopretrain_x4') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '4'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (256 // down_scale, 256)
        # if args.test_only:
        #     args.test_patch_size = (512 // down_scale, 512)
        # else:
        #     args.test_patch_size = (256 // down_scale, 256)
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # 10L1? 40:bury
        # args.loss = '1*RL1'
        args.save = str(args.downsample_gt)+'_'+'polar_swinir_L1_nopretrain_downgt_x4_burst-' +str(args.burst_size)+ '_' + args.loss
        # args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_nopretrain_downgt_x2_burst-' + str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'
        args.no_augment = True

    if args.template.find('EDSR_paper_nopretrain_downgt_x2') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '2'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (512 // down_scale, 512)
        # if args.test_only:
        #     args.test_patch_size = (512 // down_scale, 512)
        # else:
        #     args.test_patch_size = (256 // down_scale, 256)
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # 10L1? 40:bury
        # args.loss = '1*RL1'
        args.save = str(args.downsample_gt)+'_'+'EDSR_paper_nopretrain_downgt_x2_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss
        # args.pre_train = '../experiment1/pretrain_model/edsr_x2-0edfb8a3.pt'
        args.no_augment = True
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'EDSR_paper_x2_NoAugu_burst-'+ str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'


#########################################################################################
######################################### single ########################################
#########################################################################################
    if args.template.find('polar_swinir_GradientL1_x2') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '2'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (512 // down_scale, 512)
        # if args.test_only:
        #     args.test_patch_size = (512 // down_scale, 512)
        # else:
        #     args.test_patch_size = (256 // down_scale, 256)
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # 10L1? 40:bury
        # args.loss = '1*Gradient_L1'
        args.save = str(args.downsample_gt)+'_'+'v2_polar_swinir_GradientL1_x2_burst-' +str(args.burst_size)+ '_' + args.loss
        args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'v2_polar_swinir_GradientL1_x2_burst-' + str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'
        args.no_augment = True

    if args.template.find('polar_swinir_GradientL1_x4') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '4'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (256 // down_scale, 256)
        # if args.test_only:
        #     args.test_patch_size = (256 // down_scale, 256)
        # else:
        #     args.test_patch_size = (128 // down_scale, 128)
        # args.loss = '1*Gradient_L1'  # 10L1? 40:bury
        args.save = str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_x4_burst-' +str(args.burst_size)+ '_' + args.loss
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_x2_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_x4_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
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
        # args.tile = True
        args.tile = True
        args.scale = '2'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (512 // down_scale, 512)
        # if args.test_only:
        #     args.test_patch_size = (512 // down_scale, 512)
        # else:
        #     args.test_patch_size = (256 // down_scale, 256)
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # 10L1? 40:bury
        # args.loss = '1*RL1'
        args.save = str(args.downsample_gt)+'_'+'V2_EDSR_paper_x2_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss
        args.pre_train = '../experiment1/pretrain_model/edsr_x2-0edfb8a3.pt'
        args.no_augment = True
        # args.pre_train = '../experiment3/'+str(args.downsample_gt)+'_'+'V1_EDSR_paper_x2_NoAugu_burst-'+ str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'

    if args.template.find('EDSR_paper_x4') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '4'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (256 // down_scale, 256)
        # if args.test_only:
        #     args.test_patch_size = (256 // down_scale, 256)
        # else:
        #     args.test_patch_size = (128 // down_scale, 128)
        # args.loss = '1*RL1'  
        # args.save = str(args.downsample_gt)+'_'+'V1_EDSR_paper_x4_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss
        args.save = 'test111'
        # args.pre_train = '../experiment/' + str(args.downsample_gt)+'_'+'V1_EDSR_paper_x2_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True
        # args.pre_train = '../experiment/' + str(args.downsample_gt)+'_'+'V1_EDSR_paper_x4_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'

    if args.template.find('EDSR_paper_x8') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '8'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (128 // down_scale, 128)
        # if args.test_only:
        #     args.test_patch_size = (128 // down_scale, 128)
        # else:
        #     args.test_patch_size = (64 // down_scale, 64)
        # args.loss = '1*RL1'  
        args.save = str(args.downsample_gt)+'_'+'EDSR_paper_x8_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss
        # args.pre_train = '../experiment/' + str(args.downsample_gt)+'_'+'EDSR_paper_x2_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True
        args.pre_train = '../experiment/' + str(args.downsample_gt)+'_'+'EDSR_paper_x8_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'
########################################## swinir_L1 #################################################
    if args.template.find('polar_swinir_L1_x2') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '2'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (512 // down_scale, 512)
        # if args.test_only:
        #     args.test_patch_size = (512 // down_scale, 512)
        # else:
        #     args.test_patch_size = (256 // down_scale, 256)
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # 10L1? 40:bury
        # args.loss = '1*RL1'
        args.save = str(args.downsample_gt)+'_'+'v2_polar_swinir_L1_x2_burst-' +str(args.burst_size)+ '_' + args.loss
        args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # args.pre_train = '../experiment3/'+str(args.downsample_gt)+'_'+'v2_polar_swinir_L1_x2_burst-' + str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'
        args.no_augment = True

    if args.template.find('polar_swinir_L1_x4') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '4'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (256 // down_scale, 256)
        # if args.test_only:
        #     args.test_patch_size = (256 // down_scale, 256)
        # else:
        #     args.test_patch_size = (128 // down_scale, 128)
        # args.loss = '1*RL1'  # 10L1? 40:bury
        args.save = str(args.downsample_gt)+'_'+'polar_swinir_L1_x4_burst-' +str(args.burst_size)+ '_' + args.loss
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_x2_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_x4_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True

########################################################################################
################################### odd-even divide#####################################
########################################################################################
########################################## swinir_L1 #################################################
    if args.template.find('polar_swinir_L1_downgt_x2') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '2'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (512 // down_scale, 512)
        # if args.test_only:
        #     args.test_patch_size = (512 // down_scale, 512)
        # else:
        #     args.test_patch_size = (256 // down_scale, 256)
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # 10L1? 40:bury
        # args.loss = '1*RL1'
        args.save = str(args.downsample_gt)+'_'+'noact_polar_swinir_L1_downgt_x2_burst-' +str(args.burst_size)+ '_' + args.loss
        # args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        args.pre_train = '../experiment3/'+str(args.downsample_gt)+'_'+'noact_polar_swinir_L1_downgt_x2_burst-' + str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'
        args.no_augment = True
        
    if args.template.find('polar_swinir_L1_downgt_x4') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '4'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (256 // down_scale, 256)
        # if args.test_only:
        #     args.test_patch_size = (256 // down_scale, 256)
        # else:
        #     args.test_patch_size = (128 // down_scale, 128)
        # args.loss = '1*RL1'  # 10L1? 40:bury
        args.save = str(args.downsample_gt)+'_'+'polar_swinir_L1_downgt_x4_burst-' +str(args.burst_size)+ '_' + args.loss
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_downgt_x2_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_downgt_x4_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True

    if args.template.find('polar_swinir_L1_downgt_x8') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '8'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (128 // down_scale, 128)

        args.save = str(args.downsample_gt)+'_'+'polar_swinir_L1_downgt_x8_burst-' +str(args.burst_size)+ '_' + args.loss
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_downgt_x4_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_downgt_x8_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True


################################### swinir_GradientL1 ##################################
    if args.template.find('polar_swinir_GradientL1_downgt_x2') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        # args.tile = True
        args.tile = True
        args.scale = '2'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        args.downsample_gt = False  # control traning and val set
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (512 // down_scale, 512)
        # if args.test_only:
        #     args.test_patch_size = (512 // down_scale, 512)
        # else:
        #     args.test_patch_size = (256 // down_scale, 256)
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # 10L1? 40:bury
        # args.loss = '1*Gradient_L1'
        args.save = str(args.downsample_gt)+'_noact_polar_swinir_GradientL1_downgt_x2_burst-' +str(args.burst_size)+ '_' + args.loss
        # args.save = 'test111'
        # args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        args.pre_train = '../experiment3/'+str(args.downsample_gt)+'_'+'noact_polar_swinir_GradientL1_downgt_x2_burst-' + str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'
        args.no_augment = True

    if args.template.find('polar_swinir_GradientL1_downgt_x4') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '4'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (256 // down_scale, 256)
        args.save = str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_downgt_x4_burst-' +str(args.burst_size)+ '_' + args.loss
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_downgt_x4_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_epoch_5.pt'
        args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_downgt_x4_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True

    if args.template.find('polar_swinir_GradientL1_downgt_x8') >= 0:
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '8'
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (128 // down_scale, 128)

        args.save = str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_downgt_x8_burst-' +str(args.burst_size)+ '_' + args.loss
        # args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_downgt_x8_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_epoch_1.pt'
        args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_downgt_x8_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True
########################################################################################
################################### rec_loss ###########################################
########################################################################################








#########################################################################################
######################################### burst #########################################
#########################################################################################
    if args.template.find('polar_bipnet_swinir_GradientL1_x2') >= 0:
        args.model = 'swinir_burst'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 5
        args.scale = '2'
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # [10,4,0.1]->[100,1,0.1]
        args.output_channels = 1
        args.tile = True
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.batch_size = (len(args.gpu_ids) + 1) // 2
        args.rgb_range = 1
        # args.downsample_gt = False
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
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # [10,4,0.1]->[100,1,0.1]
        args.output_channels = 1
        args.tile = True
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
        args.batch_size = (len(args.gpu_ids) + 1) // 2
        args.rgb_range = 1
        # args.downsample_gt = False
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
        # args.loss = '1*RL1'  # [10,4,0.1]->[100,1,0.1]
        args.output_channels = 1
        args.tile = True
        args.save = 'polar_bipnet_swinir_RL1_x2_burst-' + str(args.burst_size)+ '_' + args.loss
        args.data_train = 'burst_v2'
        args.data_test = 'burst_v2'
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
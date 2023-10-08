def set_template(args):
    # Set the templates here
        
    if args.template == 'swinir_burst1':
        args.model = 'swinir_burst_sifa'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 3
        args.n_colors = args.burst_size
        args.tile = True
        # args.scale = '2'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'

        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.loss = '1*RL1'
        args.save =  flag + args.template + '_x' + str(args.scale) + '_' + str(args.burst_size) + '_' + args.model + '_' + args.data_train + '_' + args.loss
        args.no_augment = True
        # #args.pre_train = '../experiment/'+ args.save + '/'+str(args.fold)+'/model/model_latest.pt'
        #args.pre_train = ''
        print(args.pre_train,'12121')



    if args.template == 'test_only':
        # args.model = 'EDSR'#'Swinir'
        # args.n_resblocks = 32
        # args.n_feats = 256
        # args.res_scale = 0.1
        
        # args.model = 'HAN'#'Swinir'
        # args.n_resblocks = 20
        # args.n_feats = 128
        # args.res_scale = 1
        # args.rgb_range = 1
        # args.n_colors = 3
        # args.output_channels = 1
        
        # args.model = 'RCAN'#'Swinir'
        # args.n_resblocks = 20
        # args.n_feats = 128
        # args.res_scale = 1
        
        args.model = 'RDN'#'Swinir'
        args.n_resblocks = 16
        args.n_feats = 256
        args.res_scale = 1
        args.GO = 64
        
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '2'
        args.data_train = 'unreg_real_lr'
        args.data_test = 'unreg_real_lr'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.save = args.template+'_'+args.model + '_' + args.data_train + '_nomid'


    if args.template == 'FD_Unet_x2':
        args.model = 'fd_unet'
        args.patch_size = (64, 64)
        args.num_features = 64
        args.test_patch_size = (1024, 1024)
        args.chop = True
        args.tile = False
        args.no_augment = True
        args.scale = '2'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.no_augment:
            flag_aug = ''
        else:
            flag_aug = 'aug_'
        # args.data_train = 'unreg_real_lr'
        # args.data_test = 'unreg_real_lr'
        args.save = flag_aug + args.template+'_'+args.model + '_' + args.data_train
        #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        
    if args.template == 'FD_Unet_x3':
        args.model = 'fd_unet'
        args.patch_size = (64, 64)
        args.num_features = 64
        args.test_patch_size = (1024, 1024)
        args.chop = True
        args.tile = False
        args.no_augment = True
        args.scale = '3'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.no_augment:
            flag_aug = ''
        else:
            flag_aug = 'aug_'
        # args.data_train = 'unreg_real_lr'
        # args.data_test = 'unreg_real_lr'
        args.save = flag_aug + args.template+'_'+args.model + '_' + args.data_train
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'
        
    if args.template == 'FD_Unet_x4':
        args.model = 'fd_unet'
        args.patch_size = (64, 64)
        args.num_features = 64
        args.test_patch_size = (1024, 1024)
        args.chop = True
        args.tile = False
        args.no_augment = True
        args.scale = '4'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.no_augment:
            flag_aug = ''
        else:
            flag_aug = 'aug_'
        # args.data_train = 'unreg_real_lr'
        # args.data_test = 'unreg_real_lr'
        args.save = flag_aug + args.template+'_'+args.model + '_' + args.data_train
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'
        
    if args.template == 'FD_Unet_x8':
        args.model = 'fd_unet'
        args.patch_size = (64, 64)
        args.num_features = 64
        args.test_patch_size = (1024, 1024)
        args.chop = True
        args.tile = False
        args.no_augment = True
        args.scale = '8'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.no_augment:
            flag_aug = ''
        else:
            flag_aug = 'aug_'
        # args.data_train = 'unreg_real_lr'
        # args.data_test = 'unreg_real_lr'
        args.save = flag_aug + args.template+'_'+args.model + '_' + args.data_train
        #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'
######################################### influence of burst ###########################
    # if args.template == 'swinir_burst':
    #     args.model = 'Swinir'
    #     args.patch_size = (64, 64)
    #     args.num_features = 180
    #     args.burst_size = 3
    #     args.n_colors = args.burst_size
    #     args.tile = True
    #     args.scale = '2'
    #     args.data_train = 'burst_v3'
    #     args.data_test = 'burst_v3'
    #     # args.data_train = 'mid_filter'
    #     # args.data_test = 'mid_filter'
    #     args.rgb_range = 1
    #     if args.patch_select=='random':
    #         flag = ''
    #     else:
    #         flag = args.patch_select + '_'

    #     args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
    #     args.loss = '1*RL1'
    #     args.save = 'swinir_burst_x' + str(args.scale) + '_' +str(args.burst_size)+ '_' + args.loss
    #     # #args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
    #     # #args.pre_train = '../experiment/' + flag +'_'+'v1_polar_swinir_L1_x2_burst-' + str(args.burst_size) + '_' + args.loss + '/'+str(args.fold)+'/model/model_best.pt'
    #     # #args.pre_train = '/mnt/pank/SPSR/experiment/ss_swinir_burst_x2_3_1*RL1/0/model/model_best.pt'
    #     args.no_augment = True

    if args.template == 'swinir_burst_v1_x2':
        args.model = 'swinir_burst_1'
        args.patch_size = 64
        args.num_features = 180
        # args.burst_size = 3
        args.scale = '2'
        args.loss = '1*RL1'
        args.output_channels = 1
        args.tile = True
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.save = flag+'swinir_burst_v1_x2-' + str(args.burst_size)+ '_' + args.loss
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.batch_size = 1
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.rgb_range = 1
        args.no_augment = True
        # #args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # #args.pre_train = '../experiment/'+flag+'swinir_burst_v1_x2-' + str(args.burst_size)+ '_' + args.loss+'/'+str(args.fold)+'/model/model_best.pt'


    if args.template==('bipnet_swinir_x2'):
        args.model = 'swinir_burst'
        args.patch_size = 64
        args.num_features = 180
        args.burst_size = 3
        args.scale = '2'
        args.loss = '1*RL1'
        args.output_channels = 1
        args.tile = True
        args.save = 'bipnet_swinir_x2_burst-' + str(args.burst_size)+ '_' + args.loss
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.batch_size = 1
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.rgb_range = 1
        args.no_augment = True
        
        # #args.pre_train = '../experiment/bipnet_swinir_x2_burst-'+ str(args.burst_size) + '_' + args.loss+'/0/model/model_best.pt'
######################################### no_pre_train##################################
    if args.template==('polar_swinir_L1_nopretrain_x4'):
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '4'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
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
        # #args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        #args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_nopretrain_downgt_x4_burst-' + str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'
        args.no_augment = True

    if args.template==('EDSR_paper_nopretrain_downgt_x2'):
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '2'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
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
        # #args.pre_train = '../experiment1/pretrain_model/edsr_x2-0edfb8a3.pt'
        args.no_augment = True
        # #args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'EDSR_paper_x2_NoAugu_burst-'+ str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'
#########################################################################################
######################################### bsr ########################################
#########################################################################################
    if args.template == 'proposed_x2_syn_bsr':
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '2'
        args.data_train = 'burst_bsr'
        args.data_test = 'burst_bsr'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (512, 512)
        args.save = flag+ args.template+'_' + args.loss + '_' + args.data_train
        # #args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        #args.pre_train = '../experiment/'+ args.save + '/'+str(args.fold)+'/model/model_latest.pt'



#########################################################################################
######################################### single ########################################
#########################################################################################
    # if args.template == 'proposed_gradlayer_x2':
    #     args.model = 'Swinir_gradlayer'
    #     args.patch_size = (64, 64)
    #     args.num_features = 180
    #     args.burst_size = 1
    #     args.tile = True
    #     args.scale = '2'
    #     args.data_train = 'burst_v3'
    #     args.data_test = 'burst_v3'
    #     args.rgb_range = 1
    #     if args.patch_select=='random':
    #         flag = ''
    #     else:
    #         flag = args.patch_select + '_'
    #     args.no_augment = True
    #     args.test_patch_size = (512, 512)
    #     if args.rec:
    #         flag_rec = 'rec'
    #     else:
    #         flag_rec = 'norec'
    #     args.save = flag+'proposed_gradlayer_x2_' + args.loss + '_' + flag_rec
    #     #args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
    
    if args.template == 'proposed_x2_aug':
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '2'
        args.data_train = 'reg_mid'
        args.data_test = 'reg_mid'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = False
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        if args.rec:
            flag_rec = 'rec'
        else:
            flag_rec = 'norec'
        if args.add_spmap:
            flag_spmap = args.spmap_mode
        else:
            flag_spmap = ''
        args.save = flag+'proposed_x2_aug_' + args.loss + '_' + flag_rec + '_' + flag_spmap + '_' + args.data_train
        # #args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # #args.pre_train = '../experiment/'+ args.save + '/'+str(args.fold)+'/model/model_best.pt'
    
    if args.template == 'none_x2':
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '2'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        if args.rec:
            flag_rec = 'rec'
        else:
            flag_rec = 'norec'
        if args.add_spmap:
            flag_spmap = args.spmap_mode
        else:
            flag_spmap = ''
        args.save = flag+'proposed_x2_' + args.loss + '_' + flag_rec + '_' + flag_spmap + '_' + args.data_train
        
    if args.template == 'proposed_x2':
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '2'
        args.data_train = 'reg_mid'
        args.data_test = 'reg_mid'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        if args.rec:
            flag_rec = 'rec'
        else:
            flag_rec = 'norec'
        if args.add_spmap:
            flag_spmap = args.spmap_mode
        else:
            flag_spmap = ''
        args.save = flag+'proposed_x2_' + args.loss + '_' + flag_rec + '_' + flag_spmap + '_' + args.data_train
        # #args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # #args.pre_train = '../experiment/'+ args.save + '/'+str(args.fold)+'/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/{}'+'/model/model_best.pt'

        
    if args.template == 'proposed_x3':
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '3'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1008 // int(args.scale), 1008 // int(args.scale))
        if args.rec:
            flag_rec = 'rec'
        else:
            flag_rec = 'norec'
        if args.add_spmap:
            flag_spmap = args.spmap_mode
        else:
            flag_spmap = ''
        args.save = flag+'proposed_x3_' + args.loss + '_' + flag_rec + '_' + flag_spmap + '_' + args.data_train
        #args.pre_train = '../experiment/'+ args.save + '/'+str(args.fold)+'/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/'+str(args.fold)+'/model/model_best_x2.pt'
        
    if args.template == 'proposed_x4':
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '4'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        if args.rec:
            flag_rec = 'rec'
        else:
            flag_rec = 'norec'
        args.save = flag+'proposed_x4_' + args.loss + '_' + flag_rec + '_' + args.data_train
        #args.pre_train = '../experiment/'+ args.save + '/'+str(args.fold)+'/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/'+str(args.fold)+'/model/model_best_x2.pt'

    if args.template == 'proposed_x8':
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '8'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        if args.rec:
            flag_rec = 'rec'
        else:
            flag_rec = 'norec'
        if args.add_spmap:
            flag_spmap = args.spmap_mode
        else:
            flag_spmap = ''
        args.save = flag+'proposed_x8_' + args.loss + '_' + flag_rec + '_' + flag_spmap + '_' + args.data_train
        #args.pre_train = '../experiment/'+ args.save + '/'+str(args.fold)+'/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/'+str(args.fold)+'/model/model_best_x2.pt'

###########################################################################

    if args.template==('polar_swinir_GradientL1_x2'):
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '2'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.test_patch_size = (512 // down_scale, 512)
        # if args.test_only:
        #     args.test_patch_size = (512 // down_scale, 512)
        # else:
        #     args.test_patch_size = (256 // down_scale, 2?56)
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # 10L1? 40:bury
        # args.loss = '1*Gradient_L1'
        args.save = flag+'L1Warm_'+'polar_swinir_GradientL1_x2_burst-' +str(args.burst_size)+ '_' + args.loss + '_lr_'+ str(args.lr)
        # args.save = 'grad_window_L1Warm_polar_swinir_GradientL1_x2_burst-1_1*RL1+1*rec'
        #args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        # #args.pre_train = '../experiment/'+ args.save + '/'+str(args.fold)+'/model/model_best.pt'
        # #args.pre_train = '../experiment/'+flag+'L1Warm_'+'polar_swinir_GradientL1_x2_burst-' + str(args.burst_size) + '_' + args.loss + '/'+str(args.fold)+'/model/model_best.pt'
        args.no_augment = True

    if args.template==('polar_swinir_GradientL1_x4'):
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '4'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (256 // down_scale, 256)
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        # if args.test_only:
        #     args.test_patch_size = (256 // down_scale, 256)
        # else:
        #     args.test_patch_size = (128 // down_scale, 128)
        # args.loss = '1*Gradient_L1'  # 10L1? 40:bury
        args.save = flag+'L1Warm_'+str(args.downsample_gt)+'_'+'v2_polar_swinir_GradientL1_x4_burst-' +str(args.burst_size)+ '_' + args.loss
        #args.pre_train = '../experiment/'+flag+'L1Warm_'+str(args.downsample_gt)+'_'+'v2_polar_swinir_GradientL1_x4_burst-' +str(args.burst_size)+ '_' + args.loss+ '/'+str(args.fold)+'/model/model_best_x4_L1.pt'
        # #args.pre_train = '../experiment/'+flag+'L1Warm_'+str(args.downsample_gt)+'_'+'v2_polar_swinir_GradientL1_x4_burst-' +str(args.burst_size)+ '_' + args.loss+ '/'+str(args.fold)+'/model/model_best.pt'
        args.no_augment = True

####################################################################################################
    if args.template == 'edsr_x2':
        args.model = 'EDSR'#'Swinir'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '2'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.save = args.template+'_'+args.model + '_' + args.data_train
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'
        
    if args.template == 'edsr_x3':
        args.model = 'EDSR'#'Swinir'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '3'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.save = args.template+'_'+args.model + '_' + args.data_train
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'
    
    if args.template == 'han_x2':
        args.model = 'HAN'#'Swinir'
        args.n_resblocks = 20
        args.n_feats = 128
        args.res_scale = 1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '2'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        if args.no_augment:
            flag_aug = ''
        else:
            flag_aug = 'aug_'
        args.save = flag_aug + args.template+'_'+args.model + '_' + args.data_train
        #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'       
      
    if args.template == 'han_x3':
        args.model = 'HAN'#'Swinir'
        args.n_resblocks = 20
        args.n_feats = 128
        args.res_scale = 1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '3'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1008 // int(args.scale), 1008 // int(args.scale))
        if args.no_augment:
            flag_aug = ''
        else:
            flag_aug = 'aug_'
        args.save = flag_aug + args.template+'_'+args.model + '_' + args.data_train
        #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'
        args.n_threads = 0
        
    if args.template == 'han_x4':
        args.model = 'HAN'#'Swinir'
        args.n_resblocks = 20
        args.n_feats = 128
        args.res_scale = 1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '4'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        if args.no_augment:
            flag_aug = ''
        else:
            flag_aug = 'aug_'
        args.save = flag_aug + args.template+'_'+args.model + '_' + args.data_train
        #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'
        
    if args.template == 'han_x8':
        args.model = 'HAN'#'Swinir'
        args.n_resblocks = 20
        args.n_feats = 128
        args.res_scale = 1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '4'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        if args.no_augment:
            flag_aug = ''
        else:
            flag_aug = 'aug_'
        args.save = flag_aug + args.template+'_'+args.model + '_' + args.data_train
        #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'
        
    if args.template == 'rcan_x2':
        args.model = 'RCAN'#'Swinir'
        args.n_resblocks = 20
        args.n_feats = 128
        args.res_scale = 1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '2'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.save = args.template+'_'+args.model + '_' + args.data_train
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'

    if args.template == 'rcan_x3':
        args.model = 'RCAN'#'Swinir'
        args.n_resblocks = 20
        args.n_feats = 128
        args.res_scale = 1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '3'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.save = args.template+'_'+args.model + '_' + args.data_train
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'
        
    if args.template == 'rcan_x4':
        args.model = 'RCAN'#'Swinir'
        args.n_resblocks = 20
        args.n_feats = 128
        args.res_scale = 1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '4'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.save = args.template+'_'+args.model + '_' + args.data_train
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'
        
    if args.template == 'rcan_x8':
        args.model = 'RCAN'#'Swinir'
        args.n_resblocks = 20
        args.n_feats = 128
        args.res_scale = 1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '8'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.save = args.template+'_'+args.model + '_' + args.data_train

    if args.template == 'rdn_x2':
        args.model = 'RDN'#'Swinir'
        args.n_resblocks = 16
        args.n_feats = 256
        args.GO = 64
        args.res_scale = 1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '2'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.save = args.template+'_'+args.model + '_' + args.data_train
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'
     
    if args.template == 'rdn_x3':
        args.model = 'RDN'#'Swinir'
        args.n_resblocks = 16
        args.n_feats = 256
        args.GO = 64
        args.res_scale = 1
        
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '3'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.save = args.template+'_'+args.model + '_' + args.data_train
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'

    if args.template == 'mdsr_x2':
        args.model = 'MDSR'#'Swinir'
        args.n_resblocks = 16
        args.n_feats = 256
        args.res_scale = 1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '2'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.save = args.template+'_'+args.model + '_' + args.data_train
        # args.n_threads = 0
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'
  
    if args.template == 'mdsr_x3':
        args.model = 'MDSR'#'Swinir'
        args.n_resblocks = 16
        args.n_feats = 256
        args.res_scale = 1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '3'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.save = args.template+'_'+args.model + '_' + args.data_train
        args.n_threads = 0
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'

    if args.template == 'mdsr_x4':
        args.model = 'MDSR'#'Swinir'
        args.n_resblocks = 16
        args.n_feats = 256
        args.res_scale = 1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = False
        args.chop = True
        args.scale = '4'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.save = args.template+'_'+args.model + '_' + args.data_train
        args.n_threads = 0
        #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best.pt'
        # #args.pre_train = '../experiment/'+ args.save + '/0/model/model_best_x2.pt'
    
    if args.template == 'EDSR_paper_x2':
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
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
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
        args.loss = '1*RL1'
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        # args.save = 'test'
        args.save = flag+'EDSR_paper_x2-' + str(args.burst_size) + '_' + args.loss + '_' + args.data_train
        #args.pre_train = '../experiment1/pretrain_model/edsr_x2-0edfb8a3.pt'
        args.no_augment = True
        # #args.pre_train = '../experiment/'+flag+'EDSR_paper_x2-'+ str(args.burst_size) + '_' + args.loss + '/'+str(args.fold)+'/model/model_best.pt'

    if args.template==('EDSR_paper_x4'):
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '4'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
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
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.save = flag + str(args.downsample_gt)+'_'+'V1_EDSR_paper_x4_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss
        # #args.pre_train = '/mnt/pank/SPSR/experiment/False_V1_EDSR_paper_x4_NoAugu_burst-1_'+args.loss+'/0/model/model_best_x2.pt'
        args.no_augment = True
        #args.pre_train = '/mnt/pank/SPSR/experiment/False_V1_EDSR_paper_x4_NoAugu_burst-1_'+args.loss + '/'+str(args.fold)+'/model/model_best.pt'

    if args.template==('EDSR_paper_x8'):
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '8'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
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
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.save = flag + str(args.downsample_gt)+'_'+'V2_EDSR_paper_x8_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss
        #args.pre_train = '../experiment/' + str(args.downsample_gt)+'_'+'V2_EDSR_paper_x8_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss+'/model/model_best_x4.pt'
        args.no_augment = True
        # #args.pre_train = '../experiment/' + str(args.downsample_gt)+'_'+'V2_EDSR_paper_x8_NoAugu_burst-' + str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'
########################################## swinir_L1 #################################################
    if args.template==('polar_swinir_L1_x2'):
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '2'
        # args.data_train = 'burst_v3'
        # args.data_test = 'burst_v3'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
        args.rgb_range = 1
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1

        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'

        args.test_patch_size = (512 // down_scale, 512)
        # if args.test_only:
        #     args.test_patch_size = (512 // down_scale, 512)
        # else:
        #     args.test_patch_size = (256 // down_scale, 256)
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # 10L1? 40:bury
        args.loss = '1*RL1'
        args.save = flag+'v1_polar_swinir_L1_x2_burst-' +str(args.burst_size)+ '_' + args.loss
        # #args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        #args.pre_train = '../experiment/' + flag +'v1_polar_swinir_L1_x2_burst-' + str(args.burst_size) + '_' + args.loss + '/'+str(args.fold)+'/model/model_best.pt'
 
        args.no_augment = True

    if args.template==('polar_swinir_L1_x4'):
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '4'
        args.loss = '1*RL1'
        args.n_colors = args.burst_size
        # args.data_train = 'burst_v3'
        # args.data_test = 'burst_v3'
        args.data_train = 'mid_filter'
        args.data_test = 'mid_filter'
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
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.save = flag +'v1_polar_swinir_L1_x4_burst-' +str(args.burst_size)+ '_' + args.loss
        # #args.pre_train = '../experiment/' + flag +'v1_polar_swinir_L1_x4_burst-' + str(args.burst_size) + '_' + args.loss + '/'+str(args.fold)+'/model/model_best_x2_L1.pt'
        # #args.pre_train = '../experiment/' + flag + +'v1_polar_swinir_L1_x4_burst-' + str(args.burst_size) + '_' + args.loss + '/'+str(args.fold)+'/model/model_best.pt'
        args.no_augment = True

########################################################################################
################################### odd-even divide#####################################
########################################################################################
########################################## swinir_L1 #################################################
    if args.template==('polar_swinir_L1_downgt_x2'):
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '2'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
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
        # #args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        #args.pre_train = '../experiment3/'+str(args.downsample_gt)+'_'+'noact_polar_swinir_L1_downgt_x2_burst-' + str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'
        args.no_augment = True
        
    if args.template==('polar_swinir_L1_downgt_x4'):
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '4'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
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
        # #args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_downgt_x2_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        #args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_downgt_x4_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True

    if args.template==('polar_swinir_L1_downgt_x8'):
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '8'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (128 // down_scale, 128)

        args.save = str(args.downsample_gt)+'_'+'polar_swinir_L1_downgt_x8_burst-' +str(args.burst_size)+ '_' + args.loss
        # #args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_downgt_x4_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        #args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_L1_downgt_x8_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True


################################### swinir_GradientL1 ##################################
    if args.template==('polar_swinir_GradientL1_downgt_x2'):
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        # args.tile = True
        args.tile = True
        args.scale = '2'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
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
        # #args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        #args.pre_train = '../experiment3/'+str(args.downsample_gt)+'_'+'noact_polar_swinir_GradientL1_downgt_x2_burst-' + str(args.burst_size) + '_' + args.loss + '/model/model_best.pt'
        args.no_augment = True

    if args.template==('polar_swinir_GradientL1_downgt_x4'):
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '4'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (256 // down_scale, 256)
        args.save = str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_downgt_x4_burst-' +str(args.burst_size)+ '_' + args.loss
        # #args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_downgt_x4_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_epoch_5.pt'
        #args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_downgt_x4_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True

    if args.template==('polar_swinir_GradientL1_downgt_x8'):
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '8'
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
        args.rgb_range = 1
        args.downsample_gt = False
        if args.downsample_gt:
            down_scale = 2
        else:
            down_scale = 1
        args.test_patch_size = (128 // down_scale, 128)

        args.save = str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_downgt_x8_burst-' +str(args.burst_size)+ '_' + args.loss
        # #args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_downgt_x8_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_epoch_1.pt'
        #args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_swinir_GradientL1_downgt_x8_burst-' +str(args.burst_size)+ '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True
########################################################################################
################################### rec_loss ###########################################
########################################################################################








#########################################################################################
######################################### burst #########################################
#########################################################################################
    if args.template==('polar_bipnet_swinir_GradientL1_x2'):
        args.model = 'swinir_burst'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 5
        args.scale = '2'
        # args.loss = '200*Gradient_L1+0.5*VGG54'  # [10,4,0.1]->[100,1,0.1]
        args.output_channels = 1
        args.tile = True
        args.data_train = 'burst_v3'
        args.data_test = 'burst_v3'
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
        # #args.pre_train = '../experiment1/pretrain_model/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
        #args.pre_train = '../experiment/'+str(args.downsample_gt)+'_'+'polar_bipnet_swinir_GradientL1_x2_burst-'+ str(args.burst_size) + '_' + args.loss+'/model/model_best.pt'
        args.no_augment = True

##################################################################################################
def set_template(args):
    # Set the templates here
    if args.template == 'test_only':
        args.model = 'Swinir'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 1
        args.tile = True
        args.scale = '8'

        args.data_train = 'unreg_real_lr'
        args.data_test = 'unreg_real_lr'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.no_augment = True
        args.test_patch_size = [500,1000]#(1024 // int(args.scale), 1024 // int(args.scale))
        args.save = args.template+'_'+args.model + '_' + args.data_train + '_nomid'+'x_{}'.format(args.scale)
        args.pre_train = '/mnt/pank/SPSR/experiment9/grad_window_proposed_x8_1*RL1+0.5*rec_norec_burst_v3/0/model/model_best.pt' 
        
        
    if args.template == 'swinir_burst1':
        args.model = 'swinir_burst_sifa'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 3
        # args.n_colors = args.burst_size
        args.tile = True
        args.chop = False
        # args.scale = '2'
        args.data_train = 'reg_mid'
        args.data_test = 'reg_mid'
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
        
    if args.template == 'swinir_burst2':
        args.model = 'swinir_burst_sifa2'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 3
        # args.n_colors = args.burst_size
        args.tile = True
        args.chop = False
        # args.scale = '2'
        args.data_train = 'reg_mid'
        args.data_test = 'reg_mid'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.loss = '1*RL1'
        args.save =  flag + args.template + '_x' + str(args.scale) + '_' + str(args.burst_size) + '_' + args.model + '_' + args.data_train + '_' + args.loss
        args.no_augment = True
        
    if args.template == 'swinir_burst3':
        args.model = 'swinir_burst_sifa3'
        args.patch_size = (64, 64)
        args.num_features = 180
        args.burst_size = 3
        # args.n_colors = args.burst_size
        args.tile = True
        args.chop = False
        # args.scale = '2'
        args.data_train = 'reg_mid'
        args.data_test = 'reg_mid'
        args.rgb_range = 1
        if args.patch_select=='random':
            flag = ''
        else:
            flag = args.patch_select + '_'
        args.test_patch_size = (1024 // int(args.scale), 1024 // int(args.scale))
        args.loss = '1*RL1'
        args.save =  flag + args.template + '_x' + str(args.scale) + '_' + str(args.burst_size) + '_' + args.model + '_' + args.data_train + '_' + args.loss
        args.no_augment = True

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
        args.save = flag + args.template + '_x' + str(args.scale) + '_' + str(args.burst_size) + '_' + args.model + '_' + args.data_train + '_' + args.loss
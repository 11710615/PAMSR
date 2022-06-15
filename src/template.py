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
        # args.save = 'swinir_ema_x' + args.scale + '_' + args.loss
    
    
    if args.template.find('swinir_ema_x4') >= 0:
        args.scale = '4'
        args.model = 'SwinIR'
        args.patch_size = 64
        args.chop = True
        args.use_ema = True
        args.save = 'swinir_ema_x4' + '_' + args.loss
    
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
        
    
    if args.template.find('swinir_hf_x4') >= 0:
        args.model = 'swinir_sp'
        args.patch_size = 64
        args.chop = True
        args.output_channels = 1
        args.scale = '4'
        args.save = 'swinir_hf_x4'
        args.reset = True

    if args.template.find('swinir_sp_div2k') >= 0:
        args.model = 'swinir_sp'
        args.patch = 64
        args.chop = True
        args.data_train = 'DIV2K_train'
        args.data_test = 'DIV2K_valid'
        args.data_range = '1-800/1-100'
        args.output_channels = 3
        args.batch_size = 3
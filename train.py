'''
Training stage for auto encoder 
'''
import os
import sys
import argparse
import logging
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import torch.autograd
import time
from dataset.dataset import what3d_dataset_views
from model.generator_res_Folding import GeneratorVanilla
#from model.generator_folding import GeneratorVanilla
from model.atlasnet.model import EncoderDecoder
from argument_parser import parser 
from utils.utils import count_parameter_num, check_exist_or_mkdirs
from utils.loss import ChamferDistance
from traintester import TrainTester
import model.im2mesh.config as config

abspath = os.path.dirname(os.path.abspath(__file__))

def main(args, logger):

    # load data
    starter_time = time.time()
    kwargs = {'num_workers':args.num_worker, 'pin_memory':True}

    if args.test:
        if args.generate_mesh:
            args.test_batch_size = 1
        logger.info("loading test data ...")
        test_loader = torch.utils.data.DataLoader(
                            what3d_dataset_views(data_basedir=args.data_basedir, ptcloud_path=args.ptcloud_path, 
                    img_path=args.img_path, label_path=args.label_path, 
                    splits_path=args.splits_path, split_name='test', 
                    class_path=args.class_path, sample_ratio=args.sample_ratio,
                    image_height=args.image_size, image_width=args.image_size, 
                    views=list(args.views),
                    points_num=args.pts_num, mode=args.mode, normalize=args.normalize),
                    batch_size=args.test_batch_size, shuffle=False,**kwargs)

    if args.train:
        logger.info("loading train data ...")
        train_loader = torch.utils.data.DataLoader(
                    what3d_dataset_views(data_basedir=args.data_basedir, ptcloud_path=args.ptcloud_path, 
                    img_path=args.img_path, label_path=args.label_path, 
                    splits_path=args.splits_path, split_name='train', 
                    class_path=args.class_path, sample_ratio=args.sample_ratio,
                    image_height=args.image_size, image_width=args.image_size, 
                    views=list(args.views),
                    points_num = args.pts_num, mode = args.mode, normalize=args.normalize),
                    batch_size=args.train_batch_size, shuffle=True,**kwargs)

        logger.info("loading val data ...")
        val_loader = torch.utils.data.DataLoader(
                            what3d_dataset_views(data_basedir=args.data_basedir, ptcloud_path=args.ptcloud_path, 
                    img_path=args.img_path, label_path=args.label_path, 
                    splits_path=args.splits_path, split_name='val', 
                    class_path=args.class_path, sample_ratio=args.sample_ratio,
                    image_height=args.image_size, image_width=args.image_size, 
                    views=list(args.views),
                    points_num=args.pts_num, mode=args.mode, normalize=args.normalize),
                    batch_size=args.val_batch_size, shuffle=False,**kwargs)

    logger.info("Initialize cache={}".format(time.time()-starter_time))
    
    if args.model == "foldingres":
        netG = GeneratorVanilla(
            grid_dims=(int(np.sqrt(args.pts_num)),int(np.sqrt(args.pts_num)),1),
            hidden_neurons=args.hidden_neurons,
            num_layers=args.num_layers,
            bottleneck_size=args.bottleneck_size,
            class_num=args.class_num,
            folding_twice=args.folding_twice,
            device=args.device,
            remove_all_batchNorms=args.remove_all_batchNorms,
            decoderBlock=args.decoderBlock
            )

    elif args.model == 'psgn':
        netG = config.get_model(bottleneck_size=args.bottleneck_size,
                               pts_num=args.pts_num)

    elif args.model == 'atlasnet':
        netG = EncoderDecoder(args)
        
    netG.to(args.device)

    logger.info('Number of parameters={}'.format(count_parameter_num(netG.parameters())))
    logger.info('Network Architecture:')
    logger.info(str(netG))
    sys.stdout.flush()

    # set solver and loss function
    criterion_G = ChamferDistance().to(args.device)

    optmizer_G = torch.optim.Adam(
        netG.parameters(),
        lr=args.lr_G,
        betas=(args.G_adam_beta1, 0.999),
        weight_decay=args.weight_decay,
         )    
    
    lr_scheduler_G = torch.optim.lr_scheduler.StepLR(
        optmizer_G,
        step_size=args.lr_decay_step,
        gamma=args.lr_decay_rate,
    )

    # train and test#
    runner = TrainTester(
            netG=netG,
            criterion_G=criterion_G,
            optmizer_G=optmizer_G,
            lr_scheduler_G=lr_scheduler_G,
            logger=logger,
            args=args,
        )

    if args.train:
        runner.run(
            train_loader=train_loader,
            test_loader= test_loader,
            val_loader = val_loader
        )
        logger.info('Training Done!')

    if args.test:
        runner.netG.load_state_dict(torch.load(os.path.join(args.log_dir,"model_train_best.pth")))
        #runner.optimizer_G.load_state_dict(torch.load(os.path.join(args.log_dir,"solver_train_best.pth")))
        if args.generate_mesh:
            runner.generate_mesh(
                epoch=args.total_epochs + 1,
                loader=test_loader,
                type = 'test'
            )
        else:
            runner.test(
                epoch=args.total_epochs + 1,
                loader=test_loader,
                type = 'test'
            )
        logger.info('Testing Done!')
    


if __name__ == "__main__":
    logger = logging.getLogger()
    args = parser()
    check_exist_or_mkdirs(args.log_dir)
    file_log_handler = logging.FileHandler(os.path.join(args.log_dir,"Train.log"))
    logger.addHandler(file_log_handler)
    stderr_log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stderr_log_handler)
    logger.setLevel("INFO")
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s","%Y-%m-%d %H:%M:%S")
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)
    sys.stdout.flush()
    
    for i, key in enumerate(args):
        logger.info(" {}:   {}; ".format(key, args[key]))

    main(args, logger)
















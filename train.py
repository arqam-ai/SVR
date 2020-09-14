'''
Training stage for auto encoder 
'''
import os
import sys
import argparse
import glog as logger
import logging
import open3d as o3d
import torch
import torch.utils.data
import torch.nn as nn
import torch.autograd
import optparse
import time
from dataset.dataset import what3d_dataset_views
from model.generator_res_Folding import GeneratorVanilla
from utils.utils import count_parameter_num, init_weights, check_exist_or_mkdirs
from utils.loss import ChamferDistance
from traintester import TrainTester
import model.im2mesh.config as config

abspath = os.path.dirname(os.path.abspath(__file__))


def main(args):
    
    # load data
    starter_time = time.time()
    kwargs = {'num_workers':16, 'pin_memory':True}
    print("loading train data ...")
    train_loader = torch.utils.data.DataLoader(
                what3d_dataset_views(data_basedir=args.data_basedir, ptcloud_path=args.ptcloud_path, 
                img_path=args.img_path, label_path=args.label_path, 
                splits_path=args.splits_path, split_name='train', 
                class_path=args.class_path, sample_ratio=args.sample_ratio,
                image_height=args.image_size, image_width=args.image_size, 
                views=list(args.views), read_view=args.read_view,
                points_num = args.pts_num, mode = args.mode),
                batch_size=args.train_batch_size, shuffle=True,**kwargs)
    print("loading test data ...")
    test_loader = torch.utils.data.DataLoader(
                        what3d_dataset_views(data_basedir=args.data_basedir, ptcloud_path=args.ptcloud_path, 
                img_path=args.img_path, label_path=args.label_path, 
                splits_path=args.splits_path, split_name='test', 
                class_path=args.class_path, sample_ratio=args.sample_ratio,
                image_height=args.image_size, image_width=args.image_size, 
                views=list(args.views), read_view=args.read_view,
                points_num=args.pts_num, mode=args.mode),
                batch_size=args.test_batch_size, shuffle=False,**kwargs)

    print("loading val data ...")
    val_loader = torch.utils.data.DataLoader(
                        what3d_dataset_views(data_basedir=args.data_basedir, ptcloud_path=args.ptcloud_path, 
                img_path=args.img_path, label_path=args.label_path, 
                splits_path=args.splits_path, split_name='val', 
                class_path=args.class_path, sample_ratio=args.sample_ratio,
                image_height=args.image_size, image_width=args.image_size, 
                views=list(args.views), read_view=args.read_view,
                points_num=args.pts_num, mode = args.mode),
                batch_size=args.val_batch_size, shuffle=False,**kwargs)
    print("Initialize cache={}".format(time.time()-starter_time))

    if args.model == "foldingres6" or args.model == "foldingres18":
        netG = GeneratorVanilla(
            grid_dims=(32,32,1),
            resgen_width=512,
            resgen_depth=5,
            resgen_codelength=512,
            class_num=55,
            block=args.model,
            read_view=args.read_view,
            folding_twice=args.folding_twice)

    elif args.model == 'psgn':
        netG = config.get_model(config.load_config(path = os.path.join(abspath, 'model/im2mesh/configs/img/psgn.yaml'),
					default_path = os.path.join(abspath,'model/im2mesh/configs/default.yaml')), device = args.device)
    	
    #netG = torch.nn.DataParallel(netG, device_ids=[0, 1])
    logger = logging.getLogger()
    logger.info('Number of parameters={}'.format(count_parameter_num(netG.parameters())))
    check_exist_or_mkdirs(args.log_dir)
    file_log_handler = logging.FileHandler(os.path.join(args.log_dir,args.log_filename))
    logger.addHandler(file_log_handler)

    logger.setLevel('INFO')
    formatter = logging.Formatter()
    file_log_handler.setFormatter(formatter)
    #stderr_log_handler.setFormatter(formatter)
    logger.info(args)

    # set solver and loss function
    criterion_G = ChamferDistance().to(args.device)
    criterion_C = torch.nn.CrossEntropyLoss().to(args.device)
    criterion_M = torch.nn.MSELoss(reduction='mean')

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
            criterion_C=criterion_C,
            criterion_M=criterion_M,
            optmizer_G=optmizer_G,
            lr_scheduler_G=lr_scheduler_G,
            alpha = args.alpha,
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
        runner.netG.load_state_dict(torch.load(os.path.join(args.snapshot_dir,"model_train_best.pth")))
        runner.optimizer_G.load_state_dict(torch.load(os.path.join(args.snapshot_dir,"solver_train_best.pth")))

        runner.test(
            epoch=args.total_epochs + 1,
            loader=test_loader,
            type = 'test'
        )
        logger.info('Testing Done!')
    


if __name__ == "__main__":
    parser = optparse.OptionParser(sys.argv[0], description="Training Encoder_decoder")

    # dataset info
    parser.add_option("--ptcloud-path",
                      dest="ptcloud_path", type=str,
                      default="points_1024",
                      help='path of the ptcloud')

    parser.add_option("--img-path",
                      dest="img_path", type=str,
                      default='renderings',
                      help='path of the image')

    parser.add_option("--label-path",
                      dest="label_path", type=str,
                      default='label.npz',
                      help='path of the image')
    
    parser.add_option("--data-basedir",
                      dest="data_basedir", type=str,
                      default='/home/../../public/zyf/What3D',
                      help='path of the data folder') 

    parser.add_option("--splits-path",
                      dest="splits_path", type=str,
                      default='splits',
                      help='path of the data folder') 

    parser.add_option("--class-path",
                      dest="class_path", type=str,
                      default='classes.txt',
                      help="class name list")

    parser.add_option("--image-size",
                      dest="image_size", type=int,
                      default = 224,
                      help="image size for network")
 
    parser.add_option("--sample-ratio",
                      dest="sample_ratio", type=float,
                      default = 0.001,
                      help="ratio to sample the dataset")

    parser.add_option("--views",
                      dest="views", type=str,
                      default= '0',
                      help="five view for each instance")

    parser.add_option("--pts-num",
                      dest="pts_num", type=int,
                      default=1024,
                      help="number of points in a ptcloud")

    parser.add_option("--mode",
                      dest="mode", type=str,
                      default="viewer", 
                      help="['viewer', 'object']")

    # Log file info
    parser.add_option("--log-dir",
                      dest="log_dir", type=str,
                      default="logs/",
                      help="log folder to save training stats as numpy files")

    parser.add_option("--log-filename", type=str,
                      dest="log_filename",
                      default="Train.log",
                      help="Name of log file.")

    parser.add_option("--output-dir",
                      dest="output_dir", type=str,
                      default='results/',
                      help='result folder to save generated ptc during training')

    parser.add_option("--snapshot-dir",
                      dest="snapshot_dir", type=str,
                      default='snapshots/',
                      help='snapshot folder to save training checkpoint')

    parser.add_option('--verbose_per_n_batch',
                     dest="verbose_per_n_batch", type=int,
                     default= 5,
                     help='log training stats to console every n batch (<=0 disables training log)')

    parser.add_option('--test_per_n_epoch',
                     dest="test_per_n_epoch", type=int,
                     default= 1,
                     help='test per epoch')                 

    parser.add_option("--shuffle-point-order", action="store_true",
                      dest="shuffle_point_order",
                      default=False,
                      help="whether/how to shuffle point order (no/offline/online)")

    ## training parameter
    parser.add_option("--model",
                    dest="model", type=str,
                    default="foldingnet",
                    help=['foldingnet','psgn'])
                    
    parser.add_option("--use-manifold",
                      dest="use_manifold", action="store_true",
                      default=False,
                      help='whether to use manifold learning')

    parser.add_option("--alpha",
                      dest="alpha", type=float,
                      default= 0.,
                      help='mixup interpolation strength (uniform=1., ERM=0.)')

    parser.add_option("--total-epochs",
                      dest="total_epochs", type=int,
                      default=300,
                      help='training epochs')
    
    parser.add_option("--train-batch-size",
                      dest="train_batch_size", type=int,
                      default=128,
                    help='training batch size')
    parser.add_option("--test-batch-size",
                      dest="test_batch_size", type=int,
                      default=200,
                    help='testing batch size')
    parser.add_option("--val-batch-size",
                      dest="val_batch_size", type=int,
                      default=200,
                    help='val batch size')
    parser.add_option("--lr-G",
                      dest="lr_G", type=float,
                      default=1e-4,
                    help='learning rate')
    parser.add_option("--lr-D",
                      dest="lr_D", type=float, default=1e-4,
                      help='learning rate')
    parser.add_option("--pool-size",
                      dest="pool_size", type=int,
                      default=0,
                      help='history cache for training discriminator')
    parser.add_option("--G-adam-beta1",
                      dest="G_adam_beta1", type=float,
                      default=0.5,
                      help="beta1 for Adam optimizer (G)")
    parser.add_option("--mean",
                      dest="mean", type=float,
                      default=0.0,
                      help="value of mean for noise (Gaussian)")
    parser.add_option("--stddev",
                      dest="stddev", type=float,
                      default=1.0,
                      help="value of stddev for noise (Gaussian)")

    parser.add_option("--lambda-loss-primitive",
                      dest="lambda_loss_primitive", type=float,
                      default=1.0,
                      help="lambda for point cloud first generation")

    parser.add_option("--lambda-loss-fine",
                      dest="lambda_loss_fine", type=float,
                      default=1.0,
                      help="lambda for point cloud second generation")

    parser.add_option("--lambda-loss-manifold",
                      dest="lambda_loss_manifold", type=float,
                      default= 1.0,
                      help="lambda for codeword manifold")

    parser.add_option("--lambda-loss-classification",
                      dest="lambda_loss_classification", type=float,
                      default= 1.0,
                      help="lambda for img classification")

    parser.add_option('--momentum',type=float,
                      default= 0.9,
                     help='Solver momentum')
    parser.add_option("--weight-decay",
                      dest="weight_decay", type=float,
                      default= 1e-6,
                    help='weight decay')
    parser.add_option('--lr_decay_step',
                      type=int, default=30,
                      help='learning rate decay step'
                      )
    parser.add_option('--lr_decay_rate',
                      type=float, default=0.1,
                      help='learning rate decay rate'
                      )
    parser.add_option("--tensorboard", action="store_true",
                      dest="tensorboard",
                      default= None) 

    parser.add_option("--checkpoint-model", type="str",
                      dest="checkpoint_model",
                      #default= "snapshots/model_train_best.pth",
                      default= None,
                      help="Path to pretrained model G net")

    parser.add_option("--checkpoint-solver", type="str",
                      dest="checkpoint_solver",
                      #default= "snapshots/solver_train_best.pth",
                      default= None,
                      help="Path to pretrained solver")

    parser.add_option("--save-results", action = "store_true",
                      dest="save_results",
                      default = False,
                      help = "save the results in evaluation test")

    parser.add_option("--train", action="store_true",
                      dest="train",
                      default= False,
                      help="run training", )

    parser.add_option("--test", action="store_true",
                      dest="test",
                      default= False,
                      help="run testing (generate ptcloud)", )

    parser.add_option("--read-view", action="store_true",
                      dest="read_view",
                      default= False,
                      help="if adding the view information in training", )

    parser.add_option("--folding-twice", action="store_true",
                      dest="folding_twice",
                      default= False,
                      help="if folding twice", )
    
    (args, opts) = parser.parse_args()
   
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.script_folder = os.path.dirname(os.path.abspath(__file__))
    sys.stdout.flush()
    main(args)
















import argparse
import os
import datetime
import json
from easydict import EasyDict
from os.path import exists, join
import torch
from utils.utils import check_exist_or_mkdirs
#import auxiliary.my_utils as my_utils

"""
    Author : Thibault Groueix 01.11.2019
"""


def parser():
    parser = argparse.ArgumentParser()

    ## training parameter
    parser.add_argument("--SVR", action="store_true", default = True, help="Single_view Reconstruction")
    parser.add_argument("--model",dest="model", type=str, default="foldingres", help=['foldingres','psgn','atlasnet'])
    parser.add_argument("--train", action="store_true", dest="train", default= False, help="run training", )
    parser.add_argument("--test", action="store_true", dest="test", default= False, help="run testing (generate ptcloud)", )
    parser.add_argument("--generate-mesh", action="store_true", dest="generate_mesh", default=False, help='atlasnet generate mesh')

    parser.add_argument('--num_layers', type=int, default=2, help='number of hidden MLP Layer')
    parser.add_argument('--hidden_neurons', type=int, default=512, help='number of neurons in each hidden layer')
    parser.add_argument('--bottleneck_size', type=int, default=1024, help='dim_out_patch')
    parser.add_argument("--folding-twice", action="store_true",dest="folding_twice",default=False, help="if folding twice", )
    parser.add_argument('--nb_primitives', type=int, default=4, help='number of primitives')
    parser.add_argument('--template_type', type=str, default="SQUARE", choices=["SPHERE", "SQUARE"],
                        help='dim_out_patch')
    parser.add_argument("--remove_all_batchNorms", action="store_true", default=False, help="Replace all batchnorms by identity")
    parser.add_argument("--decoderBlock", type=str, default=None, help="[ResnetFC, ResnetFCBN, ResnetFCGN, FC, FCBN, FCGN]")
    parser.add_argument('--activation', type=str, default='relu',
             choices=["relu", "sigmoid", "softplus", "logsigmoid", "softsign", "tanh"], help='dim_out_patch')
    
    
    parser.add_argument("--total-epochs",dest="total_epochs", type=int,default=300,help='training epochs')
    parser.add_argument("--train-batch-size",dest="train_batch_size", type=int,default=128,help='training batch size')
    parser.add_argument("--test-batch-size",dest="test_batch_size", type=int,default=200,help='testing batch size')
    parser.add_argument("--val-batch-size",dest="val_batch_size", type=int,default=200,help='val batch size')
    parser.add_argument("--lr-G",dest="lr_G", type=float,default=1e-4,help='learning rate')
    parser.add_argument("--G-adam-beta1",dest="G_adam_beta1", type=float,default=0.5, help="beta1 for Adam optimizer (G)")
    parser.add_argument("--lambda-loss-primitive",dest="lambda_loss_primitive", type=float,default=1.0,help="lambda for point cloud first generation")
    parser.add_argument("--lambda-loss-fine",dest="lambda_loss_fine", type=float,default=1.0,help="lambda for point cloud second generation")
    parser.add_argument('--momentum',type=float,default= 0.9,help='Solver momentum')
    parser.add_argument("--weight-decay",dest="weight_decay", type=float,default= 1e-6,help='weight decay')
    parser.add_argument('--lr_decay_step',type=int, default=30,help='learning rate decay step')
    parser.add_argument('--lr_decay_rate',type=float, default=0.1,help='learning rate decay rate')

    parser.add_argument("--mode", dest="mode", type=str,default="object", help="['viewer', 'object']")
    parser.add_argument("--image-size",   dest="image_size",   type=int, default = 224, help="image size for network")
    parser.add_argument("--sample-ratio", dest="sample_ratio", type=float, default = 0.001, help="ratio to sample the dataset")
    parser.add_argument("--normalize", dest="normalize", type=str, default="unitL2ball", help="[bbox, unitL2ball]")
    parser.add_argument("--fscore-renorm", dest="fscore_renorm", type=str, default="unitL2ball", help="[bbox, unitL2ball]")

    parser.add_argument("--views", dest="views", type=str,default= '0', help="five view for each instance")
    parser.add_argument("--pts-num", dest="pts_num", type=int, default=1024,help="number of points in a ptcloud")
    parser.add_argument("--class-num", dest="class_num", type=int, default=55, help="number of classes" )
    parser.add_argument("--num-worker", dest="num_worker", type=int, default=4, help="number of cpu core to load data")

    parser.add_argument("--ptcloud-path", dest="ptcloud_path", type=str,   default="ptcloud_n.npz",                help='path of the ptcloud')
    parser.add_argument("--img-path",     dest="img_path",     type=str,   default='renderings',                   help='path of the image')
    parser.add_argument("--label-path",   dest="label_path",   type=str,   default='label.npz',                    help='path of the image')
    parser.add_argument("--data-basedir", dest="data_basedir", type=str,   default='/home/../../public/zyf/What3D',help='path of the data folder') 
    parser.add_argument("--splits-path",  dest="splits_path",  type=str,   default='splits',help='path of the data folder') 
    parser.add_argument("--class-path",   dest="class_path",   type=str,   default='classes.txt',help="class name list")

    # Log file info
    parser.add_argument("--log-dir",dest="log_dir", type=str, default="logs/",help="log folder to save training stats as numpy files")
    parser.add_argument('--verbose_per_n_batch',dest="verbose_per_n_batch", type=int,default= 5,help='log training stats to console every n batch (<=0 disables training log)')
    parser.add_argument('--test_per_n_epoch',dest="test_per_n_epoch", type=int,default= 1,help='test per epoch')                
    parser.add_argument("--tensorboard", action="store_true",dest="tensorboard",default= None) 
    parser.add_argument("--save-results", action = "store_true",dest="save_results",default = False, help = "save the results in evaluation test")
    parser.add_argument("--checkpoint-model", type=str,dest="checkpoint_model",default= None,help="Path to pretrained model G net")
    parser.add_argument("--checkpoint-solver", type=str,dest="checkpoint_solver",default= None,help="Path to pretrained solver")
    
    parser.add_argument("--if-BNstats", action="store_true", dest="if_BNstats", default=False, help="if calculate bn stats")
    parser.add_argument("--bnstats-step", type=int, dest="bnstats_step", default=30, help="step to log batch norm stats")
    
    #parser.add_argument("--random_rotation", action="store_true", help="apply data augmentation : random rotation")
    #parser.add_argument("--data_augmentation_axis_rotation", action="store_true",
    #                    help="apply data augmentation : axial rotation ")
    #parser.add_argument("--data_augmentation_random_flips", action="store_true",
    #                    help="apply data augmentation : random flips")
    #parser.add_argument("--random_translation", action="store_true",
    #                    help="apply data augmentation :  random translation ")
    #parser.add_argument("--anisotropic_scaling", action="store_true",
    #                    help="apply data augmentation : anisotropic scaling")
    #parser.add_argument("--sample", action="store_false", help="Sample the input pointclouds")
    #parser.add_argument('--class_choice', nargs='+', default=["airplane"], type=str)
    # Training parameters
    #parser.add_argument("--no_learning", action="store_true", help="Learning mode (batchnorms...)")
    #parser.add_argument("--train_only_encoder", action="store_true", help="only train the encoder")
    #parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    #parser.add_argument('--batch_size_test', type=int, default=32, help='input batch size')
    #parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    #parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
    #parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to train for')
    #parser.add_argument("--random_seed", action="store_true", help="Fix random seed or not")
    #parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
    #parser.add_argument('--lr_decay_1', type=int, default=120, help='learning rate decay 1')
    #parser.add_argument('--lr_decay_2', type=int, default=140, help='learning rate decay 2')
    #parser.add_argument('--lr_decay_3', type=int, default=145, help='learning rate decay 2')
    #parser.add_argument("--run_single_eval", action="store_true", help="evaluate a trained network")
    #parser.add_argument("--demo", action="store_true", help="run demo autoencoder or single-view")
    # Data
    #parser.add_argument('--normalization', type=str, default="UnitBall",
    #                    choices=['UnitBall', 'BoundingBox', 'Identity'])
    #parser.add_argument("--shapenet13", action="store_true", help="Load 13 usual shapenet categories")

    args = parser.parse_args()
    args.date = str(datetime.datetime.now())
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now = datetime.datetime.now()
    args = EasyDict(args.__dict__)

    # Hard code dimension of the template.
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    args.dim_template = dim_template_dict[args.template_type]


    return args

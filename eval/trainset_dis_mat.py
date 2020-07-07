"""
Author: Yefan Zhou
Purpose: compute block distance matrix of training set about 35000 instances
"""
import os
import sys
import argparse
import torch
import tqdm
import numpy as np
import random
import cv2
sys.path.append("../")
from utils.loss import ChamfersDistance3
from compute_testset_distance_matrix import Block_compute_ptcloud_dismatrix
import torch.nn as nn
import time
import logging
from pathlib import Path
#torch.backends.cudnn.enabled = True
#torch.backends.cudnn.benchmark = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(args):
    # Load the training dataset
    ptcloud_set = np.load(args.ptcloud_path)
    train_ptcloud_set = ptcloud_set['train']
    # Convert numpy array to torch tensor
    train_ptcloud_set = torch.from_numpy(train_ptcloud_set).to(args.device)

    # define the distance between point cloud 
    if args.distance_matrix == 'CD':
        pt_criterion = ChamfersDistance3().to(args.device)
    elif args.distance_matrix == 'EMD':
        pass
    # define the partial matrix save path
    partial_DM_path = os.path.join(args.matrix_save_path, args.experiment_name)
    Path(partial_DM_path).mkdir(parents=True, exist_ok=True)
    # compute the block distance matrix and save 
    train_pt_matrix = Block_compute_ptcloud_dismatrix(X1=train_ptcloud_set, X2=train_ptcloud_set,
                                                     distance_metric=pt_criterion,
                                                     start_index=args.start_index,
                                                     end_index=args.end_index,
                                                     title='trainset_mat_{}_{}_{}.npy'.format(
                                                     args.start_index,
                                                     args.end_index,
                                                     args.experiment_name),
                                                     results_dir=partial_DM_path,
                                                     ifsave=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--data-basedir', type=str, default='/home/yirus/Datasets/what3d/',
                        help='')
    parser.add_argument('--img-path', type=str, default='renderings',
                        help='')
    parser.add_argument('--splits-path', type=str, default='splits',
                        help='')
    parser.add_argument('--class-path', type=str, default='classes.txt',
                        help='')
    parser.add_argument('--clusterfile-path', type=str,
                        default='../what3d_clusters/cluster_assignments/',
                        help='')
    parser.add_argument('--prediction-path', type=str,
                        default='../what3d_clusters/predictions.txt',
                        help='')
    parser.add_argument("--ptcloud-path", type=str,
                        default="/home/zyf/What3D/ptcloud_0.npz",
                        help=' ')
    parser.add_argument("--matrix-save-path", type=str,
                        default="distance_matrix",
                        help=' ')
    parser.add_argument("--experiment-name", type=str,
                        default='GT_partial',
                        help=' ')
    parser.add_argument("--start_index", type=int,
                        help=' ')
    parser.add_argument("--end_index", type=int,
                        help=' ')
    parser.add_argument('--distance_matrix', type=str,
                        default='CD', 
                        help='option: ["CD","EMD"]')
    parser.add_argument('--selection_list', type=str,
                        default=None,
                        help='')
    parser.add_argument('--i_start', type=int,
                        default=0,
                        help='i_iter to start')
    parser.add_argument('--i_end', type=int,
                        default=10,
                        help='i_iter to end')
    parser.add_argument('--gpu_id', type=str,
                        default="0",
                        help='GPU ID to set the gpu to use')


    args = parser.parse_args(sys.argv[1:])
    cuda_cmd = "cuda:" + args.gpu_id
    args.device = torch.device(cuda_cmd if torch.cuda.is_available() else "cpu")
    args.script_folder = os.path.dirname(os.path.abspath(__file__))
    print(args)
    main(args)
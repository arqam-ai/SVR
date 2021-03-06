"""
Author: Yiru Shen, Yefan Zhou
Purpose: compute block distance matrix and calculate the SScore
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
from model.clustering import class_counter
from utils.ptcloud_dataset_graph_three_shapes_yefan import image_preprocessing, compute_squared_EDM_method, \
    transform_mat, cal_pref, get_partition, silhouette
from utils.utils import check_exist_or_mkdirs
from utils.loss import ChamferDistance
import torch.nn as nn
import time
import logging

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def compute_img_dismatrix(X1, X2, distance_metric, title=None, results_dir=None, ifsave=False):
    """return distance matrix between img set X1 and img set X2
    Params:
    ----------------------------------
    X1: (N, 3, img_size, img_size) torch.tensor
        img set 1
    X2: (N, 3, img_size, img_size) torch.tensor
        img set 2
    distance_metric: func
        metric to measure the distance of two image
    ifsave: boolean
        if to save the distance matrix to disk
    title: string
        name of the saved matrix
    results_dir:string
        the path to save the distance matrix

    Returns:
    ----------------------------------
    D: (ptnum, ptnum) torch.tensor
        distance matrix
    """
    N = X1.shape[0]
    # initialize distance matrix
    D = torch.zeros([N, N])
    # iterate over one group of ptcloud
    for i in tqdm.tqdm(range(N), total=N, desc='image distance matrix'):
        for j in range(i + 1, N):
            D[i, j] = distance_metric(X1[i].unsqueeze(0), X2[j].unsqueeze(0))
            D[j, i] = D[i, j]
    if ifsave:
        D = D.cpu().numpy()
        np.save(os.path.join(results_dir, title), D)
        print("saved to " + os.path.join(results_dir, title))
    return D


def compute_ptcloud_dismatrix(X1, X2, distance_metric, title=None, results_dir=None, ifsave=False):
    """return distance matrix between ptcloud X1 ptcloud X2
    Params:
    ----------------------------------
    X1: (N, ptnum,3) torch.tensor
        point cloud set 1
    X2: (N, ptnum,3) torch.tensor
        point cloud set 2
    distance_metric: func
        metric to measure the distance of two point cloud
    ifsave: boolean
        if to save the distance matrix to disk
    title: string
        name of the saved matrix
    results_dir:string
        the path to save the distance matrix

    Returns:
    ----------------------------------
    D: (ptnum, ptnum) torch.tensor
        distance matrix
    """
    N = X1.shape[0]
    # initialize distance matrix
    D = torch.zeros([N, N])
    # iterate over one group of ptcloud
    for i in range(N):
        if i % 10 == 0:
            print("Processing {} ...........................................".format(i))
        for j in range(i + 1, N):
            D[i, j] = distance_metric(X1[i].unsqueeze(0), X2[j].unsqueeze(0))
            D[j, i] = D[i, j]
    if ifsave:
        D = D.cpu().numpy()
        print("Saving distance matrix ............................................")
        np.save(os.path.join(results_dir, title), D)
        print("saved to " + os.path.join(results_dir, title))
    return D


def Block_compute_codeword_dismatrix(X1, X2, distance_metric, start_index, end_index, title=None, results_dir=None, ifsave=False):
    """ return distance matrix of X1 
    Params:
    ----------
    X1: (N, 512) torch.tensor
        codeword
    """
    N = X1.shape[0]
    # initialize distance matrix
    D = torch.zeros([N, N])
    # iterate over one group of ptcloud
    for i in range(start_index, end_index):
        if i % 10 == 0:
            print("Processing {} ...........................................".format(i))
        for j in range(i+1, N):
            D[i, j] = distance_metric(X1[i], X2[j])
    if ifsave:
        D = D.cpu().numpy()
        print("Saving distance matrix ............................................")
        np.save(os.path.join(results_dir, title), D)
        print("saved to " + os.path.join(results_dir, title))
    return D

def Block_compute_ptcloud_dismatrix(X1, X2, distance_metric, start_index, end_index, title=None, results_dir=None, ifsave=False):
    """return distance matrix between ptcloud X1 ptcloud X2
    Params:
    ----------------------------------
    X1: (N, ptnum,3) torch.tensor
        point cloud set 1
    X2: (N, ptnum,3) torch.tensor
        point cloud set 2
    distance_metric: func
        metric to measure the distance of two point cloud
    start_index: int
        index to start the block matrix computing 
    end_index:  int 
        index to end the block matrix computing 
    ifsave: boolean
        if to save the distance matrix to disk
    title: string
        name of the saved matrix
    results_dir: string
        the path to save the distance matrix

    Returns:
    -------------------------#---------
    D: (ptnum, ptnum) torch.tensor
        distance matrix
    """
    N = X1.shape[0]
    # initialize distance matrix
    D = torch.zeros([N, N])
    # iterate over one group of ptcloud
    for i in range(start_index, end_index):
        for j in range(i+1, N):
            D[i, j] = distance_metric(X1[i].unsqueeze(0), X2[j].unsqueeze(0))
    if ifsave:
        D = D.cpu().numpy()
        np.save(os.path.join(results_dir, title), D)
        print("saved to " + os.path.join(results_dir, title))
    return D

def main(args):
    ## set up logger
    logger = logging.getLogger()
    file_log_handler = logging.FileHandler('Train_%s.log' % (args.experiment_name))
    logger.addHandler(file_log_handler)
    logger.setLevel('INFO')
    formatter = logging.Formatter()
    file_log_handler.setFormatter(formatter)
    stderr_log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stderr_log_handler)
    stderr_log_handler.setFormatter(formatter)

    check_exist_or_mkdirs(args.matrix_save_path)
    starter_time = time.time()
    # ## LOAD class info
    # train_num, train_class_dic, train_class_list, train_fileid_list = class_counter(data_basedir=args.data_basedir,
    #                                                                                 splits_path=args.splits_path,
    #                                                                                 class_path=args.class_path,
    #                                                                                 split_name='train')
    # class_list = list(train_class_dic.keys())  ## list of class name length = 55

    ## LOAD train point cloud
    train_ptcloud_set = np.load(args.ptcloud_path)
    print(train_ptcloud_set.shape)

    ## Convert numpy array to torch tensor
    train_ptcloud_set = torch.from_numpy(train_ptcloud_set).to(args.device)

    base_filename = "distance_matrix/testset_iter_"

    for i_iter in range(args.i_start, args.i_end):
        print("Processing {} .........................".format(i_iter))
        # ##randomly sample 10% instances from train set
        # ### generate random sample index list
        num_samples = 484
        # index = np.arange(train_ptcloud_set.shape[0])
        # np.random.shuffle(index)
        # sample_index_list = index[0:num_samples]
        # filename = os.path.join(args.matrix_save_path, "testset_iter_{}.npy".format(i_iter))
        # np.save(filename, sample_index_list)
        sample_index_filename = base_filename+str(i_iter)+".npy"
        sample_index_list = np.load(sample_index_filename)

        ### slice the all set
        sample_train_ptcloud_set = train_ptcloud_set[sample_index_list].to(args.device)
        print("#sampls is {}".format(sample_train_ptcloud_set.shape[0]))
        print(sample_train_ptcloud_set.shape)

        ## compute distance matrix
        pt_criterion = ChamfersDistance3().to(args.device)
        train_pt_matrix = compute_ptcloud_dismatrix(
            X1=sample_train_ptcloud_set,
            X2=sample_train_ptcloud_set,
            distance_metric=pt_criterion,
            title='pred__{}_DM_iter__{}.npy'.format(num_samples, i_iter),
            results_dir=args.matrix_save_path,
            ifsave=True
        )

        # train_pt_matrix = Block_compute_ptcloud_dismatrix(X1=train_ptcloud_set, X2=train_ptcloud_set,
        #                                             distance_metric=pt_criterion,
        #                                             start_index=args.start_index,
        #                                             end_index=args.end_index,
        #                                             title='testset_similarity_matrix_{}_{}_{}.npy'.format(
        #                                                 args.start_index,
        #                                                 args.end_index,
        #                                                 args.experiment_name),
        #                                             results_dir=args.matrix_save_path,
        #                                             ifsave=True)

        # train_pt_matrix = compute_ptcloud_dismatrix(X1=train_ptcloud_set, X2=train_ptcloud_set,
        #                                                   distance_metric=pt_criterion,
        #                                                   title='clustering_DM_{}.npy'.format(
        #                                                       args.experiment_name),
        #                                                   results_dir=args.matrix_save_path,
        #                                                   ifsave=True)

        # ####### compute distance matrix from NPS/FPS #######
        # if args.selection_list is not None:
        #     train_ptcloud_set = np.load(args.ptcloud_path)
        #     subsample_list = np.load(args.selection_list)
        #     subsample_type = args.selection_list.replace(".npy", "").split("_")[0].split("/")[1]
        #     start_index = args.selection_list.replace(".npy", "").split("_")[-1]
        #     subset_ptc = train_ptcloud_set[subsample_list,:]
        #     subset_ptc = torch.from_numpy(subset_ptc).to(args.device)
        #     train_pt_matrix = compute_ptcloud_dismatrix(
        #         X1=subset_ptc, X2=subset_ptc,
        #         distance_metric=pt_criterion,
        #         title='distance_matrix_sample_{}__{}_starIndex_{}.npy'.format(
        #         subsample_type, len(subsample_list), start_index),
        #         results_dir=args.matrix_save_path,
        #         ifsave=True)
        # else:
        #     #### read GT/Pred complete distance matrix ####
        #     train_pt_matrix = np.load(args.distance_matrix)
        #     print("Done loading distance matrix: {}".format(train_pt_matrix.shape))
        #     #### read GT/Pred complete distance matrix ####
        # ####### compute distance matrix from NPS/FPS #######

        #
        # ## normalize matrix
        train_pt_matrix_tr = transform_mat(train_pt_matrix)  # -e^(x/max(x)) then fill the diagonal with 0
        # train_img_matrix_tr = transform_mat(train_img_matrix)
        #
        # ## get partition
        ### point cloud, calculate affinity propagation parameter: preference
        part_preference = cal_pref(train_pt_matrix_tr)  # in increasing order, float number in first 10% position in this matrix
        # ### affinity propagation
        train_pt_part = get_partition(train_pt_matrix_tr, preference=part_preference, damping=0.75)

        #
        ## silhouette score
        pt_ss = silhouette(train_pt_matrix, train_pt_part)
        # # img_ss = silhouette(train_img_matrix, train_img_part)
        #
        # ## report
        logger.info('Experiment No.{} point cloud silhouette: {}'.format(args.experiment_name, pt_ss))
        # # logger.info('Experiment No.{} img silhouette: {}'.format(args.experiment_name, img_ss))
        # logger.info("Start index: {}, end index: {}".format(args.start_index, args.end_index))
        # logger.info('Time:{:3} seconds'.format(time.time() - starter_time))


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
                        default="/home/yirus/Datasets/what3d/ptcloud_0.npz",
                        help=' ')
    parser.add_argument("--matrix-save-path", type=str,
                        default="distance_matrix",
                        help=' ')
    parser.add_argument("--experiment-name", type=str,
                        default='trainsetAll',
                        help=' ')
    parser.add_argument("--start_index", type=int,
                        help=' ')
    parser.add_argument("--end_index", type=int,
                        help=' ')
    parser.add_argument('--distance_matrix', type=str,
                        default='',
                        help='')
    parser.add_argument('--selection_list', type=str,
                        default=None,
                        help='')
    parser.add_argument('--i_start', type=int,
                        default=0,
                        help='i_iter to start')
    parser.add_argument('--i_end', type=int,
                        default=10,
                        help='i_iter to end')


    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    print(str(args))
    main(args)



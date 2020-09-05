'''
    this is script used to measure SScore on Clustering method matrix

    Author: yefan
'''
import os
import sys
sys.path.append("../")
import numpy as np
from model.clustering import load_prediction
from utils.ptcloud_dataset_graph_three_shapes_yefan import image_preprocessing, compute_squared_EDM_method, \
												transform_mat, cal_pref, get_partition, silhouette
import time
import random


def generate_cluster_prediction(cluster_dismatrix, cluster_assignment, num_of_prediction):
    """generate Clustering method prediction
    Params:
    ------------------
    cluster_dismatrix: (num_of_cluster, num_of_cluster) numpy array
        distance matrix between cluster mean shape
    cluster_assignment: python list
        list of cluster index assignment for inferred results
    num_of_prediction: int
        number of prediction should be inferred(or the size of test set)

    Returns:
    ------------------
    pred_dismatrix: (num_of_prediction, num_of_prediction) numpy array
        prediction distance matrix
    """
    ## initialize an empty matrix
    pred_dismatrix = np.zeros((num_of_prediction, num_of_prediction))
    ## iterize through the distance matrix 
    for i in range(num_of_prediction):
        for j in range(i+1, num_of_prediction):
            ## retrieve cluster index, use cluster index to retrieve the distance value
            pred_dismatrix[i, j] = cluster_dismatrix[cluster_assignment[i], cluster_assignment[j]]
            pred_dismatrix[j, i] = pred_dismatrix[i, j]
    return pred_dismatrix

def silhouette_score(distance_matrix):
    """calculate silhouette score based on precomputed distance matrix
    Params:
    ------------------
    distance_matrix: (N, N) numpy array
        distance matrix to be calcuated 
    Returns:
    ------------------
    silhouette_score: float

    """
    starter_time = time.time()
    ## normalize matrix   
    distance_matrix_tr = transform_mat(distance_matrix)     # -e^(x/max(x)) then fill the diagonal with 0
    #print("normalize done Time : {:3f} seconds".format(time.time()-starter_time))
    ## get partition
    ### calculate affinity propagation parameter: preference
    part_preference = cal_pref(distance_matrix_tr)          # in increasing order, float number in first 10% position in this matrix
    ### affinity propagation
    matrix_part = get_partition(distance_matrix_tr, preference = part_preference)
#    print("affinity propagation label number:  ", len(set(matrix_part)))
    #print("paritition done Time : {:3f} seconds".format(time.time()-starter_time))
    ## silhouette score
    ss = silhouette(distance_matrix, matrix_part)
    #print("silhouette done Time : {:3f} seconds".format(time.time()-starter_time))
    return ss

def dismatrix_sample(matrix, indexlist):
    """downsample a distance matrix based on indexlist 
    Params:
    ------------------
    matrix   : (N,N) numpy array
        matrix to be downsampled
    indexlist: python list
        list to contain sample index
    
    Returns:
    ------------------
    M : (N,N) numpy array
        N = length of indexlist 
    """
    
    ## initialize an empty matrix
    N = len(indexlist) 
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            M[i,j] = matrix[indexlist[i], indexlist[j]]
            M[j,i] = M[i,j]
    return M

def main():
    ## Randomly sample 2000 instances from test set index 
    cluster_ss_sum = 0
    gt_ss_sum = 0
    auto_ss_sum = 0
    
    
    for idx in range(10):
        selection_list = random.sample(range(10432), 2000)
#        np.save('subsample/downsample_all/random_2000_index_exp_{}.npy'.format(idx), np.array(selection_list))
        #print(selection_list)
        ###################################################################
        ### Clustering Method
        ## LOAD list of cluster index assignment for inferred results
#        cluster_assignment = load_prediction('../what3d_clusters/predictions.txt')
        ## LOAD distance matrix of cluster mean shape
#        cluster_dismatrix = np.load('clustering_baseline/clustering_DM_trainsetAll.npy')
        ## generate prediction distance matrix based on cluster distance matrix
#        cluster_pred_dismatrix = generate_cluster_prediction(cluster_dismatrix = cluster_dismatrix, cluster_assignment = cluster_assignment,
#                                                num_of_prediction = 10432)
        cluster_pred_dismatrix = np.load('distance_matrix/Cluster_all_distance_matrix.npy')
        ## sample cluster distance matrix from 10432 x 10432 to 2000 x 2000
        sampled_cluster_pred_dismatrix = dismatrix_sample(cluster_pred_dismatrix, selection_list)
        ## calculate silhouette score
        ss = silhouette_score(sampled_cluster_pred_dismatrix)
        cluster_ss_sum += ss
        print("Clustering Method silhouette_score is {:3f}".format(ss))
    
        ###################################################################
        ### Ground truth and auto encoder prediction
        ## LOAD distance matrix of GT 
        gt_dismatrix = np.load('distance_matrix/GT_all_distance_matrix.npy')
        ## LOAD distance matrix of prediction
        auto_pred_dismatrix = np.load('distance_matrix/Pred_all_distance_matrix.npy')
        ## sample gt distance matrix from 10432 x 10432 to 2000 x 2000
        sampled_gt_dismatrix = dismatrix_sample(gt_dismatrix, selection_list)
        ## sample autoencoder distance matrix from 10432 x 10432 to 2000 x 2000
        sampled_auto_pred_dismatrix = dismatrix_sample(auto_pred_dismatrix, selection_list)
        ## silhouette score
        gt_ss = silhouette_score(sampled_gt_dismatrix)
        gt_ss_sum += gt_ss
        print("GT silhouette_score is {:3f}".format(gt_ss))
        ## silhouette score
        auto_ss = silhouette_score(sampled_auto_pred_dismatrix)
        auto_ss_sum += auto_ss
        print("autoencoder silhouette_score is {:3f}".format(auto_ss))
    
    print("Mean of Clustering Method silhouette_score for ten times is {:3f}".format(cluster_ss_sum/10))
    print("Mean of GT silhouette_score for ten times is {:3f}".format(gt_ss_sum/10))
    print("Mean of autoencoder silhouette_score for ten times is {:3f}".format(auto_ss_sum/10))    
    
    ##############################################################
    '''
    selection_list = list(np.load('subsample/downsample_all/random_2000_index_exp_5.npy'))
    cluster_assignment = load_prediction('../what3d_clusters/predictions.txt')
    cluster_index = [cluster_assignment[item] for item in selection_list]
    cluster_stats = {}
    for index in cluster_index:
        if index not in cluster_stats:
            cluster_stats[index] = 1
        else:
            cluster_stats[index] += 1
    cluster_stats = {k: v for k, v in sorted(cluster_stats.items(), key=lambda item: item[1])}
    counter = 0
    for item in cluster_stats:
        if cluster_stats[item] > 20:
            counter += 1
    cluster_pred_dismatrix = np.load('distance_matrix/Cluster_all_distance_matrix.npy')
    sampled_cluster_pred_dismatrix = dismatrix_sample(cluster_pred_dismatrix, selection_list)
    print("Clustering Method silhouette_score is {:3f}".format(silhouette_score(sampled_cluster_pred_dismatrix)))
    '''

if __name__ == '__main__':
    main()

import os
import sys
sys.path.append("../")
import numpy as np
from model.clustering import load_prediction
from utils.ptcloud_dataset_graph_three_shapes_yefan import image_preprocessing, compute_squared_EDM_method, \
												transform_mat, cal_pref, get_partition, silhouette
import time

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
    print("normalize done Time : {:3f} seconds".format(time.time()-starter_time))
    ## get partition
    ### calculate affinity propagation parameter: preference
    part_preference = cal_pref(distance_matrix_tr)          # in increasing order, float number in first 10% position in this matrix
    ### affinity propagation
    matrix_part = get_partition(distance_matrix_tr, preference = part_preference)
    print("paritition done Time : {:3f} seconds".format(time.time()-starter_time))
    ## silhouette score
    ss = silhouette(distance_matrix, matrix_part)
    print("silhouette done Time : {:3f} seconds".format(time.time()-starter_time))
    return ss
    

def main():
    ## LOAD list of cluster index assignment for inferred results
    cluster_assignment = load_prediction('../what3d_clusters/predictions.txt')
    ## LOAD distance matrix between cluster mean shape
    cluster_dismatrix = np.load('clustering_baseline/clustering_DM_trainsetAll.npy')
    ## generate prediction distance matrix based on cluster distance matrix
    pred_dismatrix = generate_cluster_prediction(cluster_dismatrix = cluster_dismatrix, cluster_assignment = cluster_assignment,
                                            num_of_prediction = 10432)
    ## calculate silhouette score
    ss = silhouette_score(pred_dismatrix)
    print("silhouette_score is {:3f}".format(ss))
    


if __name__ == '__main__':
    main()

"""
Author: Yefan (adapted from Yiru)
Purpose: FPS and NPS from distance matrix
"""
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import collections
from compute_prediction import silhouette_score
import tqdm
import logging
import sys
'''
def farthest_selection_yefan(distance_matrix, num_samples, start_index):
    n = distance_matrix.shape[0]
    distance_matrix[range(n), range(n)] = -1
    seen = list()
    seen.append(start_index)
    current_index = start_index
    cur_patch = 1

    farthest_distance = []
    while True:
        if cur_patch >= num_samples:
            break
        select_idx = np.argmax(distance_matrix[current_index,:])
        if select_idx not in seen:
            seen.append(select_idx)
            farthest_distance.append(distance_matrix[current_index, select_idx])
            cur_patch += 1
            distance_matrix[current_index, select_idx] = -1
            distance_matrix[select_idx, current_index] = -1
            current_index = select_idx
            continue
        else:
            distance_matrix[current_index, select_idx] = -1
    
    return seen, farthest_distance

def nearest_selection_yefan(distance_matrix, num_samples, start_index):
    n = distance_matrix.shape[0]
    distance_matrix[range(n), range(n)] = 9999
    seen = list()
    seen.append(start_index)
    current_index = start_index
    cur_patch = 1

    farthest_distance = []
    while True:
        if cur_patch >= num_samples:
            break
        select_idx = np.argmin(distance_matrix[current_index,:])
        if select_idx not in seen:
            seen.append(select_idx)
            farthest_distance.append(distance_matrix[current_index, select_idx])
            cur_patch += 1
            distance_matrix[current_index, select_idx] = 9999
            distance_matrix[select_idx, current_index] = 9999
            current_index = select_idx
            continue
        else:
            distance_matrix[current_index, select_idx] = 9999
    
    return seen, farthest_distance
'''


def farthest_selection(distance_matrix, num_samples, start_index):
    ### replace diagnal distance to a large number ###
    n = distance_matrix.shape[0]
    distance_matrix[range(n), range(n)] = 9999

    threshold = 0.7
    seen = list()
    seen.append(start_index)
    current_index = start_index
    cur_patch = 1

    farthest_distance = []

    while True:
        if cur_patch >= num_samples:
            break
        # select_idx = np.argmin(distance_matrix[current_index,:])
        relax_idx = np.where((abs(distance_matrix[current_index,:]!=9999) & (abs(distance_matrix[current_index,:])>=threshold) ))[0]
        if len(relax_idx) == 0:
            th = threshold
            while True:
                th = th - 0.05
                relax_idx = np.where((abs(distance_matrix[current_index,:]!=9999) & (abs(distance_matrix[current_index,:])>=th) ))[0]
                if len(relax_idx) > 0:
                    break
        select_idx = random.choice(relax_idx)
        if select_idx not in seen:
            # seen.add(select_idx)
            seen.append(select_idx)
            # print("{} --> {}, distance: {}, {}".format(current_index, select_idx, distance_matrix[current_index, select_idx], cur_patch))
            farthest_distance.append(distance_matrix[current_index, select_idx])
            cur_patch += 1
            distance_matrix[current_index, select_idx] = 9999
            current_index = select_idx
            continue
        else:
            distance_matrix[current_index, select_idx] = 9999

    return seen, farthest_distance


def nearest_selection(distance_matrix, num_samples, start_index):
    ### replace diagnal distance to a large number ###
    n = distance_matrix.shape[0]
    distance_matrix[range(n), range(n)] = 9999

    threshold = 0.05
    seen = list()
    seen.append(start_index)
    current_index = start_index
    cur_patch = 1

    nearest_distance = []

    while True:
        if cur_patch >= num_samples:
            break

        relax_idx = np.where((abs(distance_matrix[current_index,:]!=9999) & (abs(distance_matrix[current_index,:])<=threshold) ))[0]
        if len(relax_idx) == 0:
            th = threshold
            while True:
                th = th + 0.05
                relax_idx = np.where((abs(distance_matrix[current_index,:]!=9999) & (abs(distance_matrix[current_index,:])<=th) ))[0]
                if len(relax_idx) > 0:
                    break
        select_idx = random.choice(relax_idx)
        if select_idx not in seen:
            seen.append(select_idx)
            # print("{} --> {}, distance: {}, {}".format(current_index, select_idx, distance_matrix[current_index, select_idx], cur_patch))
            nearest_distance.append(distance_matrix[current_index, select_idx])
            cur_patch += 1
            distance_matrix[current_index, select_idx] = 9999
            current_index = select_idx
            continue
        else:
            distance_matrix[current_index, select_idx] = 9999

    return seen, nearest_distance

def retrieve_distance(matrix, i, j):
    """return distance value by retrieving based on index i and j 

    Params:
    ------------------
    matrix : (N, N) numpy array 
        distance matrix to be retrieved
    i : int
        row index
    j : int
        col index
    Returns:
    ------------------
    distance: distance value
    """
    distance = matrix[i, j]
    return distance

def farthest_selection_yefan(distance_matrix,  num_samples , total_num):
    """search farthest points set in test set
    Params:
    ------------------
    distance_matrix : (N, N) numpy array
        distance matrix between all the instances
    num_samples : int
        number of sample desired
    total_num : int
        total number of remaining solution set 

    Returns:
    ------------------
    solution_set : list
        sampled index based on algorithm
    """
    ## initiaize the remaining set and soluation set 
    remaining_points = list(range(total_num))
    solution_set = []
    ## randomly select a start point 
    solution_set.append(remaining_points.pop(\
                            random.randint(0, len(remaining_points)-1)))   # random.randint return random integer N (a <= N <= b)
    print('enter')
    for _ in tqdm.tqdm(range(num_samples-1), total=num_samples-1):
        ## initialize an empty matrix to hold distance (rows = solution_set  cols = remaining_points) 
        distances = np.zeros((len(solution_set), len(remaining_points)))
        ## loop through every appended solution
        for i, s in enumerate(solution_set):
            ## get all distance between one solution and all remaining_points 
            distances[i] = distance_matrix[s, remaining_points]
        ## for one remaining points, find the minimum distance between it and all solution                
        mindis_in_solution = np.min(distances, axis = 0)
        assert mindis_in_solution.shape[0] == len(remaining_points), 'length of remaining points mismatch'
        ## find the farthest points based on all items in solution set 
        solution_set.append(remaining_points.pop(np.argmax(mindis_in_solution)))
    return solution_set

def nearest_selection_yefan(distance_matrix,  num_samples , total_num):
    """search nearest points set in test set
    Params:
    ------------------
    distance_matrix : (N, N) numpy array
        distance matrix between all the instances
    num_samples : int
        number of sample desired
    total_num : int
        total number of remaining solution set 

    Returns:
    ------------------
    solution_set : list
        sampled index based on algorithm
    """
    ## initiaize the remaining set and soluation set 
    remaining_points = list(range(total_num))
    solution_set = []
    ## randomly select a start point 
    solution_set.append(remaining_points.pop(\
                            random.randint(0, len(remaining_points)-1)))   # random.randint return random integer N (a <= N <= b)
    
    for _ in range(num_samples-1):
        ## initialize an empty matrix to hold distance (rows = solution_set  cols = remaining_points) 
        distances = np.zeros((len(solution_set), len(remaining_points)))
        ## loop through every appended solution
        for i, s in enumerate(solution_set):
            ## get all distance between one solution and all remaining_points 
            distances[i] = distance_matrix[s, remaining_points]
        ## for one remaining points, find the minimum distance between it and all solution                
        mindis_in_solution = np.min(distances, axis = 0)
        #!!! mindis_in_solution = np.max(distances, axis = 0)
        assert mindis_in_solution.shape[0] == len(remaining_points), 'length of remaining points mismatch'
        ## find the nearest points based on all items in solution set 
        solution_set.append(remaining_points.pop(np.argmin(mindis_in_solution)))
    return solution_set

def nearest_selection_fps500(distance_matrix,  num_samples , total_num, cluster_num = 500):
    """search nearest points set in test set
    Params:
    ------------------
    distance_matrix : (N, N) numpy array
        distance matrix between all the instances
    num_samples : int
        number of sample desired
    total_num : int
        total number of remaining solution set 
    cluster_num: int
        number of cluster need to be searched by fps
    Returns:
    ------------------
    solution_set : list
        sampled index based on algorithm
    """
    ## number of instance around cluster i.e. 4
    num_around_cluster = int(num_samples/cluster_num)
    ## initiaize solution set 
    solution_set = [] 
    ## use Farthest Point Sampling to sample cluster 
    cluster_list = farthest_selection_yefan(distance_matrix, num_samples = cluster_num, total_num = total_num)
    ## check if cluster index overlap
    assert len(set(cluster_list)) == cluster_num, 'cluster index overlap detected'
    ## loop over all cluster index
    for cluster in cluster_list:
        length_of_solution = len(solution_set)
        ## sort the distance vector
        ordered_index = distance_matrix[cluster].argsort()
        ## start from first nearest neighbor, if i = 0 then start from itself
        i = 1
        while True:
            ## if specific number of instance sampled around cluster, stop
            if len(solution_set) >= length_of_solution + num_around_cluster:
                break
            ## retrieve a neighbor index
            index = ordered_index[i]
            i = i + 1
            ## check if candidate already in solution set 
            if index not in solution_set:
                solution_set.append(index)
    ## check solution set overlap
    assert len(set(solution_set)) == num_samples
    return solution_set

def visualize_selection(ptc, selection_list, filename, num_to_vis=100):
    fig = plt.figure(figsize=(10, 10))
    for i in range(num_to_vis):
        ax = fig.add_subplot(10, 10, i+1, projection='3d')
        p = ptc[selection_list[i],:]
        ax.scatter(p[:,0], p[:,1], p[:,2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # if i < 10:
        #     name = test_listname[selection_list[i]]
        #     class_type = name.split("/")[0]
        #     print(selection_list[i], nearest_distance[i], class_type)
        ax.set_title("{}".format(selection_list[i]))
        # if i>0:
        #     ax.set_title("{:.2f}".format(nearest_distance[i-1]))
        # if i==0:
        #     ax.set_title("{}".format(selection_list[i]))
        # else:
        #     ax.set_title("{}, {:.2f}".format(selection_list[i], nearest_distance[i-1]))

    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()

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
    ## DEFINE sample type
    subsample_type = "NPS_500"
    ## DEFINE sample number 
    num_samples = 2000
    
    ## set up logger
    '''
    logger = logging.getLogger()
    file_log_handler = logging.FileHandler('{}_{}_ss.log'.format(subsample_type, num_samples))
    logger.addHandler(file_log_handler)
    logger.setLevel('INFO')
    formatter = logging.Formatter()
    file_log_handler.setFormatter(formatter)
    stderr_log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stderr_log_handler)
    stderr_log_handler.setFormatter(formatter)
    '''
    ## LOAD distance matrix
    distance_matrix = np.load("distance_matrix/GT_all_distance_matrix.npy")
    ## MAKE folder
    folder_for_subsample = "subsample"
    if not os.path.isdir(folder_for_subsample):
        os.mkdir(folder_for_subsample)
    dst_folder = os.path.join(folder_for_subsample, subsample_type)
    if not os.path.isdir(dst_folder):
        os.mkdir(dst_folder)

    ## Loop over ten experiments 
#    logger.info('Experiment Type: {}'.format(subsample_type))
    for step in range(10):
        
        if subsample_type == "FPS":
            selection_list = farthest_selection_yefan(distance_matrix= np.copy(distance_matrix), num_samples=num_samples, 
                                                                                       total_num = 10432)
        elif subsample_type == "NPS":
            selection_list = nearest_selection_yefan(distance_matrix= np.copy(distance_matrix),num_samples=num_samples, 
                                                                                        total_num = 10432)
        elif subsample_type == "random":
            selection_list = random.sample(range(10432), num_samples)
        
        elif subsample_type == "NPS_500":
            selection_list = nearest_selection_fps500(distance_matrix= np.copy(distance_matrix),num_samples=num_samples, 
                                                                                        total_num = 10432, cluster_num = 500)
        
        ## generate distance cmatrix based on total dataset distance matrix 
        sampled_matrix = dismatrix_sample(distance_matrix, selection_list)
        print('Sumple type {}, sample number {}, Sscore is {}'.format(subsample_type, num_samples, silhouette_score(sampled_matrix)))
        ## save the sampling index as npy file
#        np.save(os.path.join(dst_folder, '{}_{}_index_{}.npy'.format(subsample_type, num_samples, step)), np.array(selection_list))
#        logger.info('{} silhouette score is {}'.format(subsample_type, silhouette_score(sampled_matrix)))
        
if __name__ == "__main__":
    main()
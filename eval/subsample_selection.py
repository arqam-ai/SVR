"""
Author: Yiru Shen
Purpose: FPS and NPS from distance matrix
"""
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import collections

def nearest_selection(distance_matrix, num_samples, start_index):
    ### replace diagnal distance to a large number ###
    n = distance_matrix.shape[0]
    distance_matrix[range(n), range(n)] = 9999

    # seen = set()
    seen = list()
    # seen.add(start_index)
    seen.append(start_index)
    current_index = start_index
    cur_patch = 1

    nearest_distance = []

    while True:
        if cur_patch >= num_samples:
            break
        select_idx = np.argmin(distance_matrix[current_index,:])
        if select_idx not in seen:
            # seen.add(select_idx)
            seen.append(select_idx)
            # print("{} --> {}, distance: {}".format(current_index, select_idx, distance_matrix[current_index, select_idx]))
            nearest_distance.append(distance_matrix[current_index, select_idx])
            cur_patch += 1
            distance_matrix[current_index, select_idx] = 9999
            current_index = select_idx
            continue
        else:
            distance_matrix[current_index, select_idx] = 9999

    # seen = list(seen)

    return seen, nearest_distance


def visualize_selection(ptc, selection_list, nearest_distance, num_to_vis=100):
    fig = plt.figure(figsize=(10, 10))
    for i in range(num_to_vis):
        ax = fig.add_subplot(10, 10, i+1, projection='3d')
        p = ptc[selection_list[i],:]
        ax.scatter(p[:,0], p[:,1], p[:,2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if i==0:
            ax.set_title("{}".format(selection_list[i]))
        else:
            ax.set_title("{}, {:.3f}".format(selection_list[i], nearest_distance[i-1]))

    plt.tight_layout()
    plt.show()

def main():

    with open("../voxel/test_data_filename.pkl", "rb") as f:
        test_filename = pickle.load(f)

    distance_matrix = np.load("distance_matrix/GT_all_distance_matrix.npy")

    N = distance_matrix.shape[0]
    num_samples = 2000
    start_index = random.randint(0, N)
    print("start index: {}".format(start_index))
    # np.save("subsample/GT_start_index.npy", start_index)

    ptc = np.load("faster_ss_data/testset_256points.npy")

    ## convert from large number to small
    distance_matrix_negative = -1 * distance_matrix
    selection_list, nearest_distance = nearest_selection(distance_matrix=distance_matrix_negative, num_samples=num_samples, start_index=start_index)
    np.save("subsample/FPS_sample_2000_start_index_{}.npy".format(start_index), selection_list)
    np.save("subsample/FPS_distance_2000_start_index_{}.npy".format(start_index), nearest_distance)

    nearest_distance_positive = [-1*d for d in nearest_distance]

    sample_type = {}
    for idx, selection in enumerate(selection_list):
        name = test_filename[selection]
        class_type = name.split("/")[0]
        if class_type in sample_type:
            sample_type[class_type] += 1
        else:
            sample_type[class_type] = 1

    for ix, shape in enumerate(sample_type.keys()):
        print("{}, {}".format(shape, sample_type[shape]))

#     cnt = instance_cnt[shape]
#     emd = instance_emd[shape]
#     cd = instance_cd[shape]

    fig = plt.figure()
    plt.plot(np.arange(len(nearest_distance)), nearest_distance_positive)
    plt.show()

    # visualize_selection(ptc, selection_list, nearest_distance)
    # reversed_list = selection_list[::-1]
    # reversed_dist = nearest_distance[::-1]
    # visualize_selection(ptc, reversed_list, reversed_dist)

if __name__ == "__main__":
    main()
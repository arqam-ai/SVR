"""
Author: Yiru Shen
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

def main():

    with open("../voxel/test_data_filename.pkl", "rb") as f:
        test_filename = pickle.load(f)

    subsample_type = "NPS"
    distance_matrix = np.load("distance_matrix/GT_all_distance_matrix.npy")
    ptc = np.load("faster_ss_data/testset_256points.npy")

    folder_for_subsample = "subsample"
    if not os.path.isdir(folder_for_subsample):
        os.mkdir(folder_for_subsample)

    dst_folder = os.path.join(folder_for_subsample, subsample_type)
    if not os.path.isdir(dst_folder):
        os.mkdir(dst_folder)

    N = distance_matrix.shape[0]
    num_samples = 2000

    for step in range(10):
        # random_start = step * 1000
        # if step < 9:
        #     random_end = (step+1)*1000
        # else:
        #     random_end = N
        start_index = random.randint(0, N)
        print("{}, start index: {}".format(subsample_type, start_index))
        # np.save("subsample/GT_start_index.npy", start_index)

        if subsample_type == "FPS":
            ## convert from large number to small
            distance_matrix_negative = -1 * distance_matrix
            selection_list, nearest_distance = farthest_selection(distance_matrix=distance_matrix_negative, num_samples=num_samples, start_index=start_index)
        elif subsample_type == "NPS":
            selection_list, nearest_distance = nearest_selection(distance_matrix=distance_matrix,
                                                                  num_samples=num_samples, start_index=start_index)

        dst_filename = os.path.join(dst_folder, "sample_2000_startIndex_{}.npy".format(start_index))
        np.save(dst_filename, selection_list)
        dst_filename = os.path.join(dst_folder, "distance_2000_startIndex_{}.npy".format(start_index))
        np.save(dst_filename, nearest_distance)
        # print(selection_list[:10])

        fig = plt.figure()
        nearest_distance_positive = [-1 * d for d in nearest_distance]
        plt.plot(np.arange(len(nearest_distance)), nearest_distance_positive)
        plt.savefig(os.path.join("subsample/images", "{}_sample_distribution_startIndex_{}.png".format(subsample_type, start_index)))

        visualize_selection(
            ptc,
            selection_list,
            filename=os.path.join("subsample/images",
                                  "{}_TOP_samples_{}__startIndex_{}.png".format(subsample_type, num_samples, start_index)),
        )
        reversed_list = selection_list[::-1]
        reversed_dist = nearest_distance[::-1]
        visualize_selection(ptc, reversed_list,
                            filename=os.path.join("subsample/images",
                            "{}_BOT_samples_{}__startIndex_{}.png".format(subsample_type, num_samples, start_index)))

    # for i in range(10):
    #     name = test_filename[selection_list[i]]
    #     class_type = name.split("/")[0]
    #     print(selection_list[i], nearest_distance[i], class_type)

    # #### statistics of selected samples ####
    # sample_type = {}
    # for idx, selection in enumerate(selection_list):
    #     name = test_filename[selection]
    #     class_type = name.split("/")[0]
    #     if class_type in sample_type:
    #         sample_type[class_type] += 1
    #     else:
    #         sample_type[class_type] = 1
    #
    # for ix, shape in enumerate(sample_type.keys()):
    #     print("{}, {}".format(shape, sample_type[shape]))
    # #### statistics of selected samples ####

    # fig = plt.figure()
    # nearest_distance_positive = [-1 * d for d in nearest_distance]
    # plt.plot(np.arange(len(nearest_distance)), nearest_distance_positive)
    # plt.savefig(os.path.join("subsample/images", "FPS_sample_distribution_startIndex_{}.png".format(start_index)))
    # # plt.show()
    #
    # visualize_selection(
    #     ptc,
    #     selection_list,
    #     nearest_distance,
    #     filename=os.path.join("subsample/images", "FPS_TOP_samples_{}__startIndex_{}.png".format(num_samples, start_index)),
    #     test_listname=test_filename,
    # )
    # reversed_list = selection_list[::-1]
    # reversed_dist = nearest_distance[::-1]
    # visualize_selection(ptc, reversed_list, reversed_dist, filename=os.path.join("subsample/images", "FPS_BOT_samples_{}__startIndex_{}.png".format(num_samples, start_index)))

if __name__ == "__main__":
    main()
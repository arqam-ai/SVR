import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nearest_selection(distance_matrix, num_samples, start_index):
    ### replace diagnal distance to a large number ###
    n = distance_matrix.shape[0]
    distance_matrix[range(n), range(n)] = 9999

    seen = set()
    seen.add(start_index)
    current_index = start_index
    cur_patch = 1

    while True:
        if cur_patch >= num_samples:
            break
        select_idx = np.argmin(distance_matrix[current_index,:])
        if select_idx not in seen:
            seen.add(select_idx)
            cur_patch += 1
            distance_matrix[current_index, select_idx] = 9999
            current_index = select_idx
            continue
        else:
            distance_matrix[current_index, select_idx] = 9999

    return seen


def visualize_selection(ptc, selection_list, num_to_vis=64):
    fig = plt.figure(figsize=(8,8))
    for i in range(num_to_vis):
        ax = fig.add_subplot(8, 8, i+1, projection='3d')
        p = ptc[selection_list[i],:]
        ax.scatter(p[:,0], p[:,1], p[:,2])
        if i==0:
            ax.set_xtitle("X")
            ax.set_ytitle("Y")
            ax.set_ztitle("Z")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.tight_layout()
    plt.show()

def main():
    N = distance_matrix.shape[0]
    start_index = random.randint(0, N)
    selection_list = nearest_selection(distance_matrix=distance_matrix, num_samples=num_samples,start_index=start_index)
    visualize_selection(ptc_test, selection_list)
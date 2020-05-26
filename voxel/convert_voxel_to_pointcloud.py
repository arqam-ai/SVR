"""
Author: Yiru Shen
Purpose: convert voxel to point cloud
"""

import numpy as np
import pandas as pd
import pickle
from pyntcloud import PyntCloud
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import binvox_rw

import sys, os, glob
import scipy.ndimage as nd

from compute_miou import read_binvox, plot_voxel


def visualize(ptc):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ptc[:,0], ptc[:,1], ptc[:,2])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.tight_layout()
    plt.show()

def create_coor_table(range=1.0, n=128):
    step = range / float(n)
    start = 0.5 * step

    coor = np.empty((n,n,n,3), dtype=np.float)
    for x in np.arange(n):
        for y in np.arange(n):
            for z in np.arange(n):
                coor[x][y][z] = [start+x*step, start+y*step, start+z*step]

    return coor

def extract_pointcloud(voxel, coor):
    ix, iy, iz = np.where(voxel==1)
    ptc = []
    for a, b, c in zip(ix, iy, iz):
        ptc.append(coor[a,b,c])
    ptc = np.asarray(ptc)
    return ptc

def main():
    list_mean_shapes = glob.glob("binvox_mean/*.binvox")
    list_mean_shapes.sort()
    print("#mean shapes: {}".format(len(list_mean_shapes)))

    # coor = create_coor_table()
    # print(coor.shape)
    # np.save("voxel_2_ptc_coor.npy", coor)
    coor = np.load("voxel_2_ptc_coor.npy")

    index2 = np.arange(1024)
    np.random.shuffle(index2)

    np.save("../eval/faster_ss_data/voxel2ptc_256points_index.npy", index2)

    new_ptc = np.empty((len(list_mean_shapes), 256, 3), dtype=np.float)
    for idx, file in enumerate(list_mean_shapes):
        if idx % 50 == 0:
            print("Processing {} .........................".format(idx))

        gt_voxel = read_binvox(file, shape=(128,128,128))
        # plot_voxel(gt_voxel)

        ptc = extract_pointcloud(gt_voxel, coor)
        # print(ptc.shape)
        index = np.arange(ptc.shape[0])
        np.random.shuffle(index)
        sp = ptc[index[0:1024], :]
        sp2 = sp[index2[0:256], :]
        new_ptc[idx,:] = sp2

        # fig = plt.figure()
        # ax = fig.add_subplot(121, projection='3d')
        # ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2])
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        #
        # # index = np.arange(sp.shape[0])
        # sp2 = sp[index2[0:256], :]
        #
        # ax = fig.add_subplot(122, projection='3d')
        # ax.scatter(sp2[:, 0], sp2[:, 1], sp2[:, 2])
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        #
        # plt.tight_layout()
        # plt.show()
        # plot_voxel(gt_voxel)
        # a = 1
    np.save("../eval/faster_ss_data/voxel2ptc_256points.npy", new_ptc)

if __name__ == "__main__":
    main()
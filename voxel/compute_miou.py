"""
Author: Yiru Shen
Purpose: convert point cloud to voxel, compute mIoU between voxels
"""

import numpy as np
import pandas as pd
import pickle
from pyntcloud import PyntCloud
from collections import defaultdict
import matplotlib.pyplot as plt

import binvox_rw

import sys, os, glob
import scipy.ndimage as nd

def convert_array_to_dataframe(data):
    columns = ['x', 'y', 'z']
    df = pd.DataFrame(data=data, columns=columns)
    return df

def convert_ptc_to_voxel(ptc, n_x=128, n_y=128, n_z=128):
    cloud = PyntCloud(ptc)
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=n_x, n_y=n_y, n_z=n_z)
    voxelgrid = cloud.structures[voxelgrid_id]

    x_cords = voxelgrid.voxel_x
    y_cords = voxelgrid.voxel_y
    z_cords = voxelgrid.voxel_z

    voxel = np.zeros((n_x, n_y, n_z)).astype(np.bool)
    for x, y, z in zip(x_cords, y_cords, z_cords):
        voxel[x][y][z] = True

    # plot_voxel(voxel)

    return 1.0*voxel

def resize(voxel, shape):
    """
    resize voxel shape
    """
    ratio = shape[0] / voxel.shape[0]
    voxel = nd.zoom(voxel,
            ratio,
            order=1,
            mode='nearest')
    voxel[np.nonzero(voxel)] = 1.0
    return voxel

def read_binvox(path, shape=(128, 128, 128), fix_coords=True):
    """
    read voxel data from .binvox file
    """
    with open(path, 'rb') as f:
        voxel = binvox_rw.read_as_3d_array(f, fix_coords)

    voxel_data = voxel.data.astype(np.float)
    if shape is not None and voxel_data.shape != shape:
        voxel_data = resize(voxel.data.astype(np.float64), shape)

    return voxel_data

def plot_voxel(voxel, title=None, save_file = None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel, edgecolor='k')
    # plt.title(title)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)

def parse_test_gt_filelist(path="/home/yirus/Projects/ECCV2020_rebuttal/model/test_data_filename.pkl"):
    with open(path, "rb") as f:
        test_filelist = pickle.load(f)
    return test_filelist

def read_prediction_ptc_array(path="/home/yirus/Projects/ECCV2020_rebuttal/model/cluster_matrix_fast/clustering_prediction.npy"):
    pred = np.load(path)
    return pred

def SingleIoU(s1, s2):
    """
    :param s1: shape 1
    :param s2: shape 2
    :return: Iou
    """
    v1 = np.sum(s1 > 0.5)
    v2 = np.sum(s2 > 0.5)
    I = np.sum((s1 > 0.5) * (s2 > 0.5))
    U = v1 + v2 - I
    if U == 0:
        IoU = 1
    else:
        IoU = float(I) / float(U)

    return IoU

def main():
    gt_list = parse_test_gt_filelist()
    print(len(gt_list))

    tot_iou = []
    instance_iou = defaultdict(list)

    preds = read_prediction_ptc_array(path="two_folding_predictions.npy")
    print(preds.shape)
    for idx, file in enumerate(gt_list):
        class_type = file.split("/")[0]

        if idx % 1000 == 0:
            print("Processing {} .........................".format(idx))

        file = os.path.join("/home/yirus/Datasets/what3d/voxels/", file+"/0.binvox")
        gt_voxel = read_binvox(file, shape=(32,32,32))
        # plot_voxel(gt_voxel)
        pred_pd = convert_array_to_dataframe(preds[idx])
        pred_voxel = convert_ptc_to_voxel(pred_pd, n_x=32, n_y=32, n_z=32)
        iou = SingleIoU(gt_voxel, pred_voxel)
        tot_iou.append(iou)
        instance_iou[class_type].append(iou)

    with open("two_folding_miou.pkl", "wb") as f:
        pickle.dump(instance_iou, f)

    print("..............................................................................")
    print("mIoU: {:.3f}".format(np.mean(tot_iou)))

    for class_type in instance_iou:
        print("{}\t{}\t{:.4f}\t{:.4f}".format(class_type,
                                              len(instance_iou[class_type]),
                                              np.average(instance_iou[class_type]),
                                              np.std(instance_iou[class_type])))

    with open("clustering_miou.pkl", "wb") as f:
        pickle.dump(instance_iou, f)

if __name__ == "__main__":
    main()
import os
import sys
sys.path.append('../../')
import open3d as o3d
import trimesh 
import argparse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from utils.utils import check_exist_or_mkdirs, trimesh_remove_texture
import utils.binvox_rw as binvox_rw
import utils.binvox_torch as binvox_torch
from tqdm import tqdm
import logging 
from shutil import copyfile

class DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, a, b, c):
       logging.disable(logging.NOTSET)

def create_voxel_folder(dataset_path, model_path, voxels_dirname, instance_num = 52430):

    f_class = open(os.path.join(dataset_path, "splits/classes.txt"),"r")
    voxels_path = os.path.join(dataset_path, voxels_dirname)
    check_exist_or_mkdirs(voxels_path)
    class_num = 0
    counter = 0
    fname = 'voxel_object'
    pbar = tqdm(total = instance_num)
    for cl in f_class:
        index = cl.find(' ')
        class_num += 1
        clname = cl[:index]
        clid = cl[index+1:-1]
        check_exist_or_mkdirs(os.path.join(voxels_path, clname))
        for split in ['train','val','test']:
            f = open(os.path.join(dataset_path, "splits/lists/%s/%s.txt"%(clname,split)),"r")
            for x in f:
                instance_id = x[:-1]
                check_exist_or_mkdirs(os.path.join(voxels_path, clname, instance_id))
                what3d_voxel_path = os.path.join(voxels_path, clname, instance_id, '%s.binvox' % fname)
                shapenet_voxel_path = os.path.join(args.model_path, clid, instance_id, "models", "model_normalized.solid.binvox")
                copyfile(shapenet_voxel_path, what3d_voxel_path)
                counter += 1
                pbar.update(n=1)
    pbar.close()
    print('Total Instance Number: %d, Total Class Number: %d' % (counter, class_num))
    
def visualize_points():
    for i in range(10):
        plydata = o3d.io.read_point_cloud("../results/%d.ply"% i)
        o3d.visualization.draw_geometries([plydata])

def main(args):
    #visualize_points()
    create_voxel_folder(args.dataset_path, args.model_path, args.voxel_dirname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='object-centered mode')
    parser.add_argument("--dataset-path",type=str,default="/home/../../public/zyf/What3D/",
                        help='path to dataset')
    parser.add_argument("--voxel-dirname",type=str,default="voxels_object",
                        help='path to save sampled pointcloud')                    
    parser.add_argument("--model-path",type=str,default="/home/../../public/zyf/ShapeNetCore.v2",
                        help='path of load mesh model') 
    args = parser.parse_args()
    print(str(args))
    sys.stdout.flush()
    main(args)



    # with open(shapenet_voxel_path, 'rb') as f:
    #     test_voxel = binvox_rw.read_as_3d_array(f)
    # with open(shapenet_voxel_path, 'rb') as f:
    #     test_coordinate = binvox_rw.read_as_coord_array(f)
    #     print(test_coordinate.data.shape)
    # print("%f MB" % (test_voxel.data.size * test_voxel.data.itemsize / 1000000))
    # print("%f MB" % (test_coordinate.data.size * test_voxel.data.itemsize / 1000000))
    # with open(shapenet_voxel_path, 'rb') as f:
    #     test_pytorch_coordinate = binvox_torch.read_binvox_coords(f)
    # print(test_pytorch_coordinate.shape)
    # print(test_pytorch_coordinate)
    # break
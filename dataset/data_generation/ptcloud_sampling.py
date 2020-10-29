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
from tqdm import tqdm
import logging 

class DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, a, b, c):
       logging.disable(logging.NOTSET)

def create_ptcloud_folder(dataset_path, model_path, points_dirname, num_points, instance_num = 52430):

    f_class = open(os.path.join(dataset_path, "splits/classes.txt"),"r")
    points_path = os.path.join(dataset_path, points_dirname)
    check_exist_or_mkdirs(points_path)
    class_num = 0
    counter = 0
    fname = 'ptcloud'
    #pbar = tqdm(total = instance_num)
    for cl in f_class:
        index = cl.find(' ')
        class_num += 1
        clname = cl[:index]
        clid = cl[index+1:-1]
        check_exist_or_mkdirs(os.path.join(points_path, clname))
        for split in ['train','val','test']:
            f = open(os.path.join(dataset_path, "splits/lists/%s/%s.txt"%(clname,split)),"r")
            for x in f:
                instance_id = x[:-1]
                check_exist_or_mkdirs(os.path.join(points_path, clname, instance_id))
                ptcloud_path = os.path.join(points_path, clname, instance_id, '%s.ply' % fname)
                mesh_path = os.path.join(args.model_path, clid, instance_id, "models", "model_normalized.obj")
                with DisableLogger():
                    mesh_or_scene = trimesh.load_mesh(mesh_path)
                obj_mesh = trimesh_remove_texture(mesh_or_scene)
                obj_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(obj_mesh.vertices), 
                                                    o3d.utility.Vector3iVector(obj_mesh.faces))
                ptcloud = obj_mesh.sample_points_uniformly(number_of_points = num_points)
                o3d.io.write_point_cloud(ptcloud_path, ptcloud)
                counter += 1
                print(counter)
                #pbar.update(n=1)
    #pbar.close()
    print('Total Instance Number: %d, Total Class Number: %d' % (counter, class_num))
    
def visualize_points():
    for i in range(10):
        plydata = o3d.io.read_point_cloud("../results/%d.ply"% i)
        o3d.visualization.draw_geometries([plydata])

def main(args):
    #visualize_points()
    create_ptcloud_folder(args.dataset_path, args.model_path, args.points_dirname, args.num_points)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='object-centered mode')
    parser.add_argument("--dataset-path",type=str,default="../../../What3D/",
                        help='path to dataset')
    parser.add_argument("--points-dirname",type=str,default="points_object_dense",
                        help='path to save sampled pointcloud')                    
    parser.add_argument("--model-path",type=str,default="../../../ShapeNetCore.v2",
                        help='path of load mesh model')
    parser.add_argument("--num-points",type=int,default=10000,
                        help='number of points to sample')    
    args = parser.parse_args()
    print(str(args))
    sys.stdout.flush()
    main(args)
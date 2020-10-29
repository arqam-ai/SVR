"""
Author: Yiru Shen
Purpose: convert point cloud to voxel, compute mIoU between voxels
"""
import sys
import numpy as np
import pandas as pd
import pickle
from pyntcloud import PyntCloud
from collections import defaultdict
import matplotlib.pyplot as plt
sys.path.append("../")
from utils.utils import Normalization, trimesh_remove_texture
import argparse
import utils.binvox_rw as binvox_rw
import tqdm
import torch
import trimesh
#from utils.trimesh.trimesh.voxel.creation import local_voxelize
import sys, os, glob
import scipy.ndimage as nd
#import kaolin as kal

def class_counter(args, split_name):
    f_class = open(os.path.join(args.data_basedir, args.splits_path, args.class_path),"r")
    
    class_num = 0
    class_dic = {}           # class_name : num of instance in this class
    class_index = {}         # class_name : class_index     e.g. airplane:0
    class_list = []          # 55 class
    data_class = []          # airairairchairchair
    color = []
    for line in f_class:
        index = line.find(' ')
        clname = line[:index]
        class_dic[clname] = 0
        class_list += [clname]
        class_index[clname] = class_num
        class_num += 1
        
    instance_num = 0
    for clname in tqdm.tqdm(class_list,total= len(class_list), desc = '%s'%split_name):
        f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt'%split_name),"r")
        for x in f:
            class_dic[clname] += 1
            instance_num += 1
            data_class += [clname]
            color += [class_index[clname]]
    
    #print(instance_num,class_dic, data_class) 
    return class_list, instance_num, class_dic, data_class, color


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

    x_cords = np.expand_dims(x_cords, axis=0)
    y_cords = np.expand_dims(y_cords, axis=0)
    z_cords = np.expand_dims(z_cords, axis=0)
    cords = np.concatenate((x_cords, y_cords, z_cords), axis=0)
    
    return cords, voxel



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
    plt.title(title)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)

def plot_cords(cords, title=None, save_file=None):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(cords[0,:], cords[1,:], cords[2,:], s = 2)
    ax.set_xlim([0,128])
    ax.set_ylim([0,128])
    ax.set_zlim([0,128])
    ax.view_init(30, 135)
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(cords[0,:], cords[1,:], cords[2,:], s = 2)
    ax.set_xlim([0,128])
    ax.set_ylim([0,128])
    ax.set_zlim([0,128])
    ax.view_init(30, 90)
    plt.title(title)
    if save_file is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save_file)
        plt.close()

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


def convert(args):

    pred_indexes = np.load("/home/zyf/SVR/experiment/Oracle_NN/Oracle_object/exp2/predict_trainindex.npy")
    index = pred_indexes[2200][0]
    ptcloud_pth = "/home/zyf/What3D/ptcloud_object.npz"
    ptcloud = np.load(ptcloud_pth)['train'][index]
    ptcloud_cords, voxel = convert_ptc_to_voxel(convert_array_to_dataframe(ptcloud))
    plot_cords(ptcloud_cords, title="ptcloud_cords_{}".format(index), save_file="../img/voxel/ptcloud_cords_{}.png".format(index))
    normed_ptcloud = Normalization(torch.from_numpy(ptcloud), inplace=True, keep_track=False).normalize_unitL2ball()
    normed_ptcloud_cords, voxel = convert_ptc_to_voxel(convert_array_to_dataframe(normed_ptcloud.numpy()[0]))
    plot_cords(normed_ptcloud_cords, title="normed_ptcloud_cords_{}".format(index), save_file="../img/voxel/normed_ptcloud_cords_{}.png".format(index))

    #print(ptcloud_cords.shape)
    #np.save("../img/voxel/test_cords_{}.npy".format(index), cords)

    class_list, _, _, _,_ = class_counter(args, "test")
    voxel_path_dic = {"train":[], 'test':[]}
    for split in ['train', 'test']:
        for clname in class_list:
            f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt'%split),"r")
            for x in f:
                instance_id = x[:-1]
                voxel_path_dic[split].append(os.path.join(args.data_basedir, args.voxel_path, clname, instance_id, "voxel_object.binvox"))
    
    with open(voxel_path_dic['train'][index], 'rb') as f:
        voxel_cords = binvox_rw.read_as_coord_array(f).data
    plot_cords(voxel_cords, title="voxel_cords_{}".format(index), save_file="../img/voxel/voxel_cords_{}.png".format(index))
    print(voxel_cords.shape)
    voxel_cords = voxel_cords.transpose(1,0)
    normed_voxel_cords, voxel = convert_ptc_to_voxel(convert_array_to_dataframe(voxel_cords))
    print(normed_voxel_cords.shape)
    plot_cords(normed_voxel_cords, title="normed_voxel_cords_{}".format(index), save_file="../img/voxel/normed_voxel_cords_{}.png".format(index))

def test_mesh2voxel(path):
    
    # test_mesh = trimesh.creation.uv_sphere(radius=1.0, count=[32, 32])
    # #voxel = trimesh.voxel.local_voxelize(test_mesh, point=test_mesh.centroid, pitch= test_mesh.extents.max() / 128, radius=64, fill=True)
    # voxel = test_mesh.voxelized(pitch= test_mesh.extents.max() / 128)
    # points = voxel.points      #(1128955, 3)

    # # points = trimesh.sample.volume_mesh(test_mesh, count=10000)
    # cords, voxel = convert_ptc_to_voxel(convert_array_to_dataframe(points))
    # plot_cords(cords, title="test trimesh Voxel %d"%3 , save_file="../img/voxel/trimesh_voxel/test_trimesh_%d.png"%3)
    
    # for index in range(53):
    #     print(index)
    #     vertices = np.load(os.path.join(path, "mesh_vertices_%d.npy" % index))
    #     faces = np.load(os.path.join(path, "mesh_faces_%d.npy" % index))
    #     mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    #     points = trimesh.sample.volume_mesh(mesh, count=10000)
    #     cords, voxel = convert_ptc_to_voxel(convert_array_to_dataframe(points))
    #     plot_cords(cords, title="AtlasNet Voxel %d"%index, save_file="../img/voxel/atlasnet_voxel/test_sample_%d,png"%index)
        
    for index in range(53):
        print(index)
        vertices = np.load(os.path.join(path, "mesh_vertices_%d.npy" % index))
        faces = np.load(os.path.join(path, "mesh_faces_%d.npy" % index))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        voxel = mesh.voxelized(pitch= mesh.extents.max() / 128)
        points = voxel.points
        cords, voxel = convert_ptc_to_voxel(convert_array_to_dataframe(points))
        print(voxel.shape)
        #plot_cords(cords, title="trimesh Voxel %d"%index, save_file="../img/voxel/atlasnet_voxel/trimesh_sample_%d.png"%index)
        plot_cords(cords)

    #######Mesh visualization
    '''
    for index in range(53):
        vertices = np.load(os.path.join(path, "mesh_vertices_%d.npy" % index))
        faces = np.load(os.path.join(path, "mesh_faces_%d.npy" % index))
        #faces = torch.from_numpy(faces).long()
        #mesh = kal.rep.TriangleMesh.from_tensors(torch.from_numpy(vertices), faces)
        #voxel = kal.conversions.meshconversions.trianglemesh_to_voxelgrid(mesh, resolution=128)
        #voxel = voxel>0
        #voxel = voxel.numpy()
        #cords = np.argwhere(voxel)
        #cords = np.transpose(cords, (1, 0))
        #print(cords.shape)
        #for i in range(cords.shape[0]):
        #    print(cords[i])
        #plot_cords(cords, title="AtlasNet Voxel %d"%index, save_file="../img/voxel/atlasnet_voxel/test_sample_%d,png"%index)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.show()
    '''


def vis(ptcloud_pth):
    ptcloud = np.load(ptcloud_pth)
    plot_cords(ptcloud)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--data-basedir',type=str,default='/home/zyf/What3D',
                    help='path of the jsonfile')
    parser.add_argument('--img-path',type=str,default='renderings',
                    help='path of the jsonfile')
    parser.add_argument('--voxel-path', type=str, default='voxels_object',
                    help='path of the voxel data')
    parser.add_argument("--splits-path",dest="splits_path", type=str, default='splits',
                      help='path of the data folder') 
    parser.add_argument("--ptcloud-path",type=str,default="../../What3D/ptcloud_object.npz",help=' ' )
    parser.add_argument("--class-path", dest="class_path", type=str,default='classes.txt',help="class name list")
    parser.add_argument("--views",dest="views", type=str,default= '0',
                    help="five view for each instance")
    parser.add_argument("--save-path",dest="save_path", type=str,default= 'exp2',
                    help="")                
    args = parser.parse_args(sys.argv[1:])
    #args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.script_folder = os.path.dirname(os.path.abspath(__file__))
    #convert(args)
    #vis("../img/voxel/test_cords_13000.npy")
    test_mesh2voxel(path="../experiment/object_models_v2/mesh_generation/partial_final_vis")
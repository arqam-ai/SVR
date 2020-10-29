import os
import sys
import argparse
import numpy as np
import tqdm
sys.path.append("../../../")
from utils.what3d_util import get_class_list, get_class_models
from utils.f_score import calculate_fscore
import utils.binvox_rw as binvox_rw
from utils.loss import evaluate_voxel_prediction
from utils.utils import trimesh_remove_texture
from voxel.voxel_carving import voxel_carving, voxelGrid_to_cords
from model.what3d.Cluster import find_median
import time
from voxel.compute_miou import convert_ptc_to_voxel, convert_array_to_dataframe
from multiprocessing import Pool, Manager
import open3d as o3d
import trimesh 

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--data-basedir',type=str,default='../../../../What3D',
                    help='base path')
args = parser.parse_args(sys.argv[1:])

def binvoxe_rw_search(test_index, test_voxel_path, cat):
    with open(test_voxel_path, "rb") as f:
        gt = binvox_rw.read_as_coord_array(f).data
        #gt_voxel = binvox_rw.read_as_3d_array(f).data
    gt_voxel_cords, gt_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(gt.transpose(1, 0)))
    id_lists = get_class_models(cat, mode="train")

    max_iou = 0
    max_index = 0
    for order, instance_id in enumerate(id_lists):
        train_path = os.path.join(args.data_basedir, "voxels_object", cat, instance_id, "voxel_object.binvox")
        with open(train_path, "rb") as f:
            pred = binvox_rw.read_as_coord_array(f).data
        pred_voxel_cords, pred_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(pred.transpose(1, 0)))
        tmp_iou = evaluate_voxel_prediction(pred_voxel, gt_voxel)
        if tmp_iou > max_iou:
            max_iou = tmp_iou
            max_index = order

    print('test index:%d, cat:%s, iou:%.6f, predicted idx in trainset:%d' % (test_index, cat, max_iou, max_index))
    return (test_index, max_index, max_iou)

def npz_search(test_index, test_voxel_path, cat):
    
    gt_voxel = np.load(test_voxel_path)["voxel"] 
    id_lists = get_class_models(cat, mode="train")
    max_iou = 0
    max_index = 0
    for order, instance_id in enumerate(id_lists):
        train_path = os.path.join(args.data_basedir, "voxels_object", cat, instance_id, "normalized_voxel_object.npz")
        pred_voxel = np.load(train_path)["voxel"]
        tmp_iou = evaluate_voxel_prediction(pred_voxel, gt_voxel)
        if tmp_iou > max_iou:
            max_iou = tmp_iou
            max_index = order

    print('test index:%d, cat:%s, iou:%.6f, predicted idx in trainset:%d' % (test_index, cat, max_iou, max_index))
    return (test_index, max_index, max_iou)

def normalize(args):
    class_list = get_class_list()
    f_class = open(os.path.join(args.data_basedir, "splits/classes.txt"),"r")
    class_id = {}
    exception_list = []
    for cl in f_class:
        index = cl.find(' ')
        clname = cl[:index]
        clid = cl[index+1:-1]
        class_id.update({clname:clid})
    
    path_lists = []

    for cat in class_list:
        test_id_lists = get_class_models(cat, mode="test")
        train_id_lists = get_class_models(cat, mode="train")
        for instance_id in test_id_lists:
            path_lists.append(os.path.join(args.data_basedir, "voxels_object", cat, instance_id, "voxel_object.binvox"))
        for instance_id in train_id_lists:
            path_lists.append(os.path.join(args.data_basedir, "voxels_object", cat, instance_id, "voxel_object.binvox"))
    offset = 0
    path_lists = path_lists[offset:]

    for order, path in enumerate(path_lists):
        print(offset + order)
        with open(path, "rb") as f:
            voxel_object = binvox_rw.read_as_coord_array(f)
        if voxel_object.data.shape[1] == 0:
            exception_list.append((offset + order, path))
            indexes = [i for i in range(len(path)) if path.startswith("/", i)]
            mesh_path = os.path.join("~/../../public/zyf/ShapeNetCore.v2/", class_id[path[indexes[-3]+1 : indexes[-2]]], 
                             path[indexes[-2]+1:indexes[-1]], "models", "model_normalized.obj")
            #mesh_path = os.path.join(path[:indexes[-1]], "models", "model_normalized.obj")
            #print(mesh_path)
            mesh = trimesh_remove_texture(trimesh.load(mesh_path))
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
            voxel_grid, voxel_solid, voxel_surface = voxel_carving(
                 mesh, None, o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=6), 2 * mesh.get_axis_aligned_bounding_box().get_max_extent(), voxel_resolution=128.0, surface_method="pointcloud")
            voxel_cords = voxelGrid_to_cords(voxel_grid)
            normed_voxel_cords, voxel = convert_ptc_to_voxel(convert_array_to_dataframe(voxel_cords))
            np.savez_compressed(os.path.join(path[:-19], "normalized_voxel_object.npz"), voxel=voxel)
            
        else:
            voxel_cords, voxel = convert_ptc_to_voxel(convert_array_to_dataframe(voxel_object.data.transpose(1, 0)))
            np.savez_compressed(os.path.join(path[:-19], "normalized_voxel_object.npz"), voxel=voxel)

    print("Exception Number: {}".format(len(exception_list)))
    print("Exception instances:", exception_list)
    # Exception list 
    # 11266  What3D/voxels_object/car(02958343)/876d92ce6a0e4bf399588eee976baae/voxel_object.binvox
    # 12637  '../../../../What3D/voxels_object/car/7edb40d76dff7455c2ff7551a4114669/voxel_object.binvox'


num_processors = 25
class_list = get_class_list()
ins_order = 0
path_cat_lists = [] 
for cat in class_list:
    id_lists = get_class_models(cat, mode="test")
    for instance_id in id_lists:
        path_cat_lists.append((ins_order, os.path.join(args.data_basedir, "voxels_object", cat, instance_id, "normalized_voxel_object.npz"),
                       cat))
        ins_order += 1

start_time = time.time()
##################################################################################
pool = Pool(processes = num_processors)
print("Request Pool: --- %s seconds ---" % (time.time() - start_time))
miou_tuple_list = pool.starmap(npz_search, path_cat_lists)
iou_list = []
trainindex_list = []
miou_tuple_list = sorted(miou_tuple_list, key=lambda x: x[0])
for tuple in miou_tuple_list:
    iou_list.append([tuple[0], tuple[2]])
    trainindex_list.append([tuple[0], tuple[1]])

np.save(os.path.join("exp_iou", "iou_list"), np.array(iou_list))
np.save(os.path.join("exp_iou", "train_index"), np.array(trainindex_list))

iou_list = [tuple[1] for tuple in iou_list]
print("mIoU max %.4f, min %.4f, mean %.4f" %(max(iou_list), min(iou_list), sum(iou_list)/len(iou_list)))

iou_list.sort()
middle = len(iou_list)//2
# lower quartile
lower_quartile = find_median(iou_list[:middle])
# median
median = find_median(iou_list)
# upper quartile
upper_quartile = find_median(iou_list[middle:])
print("lower_quartile %.4f, median %.4f, upper_quartile %.4f" %(lower_quartile, median, upper_quartile))

#normalize(args)



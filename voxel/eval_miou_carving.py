"""
    This is script to eval Neural Network reconstruction mIoU using open3d carving method
    Author: Yefan Zhou
"""
import os
import sys
import tqdm
from compute_miou import convert_ptc_to_voxel, convert_array_to_dataframe
from voxel_carving import voxel_carving, voxelGrid_to_cords
import trimesh
import numpy as np
sys.path.append("../")
import utils.binvox_rw as binvox_rw
from utils.loss import evaluate_voxel_prediction
from dataset.dataset import class_counter, what3d_dataset_views
from utils.utils import find_median
import random
import cv2
import torch
import time
from multiprocessing import Pool, Manager
import open3d as o3d

voxelize_method = 'trimesh'
data_basedir = "../../What3D"
splits_path = "splits"
class_path = "classes.txt"
voxel_path = "voxels_object"
voxel_name = "voxel_object.binvox"
img_path = "renderings"
views = "01234"
cluster_path = "clustering"
nn_path = "../experiment/object_models_v2/mesh_generation/"
nn_result_path = "../experiment/object_models_v2/mesh_generation/final_vis"
img_save_path = "../experiment/object_models_v2/mesh_generation/random100_miou_carving_vis"
num_processors = 20

class MIoU():
    def __init__(self, voxel_path_list, nn_result_path, image_path_list, img_save_path, resolution):
        self.voxel_path_list = voxel_path_list
        self.nn_result_path = nn_result_path
        self.img_save_path = img_save_path
        self.image_path_list = image_path_list
        self.iou_times = 0
        self.time = time.time()
        self.camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    
    def iou(self, index):
        self.iou_times += 1
        with open(self.voxel_path_list[index % inst_num], "rb") as f:
            gt_object = binvox_rw.read_as_coord_array(f)
        gt = gt_object.data 
        gt = gt.transpose(1, 0)
        gt_normed_voxel_cords, gt_normed_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(gt))

        faces = np.load(os.path.join(self.nn_result_path, "mesh_faces_{}.npy".format(index))) 
        vertices = np.load(os.path.join(self.nn_result_path, "mesh_vertices_{}.npy".format(index)))
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        voxel_grid, _ , _ = voxel_carving(mesh, output_filename=None, camera_sphere=self.camera_sphere, cubic_size=2 * mesh.get_axis_aligned_bounding_box().get_max_extent(),
                    voxel_resolution=128.0)
        voxel_cords = voxelGrid_to_cords(voxel_grid)
        pred_normed_voxel_cords, pred_normed_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(voxel_cords))

        tmp_iou = evaluate_voxel_prediction(pred_normed_voxel, gt_normed_voxel)
        print("----------------------------------")
        print("occupied voxel num in pred".format(pred_normed_voxel_cords.shape[0]))
        print("occupied voxel num in gt".format(gt_normed_voxel_cords.shape[0]))
        print("iou times %d; inferred instances %d; iou:%.4f" % (self.iou_times, index, tmp_iou))
        print("----------------------------------")
        #print("inferred instances %d; iou:%.4f" % (index, tmp_iou))

        if index % 500 == 0:
            image = cv2.imread(self.image_path_list[index])
            what3d_dataset_views.data_visualizer(torch.from_numpy(gt_normed_voxel_cords.transpose(1, 0)), 
            torch.from_numpy(pred_normed_voxel_cords.transpose(1, 0)), image, 
                "atlas_normed_voxel", self.img_save_path, index, loss = tmp_iou, type="voxel_cords")

        if self.iou_times % 30 == 0:
            print("iou times %d, used time ----------%.3f min------------------" % (self.iou_times, (time.time()-self.time)/60))

        return (index, tmp_iou)


if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

# class list 
class_list, _ = class_counter(os.path.join(data_basedir, splits_path, class_path))

## Load NN data path
split_name = "test"
instance_path_list = []
for clname in class_list:
    f = open(os.path.join(data_basedir, splits_path, 'lists', clname, '%s.txt'%split_name),"r")
    for x in f:
        instance_id = x[:-1]
        instance_path_list.append(os.path.join(clname, instance_id))


inst_num = len(instance_path_list)
view_num = len(list(views))

image_path_list = []
for view in list(views):
    for img_idx in range(inst_num):
        image_path_list.append(os.path.join(data_basedir, img_path, instance_path_list[img_idx], "{}.png".format(view)))


voxel_path_list = []
for i in range(inst_num):
    voxel_path_list.append(os.path.join(data_basedir, voxel_path, instance_path_list[i], voxel_name))

print("Loading Path Finshed------------------------------------------ ")


##Loop version
iou_list = []
camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=5)
random.seed(30)
sample_index_list = random.sample(range(inst_num * view_num), 100)
start_time = time.time()
for order, index in enumerate(sample_index_list):
    with open(voxel_path_list[index % inst_num], "rb") as f:
        gt = binvox_rw.read_as_coord_array(f).data
    gt = gt.transpose(1, 0)
    gt_normed_voxel_cords, gt_normed_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(gt))

    faces = np.load(os.path.join(nn_result_path, "mesh_faces_{}.npy".format(index))) 
    vertices = np.load(os.path.join(nn_result_path, "mesh_vertices_{}.npy".format(index)))
    if voxelize_method == 'trimesh':
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        voxel = mesh.voxelized(pitch= mesh.extents.max() / 128)
        pred_normed_voxel_cords, pred_normed_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(voxel.points))
    elif voxelize_method == 'open3d':
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        #cubic_size = mesh.get_axis_aligned_bounding_box().get_max_extent()
        cubic_size = 1.5
        print("cubic_size", cubic_size)
        voxel_grid, _ , _ = voxel_carving(mesh, output_filename=None, camera_sphere=camera_sphere, cubic_size=cubic_size,
                voxel_resolution=128.0)
        voxel_cords = voxelGrid_to_cords(voxel_grid)
        pred_normed_voxel_cords, pred_normed_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(voxel_cords))
    
    print("----------------------------------")
    print("occupied voxel num in pred {}".format(pred_normed_voxel_cords.shape[1]))
    print("occupied voxel num in gt {}".format(gt_normed_voxel_cords.shape[1]))
    tmp_iou = evaluate_voxel_prediction(pred_normed_voxel, gt_normed_voxel)
    print("inferred instances %d/%d; iou:%.4f" % (index, inst_num * view_num, tmp_iou))
    iou_list.append(tmp_iou)

    image = cv2.imread(image_path_list[index])
    what3d_dataset_views.data_visualizer(torch.from_numpy(gt_normed_voxel_cords.transpose(1, 0)), 
      torch.from_numpy(pred_normed_voxel_cords.transpose(1, 0)), image, 
        "atlas_normed_voxel_open3d_cb15", img_save_path, order, loss = tmp_iou, type="voxel_cords")
print("Used Time: {}".format(time.time()-start_time))
if voxelize_method == 'open3d':
    np.save("../experiment/object_models_v2/mesh_generation/random100_iou_{}_size1.5.npy".format(voxelize_method), np.array(iou_list))
elif voxelize_method == 'trimesh':
    np.save("../experiment/object_models_v2/mesh_generation/random100_iou_{}.npy".format(voxelize_method), np.array(iou_list))

iou_list.sort()
middle = len(iou_list)//2
# lower quartile
lower_quartile = find_median(iou_list[:middle])
# median
median = find_median(iou_list)
# upper quartile
upper_quartile = find_median(iou_list[middle:])
print("mIoU max %.4f, min %.4f, mean %.4f" %(max(iou_list), min(iou_list), sum(iou_list)/len(iou_list)))
print("lower_quartile %.4f, median %.4f, upper_quartile %.4f" %(lower_quartile, median, upper_quartile))




# ## Parallel map Mode
# start_time = time.time()
# ##################################################################################
# pool = Pool(processes = num_processors)
# print("Request Pool: --- %s seconds ---" % (time.time() - start_time))
# parallel_miou = MIoU(voxel_path_list, nn_result_path, image_path_list, img_save_path, resolution=6)
# index_list = range(inst_num * view_num)
# print("Input Amount: %d" % len(index_list))
# pall_start_time = time.time()
# tuple_iou_list = pool.map(parallel_miou.iou, index_list)
# pool.close()
# pool.join()
# print("Finshed Tasks: --- %.3f minutes ---" % ((time.time() - pall_start_time)/60))

# tuple_iou_list = sorted(tuple_iou_list, key=lambda x: x[0])
# iou_list = [tuple[1] for tuple in tuple_iou_list]
# print("Finished IoU value amount %d" % len(iou_list))

# np.save(os.path.join(nn_path, "iou_list.npy"), np.array(iou_list))
# print("mIoU max %.4f, min %.4f, mean %.4f" %(max(iou_list), min(iou_list), sum(iou_list)/len(iou_list)))

# iou_list.sort()
# middle = len(iou_list)//2
# # lower quartile
# lower_quartile = find_median(iou_list[:middle])
# # median
# median = find_median(iou_list)
# # upper quartile
# upper_quartile = find_median(iou_list[middle:])
# print("lower_quartile %.4f, median %.4f, upper_quartile %.4f" %(lower_quartile, median, upper_quartile))
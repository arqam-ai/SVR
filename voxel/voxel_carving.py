"""
    This is script to test open3d voxel carving method 
    Author: Yefan Zhou
"""


from timeit import default_timer as timer
import numpy as np
import open3d as o3d
import os
import sys
import tqdm
sys.path.append("../")
import utils.binvox_rw as binvox_rw
from utils.loss import evaluate_voxel_prediction
from voxel.compute_miou import convert_ptc_to_voxel, convert_array_to_dataframe, plot_cords
from utils.utils import Normalization, trimesh_remove_texture
from voxel.voxel_utils import plot_cords_compare
from voxel.voxel_utils import convert_cords_to_bool
from utils.utils import find_median
from dataset.dataset import what3d_dataset_views
import trimesh
import torch
import time


def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]


def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                        [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
                        [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x)


def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0, 0, 2]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans


def preprocess(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= center
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model


def voxel_carving(mesh,
                  output_filename,
                  camera_sphere,
                  cubic_size,
                  voxel_resolution,
                  w=300,
                  h=300,
                  use_depth=True,
                  surface_method='pointcloud'):
    mesh.compute_vertex_normals()
    #camera_sphere = o3d.io.read_triangle_mesh(camera_path)

    # setup dense voxel grid
    # voxel_carving = o3d.geometry.VoxelGrid.create_dense(
    #     width=cubic_size,
    #     height=cubic_size,
    #     depth=cubic_size,
    #     voxel_size=cubic_size / voxel_resolution,
    #     origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],
    #     color=[1.0, 0.7, 0.0])

    # setup dense voxel grid
    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size,
        height=cubic_size,
        depth=cubic_size,
        voxel_size=cubic_size / voxel_resolution,
        origin=np.array([[-cubic_size / 2.0], [-cubic_size / 2.0], [-cubic_size / 2.0]]))

    # rescale geometry
    camera_sphere = preprocess(camera_sphere)
    mesh = preprocess(mesh)

    # setup visualizer to render depthmaps
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # carve voxel grid
    pcd_agg = o3d.geometry.PointCloud()
    centers_pts = np.zeros((len(camera_sphere.vertices), 3))
    for cid, xyz in tqdm.tqdm(enumerate(camera_sphere.vertices), total=len(camera_sphere.vertices)):
        # get new camera pose
        trans = get_extrinsic(xyz)
        param.extrinsic = trans
        c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())
        centers_pts[cid, :] = c[:3]
        ctr.convert_from_pinhole_camera_parameters(param)

        # capture depth image and make a point cloud
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)
        pcd_agg += o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth),
            param.intrinsic,
            param.extrinsic,
            depth_scale=1)

        # depth map carving method
        if use_depth:
            voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)
        else:
            voxel_carving.carve_silhouette(o3d.geometry.Image(depth), param)
        #print("Carve view %03d/%03d" % (cid + 1, len(camera_sphere.vertices)))
    vis.destroy_window()

    # add voxel grid survace
    #print('Surface voxel grid from %s' % surface_method)
    if surface_method == 'pointcloud':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd_agg,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
    elif surface_method == 'mesh':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
    else:
        raise Exception('invalid surface method')
    voxel_carving_surface = voxel_surface + voxel_carving

    return voxel_carving_surface, voxel_carving, voxel_surface

def mesh_to_ptcloud(mesh, visualization=False):
    """convert open3d Mesh vertices to open3d Pointcloud

    Params:
    ----------
    mesh:            open3d.Mesh
    visualization :  boolean

    Returns:
    ----------
    pcd:             open3d.PointCloud
    """
    vertices = np.asarray(mesh.vertices)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(vertices))
    if visualization:
        o3d.visualization.draw_geometries([pcd])
    return pcd

def voxelGrid_to_cords(voxelGrid):
    """convert open3d VoxelGrid to numpy voxel cords 

    Params:
    ----------
    voxelGrid   : open3d.VoxelGrid

    Returns:
    ----------
    voxel_cords : numpy.array 
    """
    voxel_list = [item.grid_index for item in voxelGrid.get_voxels()]
    voxel_cords = np.array(voxel_list)
    return voxel_cords


def test_sparse_carving(mesh_path, resolution=4):
    cubic_size = 2.0
    voxel_resolution = 128.0
    output_filename = None
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    camera_path = os.path.abspath("test_data/sphere.ply")
    ori_sphere = o3d.io.read_triangle_mesh(camera_path)
    ori_vertice = mesh_to_ptcloud(ori_sphere)
    
    voxel_grid, voxel_solid, voxel_surface = voxel_carving(
        mesh, output_filename, ori_sphere, cubic_size, voxel_resolution, surface_method="pointcloud")

    ori_voxel_cords = voxelGrid_to_cords(voxel_grid)
    ori_voxel_solid = voxelGrid_to_cords(voxel_solid)
    print("---------------------------------------------------------------------------------")
    print("Resolution: {}".format(resolution))
    print("origin views number: {}".format(np.asarray(ori_vertice.points).shape[0]))
    print("surface voxels")
    print(voxel_surface)
    print("carved voxels")
    print(voxel_solid)
    print("combined voxels (carved + surface)")
    print(voxel_grid)
    print("-------------------------------")

    sparse_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    sparse_vertice = mesh_to_ptcloud(sparse_sphere)
    start = timer()
    voxel_grid, voxel_solid, voxel_surface = voxel_carving(
        mesh, output_filename, sparse_sphere, cubic_size, voxel_resolution, surface_method="pointcloud")
    sparse_voxel_cords = voxelGrid_to_cords(voxel_grid)
    end = timer()
    sparse_voxel_solid = voxelGrid_to_cords(voxel_solid)
    
    print("sparse views number: {}".format(np.asarray(sparse_vertice.points).shape[0]))
    print("Carving time %.3f seconds" % (end-start)) # Time in seconds

    print("surface voxels")
    print(voxel_surface)
    print("carved voxels")
    print(voxel_solid)
    print("combined voxels (carved + surface)")
    print(voxel_grid)
    #plot_cords(voxel_cords.transpose(1, 0))
    print("Combined MIoU: {}".format(evaluate_voxel_prediction(convert_cords_to_bool(ori_voxel_cords), convert_cords_to_bool(sparse_voxel_cords))))
    print("Solid MIoU: {}".format(evaluate_voxel_prediction(convert_cords_to_bool(ori_voxel_solid), convert_cords_to_bool(sparse_voxel_solid))))
    print("####################################################################################")


def test_shapenet_gt(mesh_path, voxel_path, resolution=4, include_origin=False):

    with open(voxel_path, "rb") as f:
        gt_object = binvox_rw.read_as_coord_array(f)
        gt_points = gt_object.data
    gt_cords, gt_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(gt_points.transpose(1, 0)))

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    cubic_size = 2 * mesh.get_axis_aligned_bounding_box().get_max_extent()
    print(cubic_size)
    voxel_resolution = 128.0
    output_filename = None
    if include_origin:
        camera_path = os.path.abspath("test_data/sphere.ply")
        ori_sphere = o3d.io.read_triangle_mesh(camera_path)
        voxel_grid, voxel_solid, voxel_surface = voxel_carving(
            mesh, output_filename, ori_sphere, cubic_size, voxel_resolution, surface_method="pointcloud")

    sparse_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    voxel_grid, voxel_solid, voxel_surface = voxel_carving(
            mesh, output_filename, sparse_sphere, cubic_size, voxel_resolution, surface_method="pointcloud")

    voxel_cords = voxelGrid_to_cords(voxel_grid)
    normed_voxel_cords, pred_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(voxel_cords))

    #print(normed_voxel_cords.shape)
    #print(gt_cords.shape)
    #plot_cords_compare(gt_cords, normed_voxel_cords)
    
    voxel_solid_cords = voxelGrid_to_cords(voxel_solid)
    normed_voxel_cords_solid, pred_voxel_solid = convert_ptc_to_voxel(convert_array_to_dataframe(voxel_solid_cords))
    print(gt_cords.shape, normed_voxel_cords.shape)
    print(gt_cords.shape, normed_voxel_cords_solid.shape)
    #plot_cords_compare(gt_cords, normed_voxel_cords_solid)

    #tri_mesh = trimesh_remove_texture(trimesh.load(mesh_path))
    #tri_voxel = tri_mesh.voxelized(pitch= tri_mesh.extents.max() / 128)
    #pred_normed_voxel_cords, pred_normed_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(tri_voxel.points))
    
    evaluate_voxel_prediction(gt_voxel, pred_voxel)
    print("IOU : %.4f" % evaluate_voxel_prediction(gt_voxel, pred_voxel))
    print("IOU : %.4f" % evaluate_voxel_prediction(gt_voxel, pred_voxel_solid))
    #print("IOU : %.4f" % evaluate_voxel_prediction(gt_voxel, pred_normed_voxel)) 


def test_shapenet_sample(mesh_path_list, voxel_solid_list, voxel_surface_list, resolution=6, combine=False):    
    total_num = len(mesh_path_list)
    iou_list = []
    for index in range(total_num):
        with open(voxel_solid_list[index], "rb") as f:
            gt_solid_object = binvox_rw.read_as_coord_array(f)
        gt_solid = gt_solid_object.data
        gt_solid_cords, gt_solid_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(gt_solid.transpose(1, 0)))
        with open(voxel_surface_list[index], 'rb') as f:
            gt_surface_object = binvox_rw.read_as_coord_array(f)
        gt_surface = gt_surface_object.data
        gt_surface_cords, gt_surface_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(gt_surface.transpose(1, 0)))
        
        if combine:
            gt_combine_voxel = np.logical_or(gt_solid_voxel, gt_surface_voxel)
            gt_combine_cords = np.concatenate((gt_surface_cords, gt_solid_cords), axis=1)

        mesh = trimesh_remove_texture(trimesh.load(mesh_path_list[index]))
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        #mesh = o3d.io.read_triangle_mesh(mesh_path_list[index])
        cubic_size = 2 * mesh.get_axis_aligned_bounding_box().get_max_extent()
        print(f"cubic_size {cubic_size}")
        voxel_resolution = 128.0
        sparse_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
        voxel_grid, voxel_solid, voxel_surface = voxel_carving(
            mesh, None, sparse_sphere, cubic_size, voxel_resolution, surface_method="pointcloud")
        voxel_cords = voxelGrid_to_cords(voxel_grid)
        normed_voxel_cords, pred_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(voxel_cords))
        
        print("occupied nums in solid {}".format(gt_solid_cords.shape[1]))
        print("occupied nums in surface {}".format(gt_surface_cords.shape[1]))
        if combine:
            print("occupied nums in combined {}".format(gt_combine_cords.shape[1]))
        print("occupied nums in carving {}".format(normed_voxel_cords.shape[1]))

        if combine:
            tmp_iou = evaluate_voxel_prediction(pred_voxel, gt_combine_voxel)
            what3d_dataset_views.data_visualizer(torch.from_numpy(gt_combine_cords.transpose(1, 0)), 
            torch.from_numpy(normed_voxel_cords.transpose(1, 0)), torch.zeros(1, 3, 224, 224), 
            "carving_combine", "../img/voxel/voxel_carving_test_gt_sample", index, loss = tmp_iou, type="voxel_cords")
        else:
            tmp_iou = evaluate_voxel_prediction(pred_voxel, gt_solid_voxel)
            what3d_dataset_views.data_visualizer(torch.from_numpy(gt_solid_cords.transpose(1, 0)), 
            torch.from_numpy(normed_voxel_cords.transpose(1, 0)), torch.zeros(1, 3, 224, 224), 
            "carving_solid", "../img/voxel/voxel_carving_test_gt_sample", index, loss = tmp_iou, type="voxel_cords")
        print("solid voxel iou %.4f" % tmp_iou)
        iou_list.append(tmp_iou)

    
    print("mIoU %.4f" % (sum(iou_list)/len(iou_list)))
    if combine:
        np.save("../img/voxel/voxel_carving_test_gt_sample/iou_list_combine.npy", np.array(iou_list))
    else:
        np.save("../img/voxel/voxel_carving_test_gt_sample/iou_list.npy", np.array(iou_list))
    return iou_list


#for r in [1, 2, 3, 4, 5, 6, 7, 8]:
#    test_sparse_carving(mesh_path="test_data/Armadillo.ply", resolution=r)

#for r in [1, 2, 3, 4, 5, 6, 7, 8]:
#    test_sparse_carving(mesh_path="test_data/airplane.obj", resolution=r)

#for r in [1, 2, 3, 4, 5, 6, 7, 8]:
#    test_sparse_carving(mesh_path="test_data/bench.obj", resolution=r)
 
#test_shapenet_gt(mesh_path="test_data/bench.obj", voxel_path="test_data/bench.binvox", resolution=6, include_origin=False)
#test_shapenet_gt(mesh_path="test_data/airplane.obj", voxel_path="test_data/airplane.binvox", resolution=6, include_origin=False)

# instance_path = []
# mesh_path = []
# voxel_solid_path = []
# voxel_surface_path = []
# with open("../../What3D/extract_sample/pathlist.txt") as f:
#     for line in f:
#         instance_path.append(line[:-1])
# instance_path = instance_path[:100]

# for path in instance_path:
#     mesh_path.append(os.path.join("../../What3D/extract_sample/", path, "models", "model_normalized.obj"))
#     voxel_solid_path.append(os.path.join("../../What3D/extract_sample/", path, "models", "model_normalized.solid.binvox"))
#     voxel_surface_path.append(os.path.join("../../What3D/extract_sample/", path, "models", "model_normalized.surface.binvox"))

# start_time = time.time()
# iou_list = test_shapenet_sample(mesh_path, voxel_solid_path, voxel_surface_path, combine=False)
# print("elasped time: {}".format(((time.time() - start_time)/60)))



# Analyze the ShapeNet ground truth mesh and voxel compare 
# iou_list = np.load("../img/voxel/voxel_carving_test_gt_sample/iou_list.npy")

# iou_list.sort()
# middle = len(iou_list)//2
# # lower quartile
# lower_quartile = find_median(iou_list[:middle])
# # median
# median = find_median(iou_list)
# # upper quartile
# upper_quartile = find_median(iou_list[middle:])
# print(max(iou_list), min(iou_list))
# print("lower_quartile %.4f, median %.4f, upper_quartile %.4f" %(lower_quartile, median, upper_quartile))




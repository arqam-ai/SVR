import os
import sys
sys.path.append("../../")
import utils.binvox_rw as binvox_rw
from model.what3d.Oracle_NN import class_counter
from dataset.dataset import what3d_dataset_views
from voxel.compute_miou import plot_cords, convert_array_to_dataframe, convert_ptc_to_voxel
from utils.loss import evaluate_voxel_prediction
import argparse
import torch
import tqdm
from dataset.dataset import what3d_dataset_views
import cv2
import numpy as np
from utils.utils import Normalization
import time
import re 

def eval(args, normalize_gt=False):

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    mean_shape_list = []
    mean_shape_cord_list = []
    for index in tqdm.tqdm(range(0, 500), total=500):
        with open(os.path.join(args.data_basedir, args.cluster_path, "binvox_mean", "{}.binvox".format(index)), 'rb') as f:
            origin_mean_voxel = binvox_rw.read_as_3d_array(f).data
            mean_shape_list.append(origin_mean_voxel)
        with open(os.path.join(args.data_basedir, args.cluster_path, "binvox_mean", "{}.binvox".format(index)), 'rb') as f:
            origin_mean_cords = binvox_rw.read_as_coord_array(f).data
            mean_shape_cord_list.append(origin_mean_cords)

    #plot_cords(origin_mean_shape, title="clustering_mean_shape", save_file=os.path.join(args.save_path, "meanshape_{}.png".format(index)))
    pred_meanidx_list = []
    f = open(os.path.join(args.data_basedir, args.cluster_path, "test_cluster_predictions.txt"),"r")
    for x in f:
        pred_meanidx_list.append(int(x[:-1]))
    
    test_path_list = []
    img_path_list = []
    f = open(os.path.join(args.data_basedir, args.cluster_path, "test_images.txt"),"r")
    for x in f:
        test_path_list.append(x[:[i.start() for i in re.finditer('/', x)][1]])
        img_path_list.append(x[:-1])

    iou_list = []
    for test_index, instance_dir in enumerate(test_path_list):
        with open(os.path.join(args.data_basedir, args.voxel_path, instance_dir, "voxel_object.binvox"), "rb") as f:
            gt_object = binvox_rw.read_as_coord_array(f)
            gt = gt_object.data
        gt = gt.transpose(1, 0)

        if normalize_gt:
            gt_u2ball = Normalization(torch.from_numpy(gt), inplace=True, keep_track=False).normalize_unitL2ball()
            gt_u2ball = gt_u2ball.numpy()[0]
            gt_normed_voxel_cords, gt_normed_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(gt_u2ball))

        else:
            gt_normed_voxel_cords, gt_normed_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(gt))

        mean_shapeidx = pred_meanidx_list[test_index]
        prediction = mean_shape_list[mean_shapeidx]
        tmp_iou = evaluate_voxel_prediction(prediction, gt_normed_voxel)
        print("inferred instances %d/%d; iou:%.4f" % (test_index, len(test_path_list), tmp_iou))
        iou_list.append(tmp_iou)
        if test_index % 1000 == 0:
            image = cv2.imread(os.path.join(args.data_basedir, args.img_path, img_path_list[test_index])).transpose((2, 0, 1))
            image = np.expand_dims(image, axis=0)
            what3d_dataset_views.data_visualizer(torch.from_numpy(gt_normed_voxel_cords.transpose(1, 0)), 
                    torch.from_numpy(mean_shape_cord_list[mean_shapeidx].transpose(1, 0)), image, "clustering_normed_voxel", args.save_path, test_index, loss=tmp_iou, type="voxel_cords")
    
    np.save(os.path.join(args.save_path, "iou_list_norm_{}.npy".format(normalize_gt)), np.array(iou_list))
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



def find_median(List): # finds the median of a sorted_list
    number_of_data = len(List)
    if number_of_data % 2 == 0:
        median = (List[(number_of_data//2)]+List[(number_of_data//2-1)])/2
    else:
        median = List[(number_of_data//2)]
    return median


def test_normalization(args):
    test_path_list = []
    f = open(os.path.join(args.data_basedir, args.cluster_path, "test_images.txt"),"r")

    for x in f:
        test_path_list.append(x[:-6])
    
    for test_index, instance_dir in tqdm.tqdm(enumerate(test_path_list), total=len(test_path_list)):
        with open(os.path.join(args.data_basedir, args.voxel_path, instance_dir, "voxel_object.binvox"), "rb") as f:
            gt_object = binvox_rw.read_as_coord_array(f)
            gt = gt_object.data

        gt = gt.transpose(1, 0)
        gt_normed_voxel_cords, gt_normed_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(gt))

        gt_u2ball = Normalization(torch.from_numpy(gt), inplace=True, keep_track=False).normalize_unitL2ball()
        gt_u2ball = gt_u2ball.numpy()[0]
        gt_normed_voxel_cords_u2ball, gt_normed_voxel_u2ball = convert_ptc_to_voxel(convert_array_to_dataframe(gt_u2ball))
        #print(gt_normed_voxel_cords.shape, gt_normed_voxel_cords_u2ball.shape)
        print(np.allclose(gt_normed_voxel, gt_normed_voxel_u2ball))
        tmp_iou = evaluate_voxel_prediction(gt_normed_voxel, gt_normed_voxel_u2ball)
        print(tmp_iou)
if __name__ == '__main__':

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
    parser.add_argument("--class-path", dest="class_path", type=str, default='classes.txt', help="class name list")
    parser.add_argument("--cluster-path", dest="cluster_path", type=str, default='clustering', help='cluster path')
    parser.add_argument("--views",dest="views", type=str,default= '0',
                    help="five view for each instance")
    parser.add_argument("--save-path",dest="save_path", type=str,default= 'meanshape_vis',
                    help="")
    parser.add_argument("--normalize_gt", action="store_true", dest="normalize_gt", default=False, help="if unit2ball normalize gt")   
                 
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.script_folder = os.path.dirname(os.path.abspath(__file__))
    eval(args, args.normalize_gt)
    #test_normalization(args)


    
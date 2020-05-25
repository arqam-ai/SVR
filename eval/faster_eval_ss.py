import os
import sys
import argparse
import torch
import tqdm
import numpy as np
import random
import cv2
sys.path.append("../")
from model.clustering import class_counter
from utils.ptcloud_dataset_graph_three_shapes_yefan import image_preprocessing, compute_squared_EDM_method, \
												transform_mat, cal_pref, get_partition, silhouette
from utils.utils import check_exist_or_mkdirs
from utils.loss import ChamfersDistance3
from utils.plot_image_grid import visuaize_pts
from eval_ss import compute_ptcloud_dismatrix
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import logging

def ptcloud_sampling(test_ptcloud_set, origin_ptnum = 1024, sampled_ptnum = 256):
	"""randomly downsample pointcloud
	Params:
	------------------
	origin_ptnum: int
		origin points number in one cloud
	sampled_ptnum: int
		points number in one cloud after sampled
	Returns:
	------------------
	"""
	sampled_index = random.sample(range(origin_ptnum), sampled_ptnum)
	sampled_index = np.array(sampled_index)
	np.save("faster_ss_data/testset_256points_index.npy", sampled_index)
	downsampled_ptcloud = test_ptcloud_set[:,sampled_index,:]
	print(downsampled_ptcloud.shape)

	return downsampled_ptcloud

def main(args):

	## set up logger
	logger = logging.getLogger()
	file_log_handler = logging.FileHandler('Train_%d.log'%(args.experiment_number))
	logger.addHandler(file_log_handler)
	logger.setLevel('INFO')
	formatter = logging.Formatter()
	file_log_handler.setFormatter(formatter)
	stderr_log_handler = logging.StreamHandler(sys.stdout)
	logger.addHandler(stderr_log_handler)
	stderr_log_handler.setFormatter(formatter)

	check_exist_or_mkdirs(args.matrix_save_path)
	starter_time = time.time()
	## LOAD class info
	test_num, test_class_dic, test_class_list, test_fileid_list  = class_counter(data_basedir = args.data_basedir, 
			splits_path = args.splits_path, class_path = args.class_path, split_name ='test')
	class_list = list(test_class_dic.keys())    			## list of class name length = 55

	##LOAD downsampled point cloud
	test_ptcloud_set = np.load('faster_ss_data/testset_256points.npy')
	pt_criterion = ChamfersDistance3().to(args.device)
#	compute_ptcloud_dismatrix(X1 = , X2 = , distance_metric = , )
######################################################################################################
	##random sample test set 1000
#	random_index = np.array(random.sample(range(test_ptcloud_set.shape[0]), 1000), dtype=int)
#	randsamp_test_ptcloud_set = test_ptcloud_set[random_index]
#	print(random_index.shape)
#	print(randsamp_test_ptcloud_set.shape)
#	np.save('faster_ss_data/testset_ptcloud_random1000_index.npy', random_index)
#	np.save('faster_ss_data/testset_ptcloud_random1000.npy', randsamp_test_ptcloud_set)

	## LOAD test point cloud########################################
#	ptcloud = np.load(args.ptcloud_path)
#	test_ptcloud_set = ptcloud['test']

	## DOWNSAMPLE test point cloud
#	downsampled_ptcloud = ptcloud_sampling(test_ptcloud_set)
	### TEST if downsampled point cloud shape valid
#	visuaize_pts(downsampled_ptcloud[random.sample(range(10432), 40)], title = 'downsampled to 256 points',
#				rows = 5, cols = 8)
#	plt.savefig('../img/test_downsampled_ptcloud_256points.png')
	### Save this downsampled point cloud
#	np.save('faster_ss_data/testset_256points.npy', downsampled_ptcloud)
######################################################################################################

	'''
	## LOAD test image
	split_name = 'test'
	num_of_test_instance = test_ptcloud_set.shape[0]
	view = '0'
	test_image_set = np.zeros((num_of_test_instance, 3, 224, 224))
	test_sample_idx = 0
	for clname in tqdm.tqdm(class_list,total= len(class_list), desc = '%s img loading...'%split_name):
		f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt'%split_name),"r")
		for x in f:
			instance_id = x[:-1]
			image = cv2.imread(os.path.join(args.data_basedir, args.img_path, clname, instance_id, '%s.png'%view))
			image = image_preprocessing(image, imgsize = 224)
			test_image_set[test_sample_idx] = image
			test_sample_idx += 1
	
	assert test_sample_idx  == num_of_test_instance, "Loading image index gets error"
	'''
if __name__ == '__main__':

	parser = argparse.ArgumentParser(sys.argv[0])
	parser.add_argument('--data-basedir',type=str,default='../../What3D',
					help='')
	parser.add_argument('--img-path',type=str,default='renderings',
					help='')
	parser.add_argument('--splits-path',type=str,default='splits',
					help='')
	parser.add_argument('--class-path',type=str,default='classes.txt',
					help='')
	parser.add_argument('--clusterfile-path', type=str, 
					default='../what3d_clusters/cluster_assignments/',
					help='')
	parser.add_argument('--prediction-path', type=str, 
					default='../what3d_clusters/predictions.txt',
					help='')
	parser.add_argument("--ptcloud-path",type=str,
					default="../../What3D/ptcloud_0.npz",
					help=' ' )
	parser.add_argument("--matrix-save-path",type=str,
					default="distance_matrix",
					help=' ' )
	parser.add_argument("--experiment-number",type=int,
					default=1,
					help=' ' )

	args = parser.parse_args(sys.argv[1:])
	args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.script_folder = os.path.dirname(os.path.abspath(__file__))

	print(str(args))
	main(args)



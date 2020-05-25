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

def compute_img_dismatrix(X1, X2, distance_metric, title=None, results_dir=None, ifsave=False):
	"""return distance matrix between img set X1 and img set X2
	Params:
	----------------------------------
	X1: (N, 3, img_size, img_size) torch.tensor
		img set 1
	X2: (N, 3, img_size, img_size) torch.tensor
		img set 2
	distance_metric: func
		metric to measure the distance of two image
	ifsave: boolean
		if to save the distance matrix to disk
	title: string
		name of the saved matrix
	results_dir:string
		the path to save the distance matrix
		
	Returns:
	----------------------------------
	D: (ptnum, ptnum) torch.tensor
		distance matrix
	"""
	N = X1.shape[0]
	# initialize distance matrix
	D = torch.zeros([N, N])
	# iterate over one group of ptcloud
	for i in tqdm.tqdm(range(N), total = N, desc = 'image distance matrix'):	
		for j in range(i+1, N):
			D[i,j] = distance_metric(X1[i].unsqueeze(0), X2[j].unsqueeze(0))
			D[j,i] = D[i,j]
	if ifsave:
		D = D.cpu().numpy()
		np.save(os.path.join(results_dir,title), D)
		print("saved to " + os.path.join(results_dir, title)) 
	return D

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
#	test_ptcloud_set = np.load('faster_ss_data/testset_256points.npy')
	random1000_ptcloud_set = np.load('faster_ss_data/testset_ptcloud_random1000.npy')
	
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


	train_ptcloud_set = random1000_ptcloud_set
	train_sample_index = np.load('faster_ss_data/testset_ptcloud_random1000_index.npy')
	## LOAD train image
	split_name = 'test'
	num_of_train_instance = 10432
	view = '0'
	train_image_set = np.zeros((num_of_train_instance, 3, 224, 224))
	train_sample_idx = 0
	for clname in tqdm.tqdm(class_list,total= len(class_list), desc = '%s img loading...'%split_name):
		f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt'%split_name),"r")
		for x in f:
			instance_id = x[:-1]
			image = cv2.imread(os.path.join(args.data_basedir, args.img_path, clname, instance_id, '%s.png'%view))
			image = image_preprocessing(image, imgsize = 224)
			train_image_set[train_sample_idx] = image
			train_sample_idx += 1
	
	assert train_sample_idx  == num_of_train_instance, "Loading image index gets error"

	## Convert numpy array to torch tensor
	train_image_set = torch.from_numpy(train_image_set)
	train_ptcloud_set = torch.from_numpy(train_ptcloud_set)
	

	sample_train_ptcloud_set = train_ptcloud_set.to(args.device)
	sample_train_image_set = train_image_set[train_sample_index].to(args.device)
	## compute distance matrix 
	pt_criterion = ChamfersDistance3().to(args.device)
	img_criterion = nn.L1Loss(reduction="sum").to(args.device)
	train_pt_matrix = compute_ptcloud_dismatrix(X1 = sample_train_ptcloud_set, X2 = sample_train_ptcloud_set, distance_metric = pt_criterion, 
								title = '%s_pt_similarity_matrix_%d.npy'% ('train',args.experiment_number), results_dir = args.matrix_save_path, ifsave = True)
	train_img_matrix = compute_img_dismatrix(X1 = sample_train_image_set, X2 = sample_train_image_set, distance_metric = img_criterion, 
								title = '%s_img_similarity_matrix_%s.npy'%('train',args.experiment_number), results_dir = args.matrix_save_path, ifsave = True)
	
	## normalize matrix   
	train_pt_matrix_tr = transform_mat(train_pt_matrix)     # -e^(x/max(x)) then fill the diagonal with 0
	train_img_matrix_tr = transform_mat(train_img_matrix)
	
	## get partition
	### point cloud, calculate affinity propagation parameter: preference
	part_preference = cal_pref(train_pt_matrix_tr)          # in increasing order, float number in first 10% position in this matrix
	### affinity propagation
	train_pt_part = get_partition(train_pt_matrix_tr, preference = part_preference)
	### image, calculate affinity propagation parameter: preference
	part_preference = cal_pref(train_img_matrix_tr)
	### affinity propagation
	train_img_part = get_partition(train_img_matrix_tr, preference = part_preference)

	## silhouette score
	pt_ss = silhouette(train_pt_matrix, train_pt_part)
	img_ss = silhouette(train_img_matrix, train_img_part)

	## report 
	logger.info('Experiment No.{} point cloud silhouette: {}'.format(args.experiment_number, pt_ss))
	logger.info('Experiment No.{} img silhouette: {}'.format(args.experiment_number, img_ss))
	logger.info('Time:{:3} seconds'.format(time.time() - starter_time))


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



'''
T-SNE comparsion between Clustering Method and Autoencoder

Author: Yefan 
'''
import os
import numpy as np
import math
import tqdm
import sys
import time
sys.path.append("../")
import matplotlib.pyplot as plt
import argparse
from MulticoreTSNE import MulticoreTSNE as TSNE
from model.clustering import class_counter


def tsne(codewords, label, num_of_class):
	"""plot the T-SNE based on codewords and label
	Params:
	------------------
	codewords: (num_of_samples, dims_of_feature) numpy array
		codewords to be dimension reduction
	label: (num_of_samples,) numpy array
		data label
	num_of_class: int
		number of class

	Returns:
	------------------
	None
	"""
	starter_time = time.time()
	embeddings = TSNE(n_jobs=4).fit_transform(codewords)
	vis_x = embeddings[:, 0]
	vis_y = embeddings[:, 1]
	plt.scatter(vis_x, vis_y, c=label, cmap=plt.cm.get_cmap("jet", num_of_class), marker='.', s = 100)
	plt.colorbar(ticks=range(num_of_class))
	plt.clim(-0.5, 9.5)
	print('TSNE TIME: {} seconds'.format(time.time()-starter_time))
	#plot_embedding(tsne_results, colors, title)
	

def load_pred(data_basedir, splits_path, class_path, split_name, results_path, results_name, 
						target_classnum = 9, total_num = 10432, origin_test_batch = 200, exp_name = 'autoencoder'):
	"""load prediction point cloud and return specific number of class 

	Params:
	------------------
	data_basedir  : string
		path to data base folder
	splits_path   : string
		data folder name to contain splits info
	class_path    : string
		class info file name
	split_name    : string
		"train" or "test"
	results_path  : string
		path to experiment results 
	results_name  : string
		name of result file
	target_classnum  : int
		number of class should be included 
	total_num        : int
		number of instances should be included
	origin_test_batch: int
		number of instances in one npy file
	exp_name         : string
		experiment name 
	Returns:
	---------------------
	pred			: 	numpy array (ptnum, 1024, 3)
		prediction of point cloud
	target_index	:   list 
		list of target index
	sorted_class	:   list
		list of class name
	target_index_dic:   dictionary
		{class_name: [target_index]}
	"""
	## Load class info
	instance_num, class_dic, data_class, fileid_list = class_counter(data_basedir = data_basedir, splits_path = splits_path, 
										class_path = class_path, split_name= split_name)
	## find N classes which have most instances
	sorted_class = sorted(class_dic.items(),key=lambda item:item[1],reverse = True)[:target_classnum]
	sorted_class = [item[0] for item in sorted_class]
	## get the class name 
	sorted_class.sort()
	print(sorted_class)
	print(len(data_class))
	## find the sample index of desired class sample
	target_index = []                                   						# all the desired sample index list
	target_index_dic = {}														# desired sample dictionary {class_name: []}
	## iterate through the class name list
	for target_class in sorted_class:											# initialize the dictionary
		target_index_dic[target_class] = []										
	## load the desired sample index by iterate through class name
	for target_class in sorted_class:        						
		target_index += [i for i,x in enumerate(data_class) if x == target_class]	 
		target_index_dic[target_class] += [i for i,x in enumerate(data_class) if x == target_class]

	## LOAD the file(npy format) from disk
	### find the number of file should be loaded 
	num_of_file = math.ceil(total_num/origin_test_batch)
	### iterate through all the npy file 
	for idx in tqdm.tqdm(range(num_of_file)):
		if exp_name == 'autoencoder':
			postidx = '%03d'%idx + '.npy'
			tmp_pred = np.load(os.path.join(results_path, results_name + postidx))
		elif exp_name == 'clustering':
			tmp_pred = np.load(os.path.join(results_path, results_name))
		if idx == 0:
			pred = tmp_pred
		else:
			pred = np.concatenate((pred, tmp_pred), axis=0)
	print(pred.shape)
	return pred, target_index, sorted_class, target_index_dic


def main(args):

	if args.exp_name == 'autoencoder':
		## SET up path to autoencoder experiment
		results_name, exp_name, log = 'fineptcloud_', 'resnet_nolabel', 'results_resnet_nolabel_lr0003_exp3'
		origin_test_batch = 200
		results_path = '../experiment/{}/{}/final_vis'.format(exp_name, log)
	elif args.exp_name == 'clustering':
		## SET up path to clustering experiment
		results_name, exp_name = 'clustering_prediction.npy', 'cluster_matrix'
		results_path = '../experiment/{}'.format(exp_name)
		origin_test_batch = 10432

	##Load prediction
	pred_pt, target_index, sorted_class, target_index_dic = load_pred(data_basedir = args.data_basedir, 
					splits_path = args.splits_path, class_path = args.class_path, split_name = 'test',
					results_path = results_path, results_name = results_name, 
					target_classnum = 9, total_num = 10432, origin_test_batch = origin_test_batch, exp_name = args.exp_name)
	
	##slicing the entire prediction with specific class sample index
	sampled_pred_pt = pred_pt[target_index]
	###Iterize through desired class name to generate numpy array label
	for idx, class_name in enumerate(sorted_class):
		if idx == 0:												           ## first class, initialize a numpy array filled with 0
			label = np.zeros(len(target_index_dic[class_name]), dtype=int)
		elif idx > 0:												           ## for next class, class label = 1,2,3....
			next_class_label = idx * np.ones(len(target_index_dic[class_name]), dtype=int)
			label = np.concatenate((label, next_class_label), axis = 0)		   ## label shape (sample_number,)
	assert sampled_pred_pt.shape[0] == label.shape[0], "size of sampled prediction mismatches the label size"
	
	## flatten a pointcloud to a codeword
	num_of_sample = sampled_pred_pt.shape[0]
	sampled_codewords = sampled_pred_pt.reshape(num_of_sample, -1)

	## plot tsne
	fig, ax = plt.subplots(figsize=(30, 30))
	tsne(codewords = sampled_codewords, label = label, num_of_class = 9)
	plt.savefig(os.path.join('../img','tsne_clustering.png'))


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
						default="cluster_matrix",
						help=' ' )
	parser.add_argument("--exp-name",type=str,
						default="clustering",     # another is autoencoder/clustering
						help='which experiment to do tsne' )

	args = parser.parse_args(sys.argv[1:])
	print(str(args))
	main(args)
#!/usr/bin/python
'''
plot interpolatoin curve

'''
import torch
import os
import sys
import glob
import time
import optparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import json
from utils.loss import ChamferDistance
import torch.nn as nn
from torchvision import transforms, utils
import tqdm
from sklearn import preprocessing
from sklearn.cluster import AffinityPropagation 
from sklearn.metrics import silhouette_score
import pandas as pd
import networkx as nx
import community
import math
import random
from sklearn.manifold import TSNE
#from sample_three_clusters import sample_three_clusters, sample_three_clusters_yefan
from collections import Counter

def silhouette(matrix, partition):
    part = [partition[i] for i in range(matrix.shape[0])]	
    return silhouette_score(matrix, part, metric='precomputed')

def image_preprocessing(image, imgsize = 224):

	convertor  = transforms.ToPILImage()
	if isinstance(image, np.ndarray):
		image = convertor(image)
	image = transforms.functional.resize(image, [imgsize, imgsize])
	image = np.array(image) / 255.0
	image = image.transpose((2, 0, 1))

	return image

def instance_load(data_dir, data, idx, imgsize = 224):

	tmp_ptcloud = np.load(os.path.join(data_dir,data[idx]['ptcloud_path']))
	tmp_image   = cv2.imread(os.path.join(data_dir,data[idx]['isometric_path']))
	tmp_image   = image_preprocessing(tmp_image, imgsize)
	tmp_ptcloud = np.expand_dims(tmp_ptcloud, axis=0)
	tmp_image = np.expand_dims(tmp_image, axis=0)

	return tmp_ptcloud, tmp_image

def json_load(data_dir, json_file):
	json_file = os.path.join(data_dir, json_file)
	data_path = open(json_file, 'r')
	data = json.load(data_path)
	return data

def load_gt(json_file, data_dir):
	'''
	Input:
	     json_file
	     dir
	Output:
	     gt_ptcloud
	     gt_image
	'''
	json_file = os.path.join(data_dir, json_file)
	data_path = open(json_file, 'r')
	data = json.load(data_path)
	#data.sort(key=lambda x:x['shapekey_value'])


	
	for i in tqdm.tqdm(range(len(data)),total= len(range(len(data))), desc = "load ground truth"):
		tmp_ptcloud, tmp_image = instance_load(data_dir, data, i, imgsize = 224)
		if i == 0:
			gt_ptcloud = tmp_ptcloud
			gt_image = tmp_image
			log_image = tmp_image
			log_ptcloud = tmp_ptcloud
		else:
			gt_ptcloud = np.concatenate((gt_ptcloud,tmp_ptcloud),axis=0)
			gt_image = np.concatenate((gt_image,tmp_image),axis=0)

	gt_ptcloud = torch.from_numpy(gt_ptcloud).float()
	gt_image = torch.from_numpy(gt_image).float()
	print("gt_ptcloud", gt_ptcloud.shape)
	print("gt_image", gt_image.shape)
	return gt_ptcloud, gt_image


def load_pred(data_dir, result_dir, test_json, indexlist, total_num = 2021, origin_test_batch = 100):
	json_file = os.path.join(data_dir, test_json)
	data_path = open(json_file, 'r')
	data = json.load(data_path)
	origin_test_batch = float(origin_test_batch)
	num_of_npy = int(math.ceil(total_num/origin_test_batch))
	print(num_of_npy)
	#if args.sample_test_json:
	#	num_of_npy = min(num_of_npy, 5)
        
	interp_list = list(range(num_of_npy))
    
	for i in tqdm.tqdm(range(len(interp_list)), total = len(range(len(interp_list))), desc = "load pred"):
		interp = interp_list[i]
		tmp_ptcloud = np.load(os.path.join(result_dir, 'fineptcloud_%03d.npy' % interp))
		tmp_image = np.load(os.path.join(result_dir, 'img_%03d.npy' % interp))
		tmp_codeword = np.load(os.path.join(result_dir, 'codeword_%03d.npy' % interp))
	
		if i == 0:
			pred_ptcloud = tmp_ptcloud
			pred_image = tmp_image
			pred_codeword = tmp_codeword
		else:
			pred_ptcloud = np.concatenate((pred_ptcloud,tmp_ptcloud),axis=0)
			pred_image = np.concatenate((pred_image,tmp_image),axis=0)
			pred_codeword = np.concatenate((pred_codeword, tmp_codeword),axis = 0)

	pred_ptcloud = torch.from_numpy(pred_ptcloud).float()
	pred_image = torch.from_numpy(pred_image).float()
	print("The shape of the original test set is")
	print(pred_ptcloud.shape)


			#if ifplot:
			#	plt.subplot(1, 3, 1)
			#	plt.scatter(shapekeys_this_shape[:,0], shapekeys_this_shape[:,1], label =shape, s=100)
        
		#samplelist = [data[idx] for idx in indexlist]
		#print(indexlist)
    
		#pred_codeword = torch.from_numpy(pred_codeword).float()
    
	pred_ptcloud = pred_ptcloud[indexlist]
	print("The shape of subsampled test set is")
	print(pred_ptcloud.shape)
        
	return pred_ptcloud, pred_image, interp_list , pred_codeword

def evaluate(ptcloud, image, gt_ptcloud, gt_image, pt_criterion, img_criterion):
	pt_origin_loss = []
	pt_target_loss = []
	img_origin_loss = []
	img_target_loss = []
	for i in tqdm.tqdm(range(ptcloud.shape[0]),total= len(range(ptcloud.shape[0])),desc = "Evaluation"):
		pt_origin_loss.append(pt_criterion(gt_ptcloud[0].unsqueeze(0), ptcloud[i].unsqueeze(0)).item())
		pt_target_loss.append(pt_criterion(gt_ptcloud[-1].unsqueeze(0), ptcloud[i].unsqueeze(0)).item())
		img_origin_loss.append(img_criterion(gt_image[0].unsqueeze(0), image[i].unsqueeze(0)).item())
		img_target_loss.append(img_criterion(gt_image[-1].unsqueeze(0), image[i].unsqueeze(0)).item())

	return (img_origin_loss, pt_origin_loss), (img_target_loss, pt_target_loss)

def plot_img_ptcloud(args, gt_origin, gt_target, pred_origin, pred_target, train_idx, whole_idx, plot_fname):


	fig = plt.figure(figsize=(30,22))
	ax1 = fig.add_subplot(2, 1, 1)
	ax1.plot([gt_origin[0][i] for i in whole_idx], [gt_origin[1][i] for i in whole_idx],'c.-', label='Whole Set - GT', linewidth=1)
	ax1.scatter([gt_origin[0][i] for i in train_idx], [gt_origin[1][i] for i in train_idx], c= 'k',marker = 'o', label='Train Set - GT')
	ax1.plot(pred_origin[0], pred_origin[1], 'ro-', label='Test Set - Inferred', linewidth=1)
	ax2 = fig.add_subplot(2, 1, 2)
	ax2.plot([gt_target[0][i] for i in whole_idx], [gt_target[1][i] for i in whole_idx], 'c.-', label='Whole Set - GT', linewidth=1)
	ax2.scatter([gt_target[0][i] for i in train_idx], [gt_target[1][i] for i in train_idx], c= 'k', marker = 'o', label='Train Set - GT')
	ax2.plot(pred_target[0], pred_target[1], 'ro-', label='Test Set - Inferred', linewidth=1)
	ax1.legend(fontsize=30)
	ax1.set_title('origin_shape',fontsize=30)
	ax2.legend(fontsize=30)
	ax2.set_title('target_shape',fontsize=30)
	ax1.set_xlabel('img_L1loss',fontsize=30)
	ax1.set_ylabel('pt_chamferloss',fontsize=30)
	ax2.set_xlabel('img_L1loss',fontsize=30)
	ax2.set_ylabel('pt_chamferloss',fontsize=30)
	plt.savefig(os.path.join(args.save_dir, plot_fname))

def plot_origin_target(args, gt_origin, gt_target, pred_origin, pred_target, train_idx, whole_idx, plot_fname):

	plt.figure(figsize=(30,22))
	plt.plot([gt_origin[1][i] for i in whole_idx], [gt_target[1][i] for i in whole_idx], 'c.-', label='Whole Set - GT', linewidth=1)
	plt.plot(pred_origin[1], pred_target[1], 'ro-', label='Test Set - Inferred', linewidth=1)
	plt.scatter([gt_origin[1][i] for i in train_idx], [gt_target[1][i] for i in train_idx], c= 'k', marker = 'o', label='Train Set - GT')
	plt.legend(fontsize=30)
	plt.title('Cubeloss vs Sphereloss', fontsize=30)
	plt.xlabel('sphere loss',fontsize=30)
	plt.ylabel('cube loss',fontsize=30)
	plt.savefig(os.path.join(args.save_dir, plot_fname))

def save_data(gt_origin, gt_target, pred_origin, pred_target, results_dir):
	interp_data = {}
	interp_data["gt_origin"] = gt_origin
	interp_data["gt_target"] = gt_target
	interp_data["pred_origin"] = pred_origin
	interp_data["pred_target"] = pred_target
	np.save(os.path.join(results_dir,"interp_data.npy"), interp_data)
	print("saved to" + os.path.join(results_dir,"interp_data.npy")) 

def load_data(dir):
	interp_data = np.load(os.path.join(dir,"interp_data.npy")).item()
	gt_origin, gt_target = interp_data["gt_origin"], interp_data["gt_target"]
	pred_origin, pred_target = interp_data["pred_origin"], interp_data["pred_target"]

	return gt_origin, gt_target, pred_origin, pred_target

def load_minloss(args, density_list, ratio_list):
	testloss_list = []
	for item in density_list:
		for ratio in ratio_list:
			if ratio == 1:           
				test_loss = np.load("../experiment/cubesphere%d_1.27/logs/stats_finecd_epochtest.npz" % item)
			elif ratio >= 0.1:
				test_loss = np.load("../experiment/cubesphere%d_1.27/cubesphere%.1f_1.29/logs/stats_finecd_epochtest.npz" % (item, ratio))
			else:
				test_loss = np.load("../experiment/cubesphere%d_1.27/cubesphere%.2f_1.29/logs/stats_finecd_epochtest.npz" % (item, ratio))
			testloss_list.append(min(test_loss["iter_loss"][:,1]))

	return testloss_list

def plot_density(args):
	testloss_list = load_minloss(args, [1000], [1, 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.02])
	plt.figure(figsize=(14,11))
	plt.plot([1, 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.02], testloss_list, 'co-', label='Minimum Test Loss', linewidth=1)
	plt.xticks([1, 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.02])
	plt.xlabel('Dataset density')
	plt.ylabel('Chamfer Loss')
	plt.title('Test Set average chamfer vs dataset density')
	plt.legend(fontsize=30)
	plt.savefig(os.path.join(args.save_dir, 'density_ratio.png'))




def matrix_to_graph_norm(matrix, threshold):
	
	#Input:

	#Output:

	#normalize the matrix before convert to pandas dataframe

	corr=pd.DataFrame(matrix)
	links = corr.stack().reset_index()
	links.columns = ['var1', 'var2','value']
	links_filtered=links.loc[(links['var1'] != links['var2']) ]
	max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
	normed_value = links_filtered[['value']].apply(max_min_scaler)
	links_filtered = links_filtered.drop("value", axis=1)
	links_filtered['value'] = normed_value
	links_filtered = links_filtered.loc[(links_filtered['value'] < threshold)]
#	print(links_filtered)
	G=nx.from_pandas_edgelist(links_filtered, source ='var1', target = 'var2', edge_attr ='value', create_using = nx.Graph())
	if G.size() == 0:
		cluster_coeff = 0
	else:
		cluster_coeff = nx.algorithms.cluster.average_clustering(G)
#		cluster_coeff = nx.transitivity(G)

#	nx.draw(G, with_labels=False, node_color='orange', node_size=4, edge_color='black', linewidths=1, font_size=3)
#	print('Number of graph node', G.number_of_nodes())
#	print('Number of edges:', G.size())
#	print('Clustering Coefficient %f'%nx.algorithms.cluster.average_clustering(G)
	return (G, cluster_coeff)


def matrix_to_graph_origin(matrix, threshold):

	#Input:

	#Output:

	#normalize the matrix before convert to pandas dataframe

	corr=pd.DataFrame(matrix)
	links = corr.stack().reset_index()
	links.columns = ['var1', 'var2','value']
	link_test  = links
	links_filtered=links.loc[(links['value'] < threshold) & (links['var1'] != links['var2']) ]

#	print(links_filtered)
	G=nx.from_pandas_edgelist(links_filtered, source ='var1', target = 'var2', edge_attr ='value', create_using = nx.Graph())
	if G.size() == 0:
		cluster_coeff = 0
		mod = 0
	else:
#		cluster_coeff = nx.algorithms.cluster.average_clustering(G)
		cluster_coeff = nx.transitivity(G)
		mod = graph_metric(G,'modularity')

#	nx.draw(G, with_labels=False, node_color='orange', node_size=4, edge_color='black', linewidths=1, font_size=3)
#	print('Number of graph node', G.number_of_nodes())
#	print('Number of edges:', G.size())
#	print('Clustering Coefficient %f'%nx.algorithms.cluster.average_clustering(G)
	return cluster_coeff

def experiment_density():
	
	if args.type == 'results':
		ptcloud, image, _	= load_pred(args.data_dir, args.results_dir, args.test_json)
	elif args.type == 'train':
		ptcloud, image = load_gt(args.train_json, args.data_dir)
	elif args.type == 'test':
		ptcloud, image = load_gt(args.test_json, args.data_dir)
	
	pt_criterion = ChamfersDistance3()

	if args.first_plot:
		ptcloud = ptcloud.cuda()
		pt_criterion = pt_criterion.cuda()
		matrix = compute_squared_EDM_method(ptcloud, pt_criterion, True, '%s_similarity_matrix.npy'%args.type, args.similarity_dir)
	else:
#		matrix = np.load(os.path.join(args.similarity_dir,'train_similarity_matrix.npy'))	
		train_matrix_random = np.load(os.path.join("../experiment/cubesphere1000_1.27/cubesphere0.143_2.1/results",'train_similarity_matrix.npy'))
		train_matrix_10by10 = np.load(os.path.join( "../experiment/cubesphere1000_1.27/cubesphere1by100_2.1/results",'train_similarity_matrix.npy'))
		train_matrix_2by50 = np.load(os.path.join( "../experiment/cubesphere1000_1.27/cubesphere2by50_2.1/results",'train_similarity_matrix.npy'))
		train_matrix_1by100 = np.load(os.path.join( "../experiment/cubesphere1000_1.27/cubesphere1by100_2.1/results",'train_similarity_matrix.npy'))
		train_matrix_3by33 = np.load(os.path.join( "../experiment/cubesphere1000_1.27/cubesphere3by33_2.1/results",'train_similarity_matrix.npy'))
		train_matrix_4by25 = np.load(os.path.join( "../experiment/cubesphere1000_1.27/cubesphere4by25_2.1/results",'train_similarity_matrix.npy'))
		train_matrix_5by20 = np.load(os.path.join( "../experiment/cubesphere1000_1.27/cubesphere5by20_2.1/results",'train_similarity_matrix.npy'))
		test_matrix = 		np.load(os.path.join("../experiment/cubesphere1000_1.27/cubesphere0.143_2.1/results", 'test_similarity_matrix.npy'))
		results_matrix_random = np.load(os.path.join("../experiment/cubesphere1000_1.27/cubesphere0.143_2.1/results",'results_similarity_matrix.npy'))
		results_matrix_10by10 = np.load(os.path.join("../experiment/cubesphere1000_1.27/cubesphere10by10_2.1/results",'results_similarity_matrix.npy'))
		results_matrix_2by50 = np.load(os.path.join("../experiment/cubesphere1000_1.27/cubesphere2by50_2.1/results", 'results_similarity_matrix.npy'))
		results_matrix_1by100 = np.load(os.path.join("../experiment/cubesphere1000_1.27/cubesphere1by100_2.1/results", 'results_similarity_matrix.npy'))
		results_matrix_3by33 = np.load(os.path.join("../experiment/cubesphere1000_1.27/cubesphere3by33_2.1/results", 'results_similarity_matrix.npy'))
		results_matrix_4by25 = np.load(os.path.join("../experiment/cubesphere1000_1.27/cubesphere4by25_2.1/results", 'results_similarity_matrix.npy'))
		results_matrix_5by20 = np.load(os.path.join("../experiment/cubesphere1000_1.27/cubesphere5by20_2.1/results", 'results_similarity_matrix.npy'))

	average_chamfer = [0, 0.0714, 0.2252, 0.1315, 0.0829, 0.1258, 0.2039, 0.0861]
	metric = 'modularity'
	name = 'average_loss'
	threshold = 0.92  
	train_matrix_set = [train_matrix_random,train_matrix_random, train_matrix_1by100,train_matrix_2by50,train_matrix_3by33,train_matrix_4by25,train_matrix_5by20,train_matrix_10by10]
	test_matrix_set = [test_matrix, results_matrix_random, results_matrix_1by100, results_matrix_2by50,results_matrix_3by33, results_matrix_4by25,results_matrix_5by20,results_matrix_10by10]
	label_set = ['random_gt', 'random_results', '1by100','2by50','3by33','4by25','5by20','10by10']
	plt.figure(figsize=(15,15))
	for i in range(len(train_matrix_set)):
#		plt.scatter(matrix_to_graph_exp(train_matrix_set[i],threshold, metric), 
#			matrix_to_graph_exp(test_matrix_set[i],threshold, metric), label = label_set[i])
		plt.scatter(average_chamfer[i], matrix_to_graph_exp(test_matrix_set[i],threshold, metric), label = label_set[i])
	plt.legend(fontsize=15)
	plt.title('clustering performance vs reconstruction loss')
	plt.savefig(os.path.join(args.save_dir,'%s.png'%name))
	

def experiment_level():

	pt_criterion = ChamfersDistance3() 
	img_criterion = nn.L1Loss(reduction="sum")
	train_pt_mod = []
	pred_pt_mod = []
	train_img_mod = []
	tsne = TSNE(n_components=2, init='pca', random_state=0)


	for sparse in range(7, 8):
		train_json = 'cluster_interp_5by20_%d.json'%sparse
		results_dir = "../experiment/cubesphere1000_1.27/cubesphere5by20_%d/results/final_vis"%sparse
		similarity_dir = "../experiment/cubesphere1000_1.27/cubesphere5by20_%d/results"%sparse
		
		## If true, then load test results, compute distance matrix 
		## If false, then load distance matrix computed before  
		if args.first_plot:
			train_ptcloud, train_image = load_gt(train_json, args.data_dir)
#			pred_ptcloud, pred_image, _ , pred_codeword = load_pred(args.data_dir, results_dir, args.test_json)
			train_ptcloud, pred_ptcloud = train_ptcloud.cuda(), pred_ptcloud.cuda()
			train_image = train_image.cuda()
			pt_criterion = pt_criterion.cuda()
			img_criterion = img_criterion.cuda()
			train_pt_matrix = compute_squared_EDM_method(train_ptcloud, pt_criterion, True, 
											'%s_pt_similarity_matrix.npy'%'train', similarity_dir)
#			pred_pt_matrix = compute_squared_EDM_method(pred_ptcloud, pt_criterion, True, 
#											'%s_pt_similarity_matrix.npy'%'results', similarity_dir)
			train_img_matrix = compute_squared_EDM_method(train_image, img_criterion, True, 
											'%s_img_similarity_matrix.npy'%'train', similarity_dir)
		else:
			pred_ptcloud, pred_image, _ , pred_codeword = load_pred(args.data_dir, results_dir, args.test_json)
			train_pt_matrix = np.load(os.path.join(similarity_dir, '%s_pt_similarity_matrix.npy'%'train'))
			print(np.max(train_pt_matrix))
			print(np.min(train_pt_matrix))
			pred_pt_matrix = np.load(os.path.join(similarity_dir, '%s_pt_similarity_matrix.npy'%'results'))
			print(np.max(pred_pt_matrix))
			print(np.min(pred_pt_matrix))
			train_img_matrix = 	np.load(os.path.join(similarity_dir, '%s_img_similarity_matrix.npy'%'train'))



		## compute modularity

		#train_pt_mod += [matrix_to_graph_exp(train_pt_matrix, None, 'modularity')]
		#pred_pt_mod += [matrix_to_graph_exp(pred_pt_matrix, None, 'modularity')]
		#train_img_mod += [matrix_to_graph_exp(train_img_matrix, None, 'modularity')]

	'''
		## TSNE
		tsne_results = tsne.fit_transform(pred_codeword)
		plt.figure()
		plot_embedding(tsne_results, None, 'dense_%d'%sparse)
		plt.savefig(os.path.join(args.save_dir,'tsne_%d.png'%sparse))
	
	## Plot modularity
	plt.figure()
	for i in range(0,7):
		plt.scatter(train_img_mod[i], pred_pt_mod[i],label = '%d'%(i+1))
	plt.plot(train_img_mod, pred_pt_mod)
	plt.legend()
	plt.title('predptmod_vs_train_img')
	plt.savefig(os.path.join(args.save_dir,'%s.png'%'predptmod_vs_train_img'))

	plt.figure()
	for i in range(0,7):
		plt.scatter(train_pt_mod[i], pred_pt_mod[i],label = '%d'%(i+1))
	plt.plot(train_pt_mod, pred_pt_mod)
	plt.legend()
	plt.title('pred_pt_mod_vs_train_pt')
	plt.savefig(os.path.join(args.save_dir,'%s.png'%'pred_pt_mod_vs_train_pt'))
	'''
'''
def get_partition(matrix):


    cl = AffinityPropagation(affinity='precomputed')

    cl.fit(matrix)
    partition = cl.labels_
    return partition
'''

def get_partition(matrix, preference, damping=0.75):


    cl = AffinityPropagation(damping=damping, affinity='precomputed',preference = preference)
    

    cl.fit(matrix)
    partition = cl.labels_
    return partition

def compute_squared_EDM_method(X, distance_metric, ifsave, reorder, title, results_dir, num_pershape, part_preference):
	# determin dimensions of data matrix
	N = X.shape[0]
	# initialize squared EDM D
	D = np.zeros([N, N])	
	# iterate over upper triangle of D
	for i in tqdm.tqdm(range(N),total= len(range(N)), desc = "similarity matrix"):	
		for j in range(i+1, N):
			D[i,j] = distance_metric(X[i].unsqueeze(0), X[j].unsqueeze(0))
			D[j,i] = D[i,j]    #*1
	
	part_new = np.repeat(range(3),num_pershape)
	if reorder:
		D_0=D[0,:]
		orders = sorted(range(D_0.shape[0]), key=lambda x:D_0[x])
		D = D[orders,:]
		D = D[:,orders]
		# D_0=D[0,:]
		# part = get_partition(D)
		# g_s = []
		# ord_dict = {}
		# for i in range(np.amax(part)+1):
		# 	select = D_0[part==i]
		# 	ord_dict[i] = [j for j in range(part.shape[0]) if part[j]==i]
		# 	g_s.append(np.mean(select))
		# g_order = sorted(range(len(g_s)), key=lambda x:g_s[x])
		# order_new = []
		# for i in g_order:
		# 	order_new += ord_dict[i]
		# D_reorder = D[orders,:]
		# D_reorder = D_reorder[:,orders]
		# dist_groups = [0]
		# group_count = dict(Counter(part))
		# g_sort = [group_count[i] for i in range(np.amax(part)+1)]
		# g_start = np.cumsum(g_sort)
		# D_0 = D_reorder[:g_start[0],:]
		# for i in range(np.amax(part)):
		# 	D_0i = D_0[:,g_start[i]:g_start[i+1]]
		# 	dist_groups.append(np.mean(D_0i))
		# group_order = sorted(range(len(dist_groups)), key=lambda x:dist_groups[x])
		# mapping = dict(zip(group_order, range(len(group_order))))
		# part_new = [mapping[part[i]] for i in range(part.shape[0])]
		# order_new = sorted(range(D.shape[0]), key=lambda x:part_new[x])
		# D = D[order_new,:]
		# D = D[:,order_new]
		part_new = get_partition(D, preference = part_preference)
	if ifsave:
		np.save(os.path.join(results_dir,title), D)
		print("saved to" + os.path.join(results_dir, title)) 
		np.save(os.path.join(results_dir,title[:-4]+'_part.npy'), part_new)
		print("saved to" + os.path.join(results_dir,title[:-4]+'_part.npy'))
	return D, part_new


def matrix_to_graph_exp(matrix, threshold, metric, temperature):

	matrix = np.exp(- temperature * matrix)
        
	if metric == 'modularity':
		G = nx.from_numpy_matrix(matrix)#, create_using=nx.Graph)
		#print(nx.get_edge_attributes(G,'weight'))
		#nx.draw(G, pos, cmap=plt.cm.Blues)
		#plt.savefig(os.path.join("../img/cluster/triangle", ))
		return graph_metric(G, 'modularity')

	elif metric == 'global_clustering':

		corr=pd.DataFrame(matrix)
		links = corr.stack().reset_index()
		links.columns = ['var1', 'var2','value']
		links_filtered=links.loc[(links['value'] > threshold) & (links['var1'] != links['var2']) ]
		G=nx.from_pandas_edgelist(links_filtered, source ='var1', target = 'var2', edge_attr ='value', create_using = nx.Graph())
		return graph_metric(G, 'global_clustering')
	
    
def graph_metric(G, metric):
	if metric == 'modularity':
		partition = community.best_partition(G)
		mod = community.modularity(partition, G)

		return mod
	if metric == 'global_clustering':
		if G.size() == 0:
			cluster_coeff = 0
		else:
#			cluster_coeff = nx.algorithms.cluster.average_clustering(G)
			cluster_coeff = nx.transitivity(G)
		return cluster_coeff

def plot_embedding(data, label, title):

	#x_min, x_max = np.min(data, 0), np.max(data, 0)
	#data = (data - x_min) / (x_max - x_min)
	for i in range(data.shape[0]):
		plt.scatter(data[i, 0], data[i, 1],c='b',s = 20)
	plt.xticks([])
	plt.yticks([])
	plt.title(title,fontsize = 30)

def cal_pref(mat):
    pc = 10
    return np.percentile(mat, pc)  
	  
def transform_mat(mat):
    mat = -np.exp(mat/np.amax(mat))
    np.fill_diagonal(mat, 0)
    return mat

def main(args):

	pt_criterion = ChamfersDistance3() 
	img_criterion = nn.L1Loss(reduction="sum")
	train_pt_mod = []
	pred_pt_mod = []
	train_img_mod = []

	num_pershape = 16         # num of training instances for each shape 
	args.first_plot = False    # if compute the distance matrix 
	ifsavemat = False          # if saving matrix in computing distance matrix 
	ifsamept = False          # if eval the results of training set only three point cloud 
	ifconverge = True        # if eval the results of training set only three point cloud but have decrease lr more times in training
	
	num_for_testshape = 100   # num of test instances for each shape 

	AP_preference = 0.56      # AP parameter  to tune 


	# Load a fixed sampled test set indexlist 
	if args.sample_test_json:
		json_file = os.path.join(args.data_dir, args.test_json)
		data_path = open(json_file, 'r')
		data = json.load(data_path)
		spherekey = np.array([item['key_sphere'] for item in data])
		cubekey = np.array([item['key_cubetetrah'] for item in data])
		spherekey = spherekey[:,np.newaxis]
		cubekey = cubekey[:,np.newaxis]
		shapekey_pair = np.concatenate((spherekey, cubekey), axis=1)
		indexkeys = sample_three_clusters_yefan(shapekey_pair, sparse = 1, num_data_each_shape = num_for_testshape)
		indexlist = []
		for shape in ["sphere", "cube", "triangle"]:
			indexlist += [item[0] for item in indexkeys[shape]]
			indexkeys_this_shape = np.array(indexkeys[shape])
			shapekeys_this_shape = indexkeys_this_shape[:,1:]
		
	else:
		# I saved a fixed test set indexlist named as 'test_index_100.json'
		with open(os.path.join('../experiment/triangle','test_index_{}.json'.format(num_for_testshape)), 'r') as result_file:
			indexlist = json.load(result_file)
			


	if ifsamept:
		sparselist = ["0.05", "0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90", "1.00"]
	
	elif ifconverge:
		sparselist = ["0.10", "0.30", "0.50", "0.70", "0.90", "1.00"]
	
	else:
		sparselist = ["0.10", "0.30", "0.50", "0.70", "0.90", "1.00"]
	
	
	for sparse in sparselist:
#		train_json = 'cluster_interp_5by20_%d.json'%sparse
		if ifsamept:
			train_json = "three_clusters_point_16_sparse_samept_{}.json".format(sparse)
			results_dir = "../experiment/triangle/results_bs8_samept_{}/final_vis".format(sparse)
			similarity_dir = "../experiment/triangle/results_bs8_samept_{}".format(sparse)

		elif ifconverge:
			train_json = "three_clusters_point_16_sparse_samept_{}.json".format(sparse)
			results_dir = "../experiment/triangle/results_bs8_samept_converge_{}/final_vis".format(sparse)
			similarity_dir = "../experiment/triangle/results_bs8_samept_converge_{}".format(sparse)

		else:
			train_json = "three_clusters_point_30_sparse_{}.json".format(sparse)
			results_dir = "../experiment/triangle/results_bs32_{}/final_vis".format(sparse)
			similarity_dir = "../experiment/triangle/results_bs32_{}".format(sparse)
		
		## If true, then load test results, compute distance matrix 
		## If false, then load distance matrix computed before  
		if args.first_plot:
			train_ptcloud, train_image = load_gt(train_json, args.data_dir)
			pred_ptcloud, pred_image, _ , pred_codeword = load_pred(args.data_dir, results_dir, args.test_json, indexlist = indexlist)
			train_ptcloud = train_ptcloud.cuda()
			pred_ptcloud = pred_ptcloud.cuda()
			train_image = train_image.cuda()
			pt_criterion = pt_criterion.cuda()
			img_criterion = img_criterion.cuda()
			train_pt_matrix, train_pt_part = compute_squared_EDM_method(train_ptcloud, pt_criterion, ifsavemat,  False,
											'%s_pt_similarity_matrix.npy'%'train', similarity_dir, num_pershape = num_pershape, part_preference = None)
			pred_pt_matrix, pred_pt_part = compute_squared_EDM_method(pred_ptcloud, pt_criterion,  ifsavemat,  True,
											'%s_pt_similarity_matrix.npy'%'results', similarity_dir, num_pershape = num_pershape, part_preference = AP_preference)
			train_img_matrix, train_image_part = compute_squared_EDM_method(train_image, img_criterion,  ifsavemat, False,
											'%s_img_similarity_matrix.npy'%'train', similarity_dir, num_pershape = num_pershape, part_preference = None)
		else:
			train_pt_matrix = np.load(os.path.join(similarity_dir, '%s_pt_similarity_matrix.npy'%'train'))  
			train_pt_part = np.load(os.path.join(similarity_dir, '%s_pt_similarity_matrix_part.npy'%'train'))
			train_img_matrix = 	np.load(os.path.join(similarity_dir, '%s_img_similarity_matrix.npy'%'train'))
			train_image_part = 	np.load(os.path.join(similarity_dir, '%s_img_similarity_matrix_part.npy'%'train'))
			pred_pt_matrix = np.load(os.path.join(similarity_dir, '%s_pt_similarity_matrix.npy'%'results'))
			pred_pt_part = np.load(os.path.join(similarity_dir, '%s_pt_similarity_matrix_part.npy'%'results'))

		
		## compute silhouette score
		train_pt_matrix_tr = transform_mat(train_pt_matrix)
		train_img_matrix_tr = transform_mat(train_img_matrix)
		pred_pt_matrix_tr = transform_mat(pred_pt_matrix)

		part_preference = cal_pref(train_pt_matrix_tr)
		train_pt_part = get_partition(train_pt_matrix_tr, preference = part_preference)
		print("train pt partition: {}".format(train_pt_part))
		
		part_preference = cal_pref(train_img_matrix_tr)
		train_img_part = get_partition(train_img_matrix_tr, preference = part_preference)
		print("train img partition: {}".format(train_img_part))

		part_preference = cal_pref(pred_pt_matrix_tr)
		pred_pt_part = get_partition(pred_pt_matrix_tr, preference = part_preference)
		print("pred pt partition: {}".format(pred_pt_part))
  
		train_pt_mod += [silhouette(train_pt_matrix, train_pt_part)]
		train_img_mod += [silhouette(train_img_matrix, train_image_part)]
		pred_pt_mod += [silhouette(pred_pt_matrix, pred_pt_part)]
			

	print(train_pt_mod)
	print(train_img_mod)
	print(pred_pt_mod)

		

	fig, ax = plt.subplots(figsize=(12, 30))
	for row, sparse in enumerate(sparselist):
		titles = ["train_img_similarity_matrix", "train_pt_similarity_matrix", "results_pt_similarity_matrix"]
		for col, title in enumerate(titles):
			if ifsamept:
				d = np.load("../experiment/triangle/results_bs8_samept_{0}/{1}.npy".format(sparse, title))
			elif ifconverge:
				d = np.load("../experiment/triangle/results_bs8_samept_converge_{0}/{1}.npy".format(sparse, title))
			else:
				d = np.load("../experiment/triangle/results_bs32_{0}/{1}.npy".format(sparse, title))
			plt.subplot(len(sparselist), 3, row  * 3 + col + 1)
			plt.imshow(d, interpolation='nearest')
			plt.title(title)
			plt.ylabel("sparsity = {0}".format(sparse))

	print("cluster-level higher means more separate clusters")
	
	if ifsamept:
		plt.savefig('../img/cluster/triangle/samept16_sparse_three_sample300_sub.png') 
		np.save(os.path.join(similarity_dir,'silouette_train_pt.npy'), train_pt_mod)
		np.save(os.path.join(similarity_dir,'silouette_train_img.npy'), train_img_mod)
		np.save(os.path.join(similarity_dir,'silouette_pred.npy'), pred_pt_mod)
	elif ifconverge:
		plt.savefig('../img/cluster/triangle/samept16_sparse_three_sample300_converge.png') 
#		np.save(os.path.join(similarity_dir,'silouette_train_pt.npy'), train_pt_mod)
#		np.save(os.path.join(similarity_dir,'silouette_train_img.npy'), train_img_mod)
#		np.save(os.path.join(similarity_dir,'silouette_pred.npy'), pred_pt_mod)
	else:
		plt.savefig('../img/cluster/triangle/samept_sparse.png') 
		np.save(os.path.join(similarity_dir,'silouette_train_pt.npy'), train_pt_mod)
		np.save(os.path.join(similarity_dir,'silouette_train_img.npy'), train_img_mod)
		np.save(os.path.join(similarity_dir,'silouette_pred.npy'), pred_pt_mod)
	

	## Plot modularity
	
	plt.figure()
	for i in range(0,len(sparselist)):
		plt.scatter(train_img_mod[i], pred_pt_mod[i],label = '%d'%(i+1))
	plt.plot(train_img_mod, pred_pt_mod)
	plt.legend()
	plt.title('silhouette_predpt_vs_train_img')
	plt_space = (max(train_img_mod) - min(train_img_mod))*0.1
	plt.xlim((min(train_img_mod)-plt_space, max(train_img_mod)+plt_space))
	if ifsamept:
		plt.savefig(os.path.join(args.save_dir,'%s.png'%'silhouette_predpt_vs_train_img_samept16_sample300_sub'))
	elif ifconverge:
		plt.savefig(os.path.join(args.save_dir,'%s.png'%'silhouette_predpt_vs_train_img_samept16_sample300_converge'))
	else:
		plt.savefig(os.path.join(args.save_dir,'%s.png'%'silhouette_predpt_vs_train_img'))

	plt.figure()
	for i in range(0,len(sparselist)):
		plt.scatter(train_pt_mod[i], pred_pt_mod[i],label = '%d'%(i+1))
	plt.plot(train_pt_mod, pred_pt_mod)
	plt.legend()
	plt.title('silhouette_pred_pt_vs_train_pt')
	plt_space = (max(train_pt_mod) - min(train_pt_mod))*0.1
	plt.xlim((min(train_pt_mod)-plt_space, max(train_pt_mod)+plt_space))
	if ifsamept:
		plt.savefig(os.path.join(args.save_dir,'%s.png'%'silhouette_pred_pt_vs_train_pt_samept16_sample300_sub'))
	elif ifconverge:
		plt.savefig(os.path.join(args.save_dir,'%s.png'%'silhouette_predpt_vs_train_pt_samept16_sample300_converge'))
	else:	
		plt.savefig(os.path.join(args.save_dir,'%s.png'%'silhouette_pred_pt_vs_train_pt'))



if __name__ == '__main__':
	parser = optparse.OptionParser(sys.argv[0], description="interpolation curve")

	parser.add_option("--origin-json",
					  dest="origin_json", type=str,
					  default="CubeSphere_interp_1000.json",
					  help='path of the origin complete json file')
	parser.add_option("--data-dir",
					  dest="data_dir", type=str,
					  default="../../Triangle",
					  help='data dir name')
	parser.add_option("--test-json",
					  dest="test_json", type=str,
					  default="test_triangle.json",
					  help='path of the testing json file')
	parser.add_option("--instance-num",
					  dest ="instance_num", type=int,
					  default = 1000,
					  help = '')
	parser.add_option("--save-dir",
					  dest="save_dir", type=str,
					  default="../img/cluster/triangle",
					  help='dir to save plot img')
	parser.add_option("--results-dir",
					  dest="results_dir", type=str,
#					  default="../experiment/cubesphere1000_1.27/cubesphere10by10_2.1/results/final_vis",
#					  default="../experiment/cubesphere1000_1.27/cubesphere0.1_1.29/results/final_vis",
#					  default="../experiment/cubesphere1000_1.27/cubesphere0.143_2.1/results/final_vis",
#					  default="../experiment/cubesphere1000_1.27/cubesphere0.02_1.29/results/final_vis",
					  default="../experiment/cubesphere1000_1.27/cubesphere1by100_2.1/results/final_vis",
#					  default="../experiment/cubesphere1000_1.27/cubesphere3by33_2.1/results/final_vis",
#					  default="../experiment/cubesphere1000_1.27/cubesphere4by25_2.1/results/final_vis",
#					  default="../experiment/cubesphere1000_1.27/cubesphere5by20_2.1/results/final_vis",
					  help='dir to save plot img')
	parser.add_option("--interpdata-dir",
					  dest="interpdata_dir", type=str,
					  default="../experiment/cubesphere1000_1.27/results/final_vis",
					  help='dir to save plot img')
	parser.add_option("--similarity-dir",
					  dest="similarity_dir", type=str,
					  default="../experiment/cubesphere1000_1.27/cubesphere1by100_2.1/results",
#					  default = "../experiment/cubesphere1000_1.27/cubesphere0.143_2.1/results",
#					  default="../experiment/cubesphere1000_1.27/cubesphere2by50_2.1/results",
#					  default="../experiment/cubesphere1000_1.27/cubesphere3by33_2.1/results",
#					  default="../experiment/cubesphere1000_1.27/cubesphere4by25_2.1/results",
#					  default="../experiment/cubesphere1000_1.27/cubesphere5by20_2.1/results",
					  help='dir to save plot img')
	parser.add_option("--first-plot",
					  dest="first_plot", action = "store_true",
					  default= True,
					  help='dir to save plot img')
	parser.add_option("--threshold",
					  dest="threshold", type = float,
					  default= 0.4,
					  help='threshold to judge the connectivity')
	parser.add_option("--type",
					  dest="type", type=str,
					  default= "train",
					  help='')
	parser.add_option("--pttemp",
					  dest="pttemp", type=float,
					  default= 10,
					  help='')
	parser.add_option("--imgtemp",
					  dest="imgtemp", type=float,
					  default= 0.01,
					  help='')
	parser.add_option("--sample-test-json",
					  dest="sample_test_json", action = "store_true",
					  default= False,
					  help='sample the large test dataset')
    
	(args, opts) = parser.parse_args()
	args.script_folder = os.path.dirname(os.path.abspath(__file__))

	main(args)

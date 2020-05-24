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
import matplotlib.pyplot as plt

import json
# from loss import ChamfersDistance3
import torch.nn as nn
from torchvision import transforms, utils
import tqdm
#from sklearn import preprocessing
#import pandas as pd
#import networkx as nx
#import community
#from sklearn.cluster import AffinityPropagation
#from sklearn.metrics import silhouette_score
#import math
#from community import community_louvain
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

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
    data.sort(key=lambda x:x['shapekey_value'])



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


def load_predpts(results_path, results_name, total_num = 10432, origin_test_batch = 200):

    num_of_npy = int(math.ceil(total_num/origin_test_batch))
    print(num_of_npy)
    for idx in tqdm.tqdm(range(num_of_npy)):
        postidx = '%03d'%idx + '.npy'
        tmp_pred = np.load(os.path.join(results_path, results_name + postidx))
        
        if idx == 0:
            pred = tmp_pred
        else:
            pred = np.concatenate((pred, tmp_pred), axis=0)
    #print(pred.shape)
    pred = np.squeeze(pred)
    assert pred.shape[0] == total_num, 'loading error'
    return pred

'''
def load_pred(data_dir, result_dir, test_json):
    json_file = os.path.join(data_dir, test_json)
    data_path = open(json_file, 'r')
    data = json.load(data_path)
    interp_list = []
    for i in range(len(data)):
        interp = round(1000 * data[i]['shapekey_value'])
        interp_list.append(interp)

    interp_list.sort()

    for i in tqdm.tqdm(range(len(interp_list)), total = len(range(len(interp_list))), desc = "load pred"):
        interp = interp_list[i]
        tmp_ptcloud = np.load(os.path.join(result_dir, 'fineptcloud_interp_%03d.npy' % interp))
        tmp_image = np.load(os.path.join(result_dir, 'img_interp_%03d.npy' % interp))
        tmp_codeword = np.load(os.path.join(result_dir, 'codeword_interp_%03d.npy' % interp))
    
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
    #pred_codeword = torch.from_numpy(pred_codeword).float()

    return pred_ptcloud, pred_image, interp_list , pred_codeword
'''
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


def compute_squared_EDM_method(X, distance_metric, ifsave, title, results_dir):
    # determin dimensions of data matrix
    N = X.shape[0]
    # initialize squared EDM D
    D = np.zeros([N, N])
    # iterate over upper triangle of D
    for i in tqdm.tqdm(range(N),total= len(range(N)), desc = "similarity matrix"):
        for j in range(i+1, N):
            D[i,j] = distance_metric(X[i].unsqueeze(0), X[j].unsqueeze(0))
            D[j,i] = D[i,j]    #*1
    print(D)
    if ifsave:
        np.save(os.path.join(results_dir,title), D)
        print("saved to" + os.path.join(results_dir, title))
    return D


def plot_embedding(data, label, title):

    #x_min, x_max = np.min(data, 0), np.max(data, 0)
    #data = (data - x_min) / (x_max - x_min)
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1],c='b',s = 20)
    plt.xticks([])
    plt.yticks([])
    plt.title(title,fontsize = 30)

# def silhouette(matrix, partition):
#     part = [partition[i] for i in range(matrix.shape[0])]
#     return silhouette_score(matrix, part, metric='precomputed')
#
# def gt_partition(matrix):
#     cl = AffinityPropagation(affinity='precomputed')
#     cl.fit(matrix)
#     partition = cl.labels_
#     return partition

def main(args):

    # pt_criterion = ChamfersDistance3()
    img_criterion = nn.L1Loss(reduction="sum")

    tsne = TSNE(n_components=2, init='pca', random_state=0)


    modelist = ['5by20','2by10']

    #plt.style.use('ggplot')

    '''
    fig, ax = plt.subplots(figsize=(20, 20))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    plt.rcParams['axes.titlesize'] = 25
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 24
    for idx, mode in enumerate(modelist):
        train_pt_mod = []
        pred_pt_mod = []
        train_img_mod = []
        args.first_plot = False
        if mode == '5by20':
            sparselist = [1, 2, 3, 4, 5, 6, 7]
            partition = 20*[0] + 20 *[1] + 20*[2] + 20*[3] + 20 *[4]
        elif mode == '2by10':
            sparselist = [1, 5, 10, 15, 20, 30, 35]
            partition = 10*[0] + 10 *[1] 

        print(mode, partition)

        test_json = 'test_interp_1000.json'
        json_file = os.path.join(args.data_dir, test_json)
        data_path = open(json_file, 'r')
        testdata = json.load(data_path)
        interp_list = []

        for i in range(len(testdata)):
            index_key_pair = (i, testdata[i]['shapekey_value'])
            interp_list.append(index_key_pair)
        interp_list.sort(key=lambda pair: pair[1])

        for sparse in sparselist:
            if mode == '5by20':
                train_json = 'cluster_interp_5by20_%d.json'%sparse	
                results_dir = "../experiment/cubesphere_5by20_resnet/results_{}/final_vis".format(sparse)
                similarity_dir = "../experiment/cubesphere_5by20_resnet/results_{}".format(sparse)
            elif mode == '2by10':
                train_json = 'cluster_samept_2by10_%d.json'%sparse
                results_dir = "../experiment/cubesphere_2by10_resnet/results_{}/final_vis".format(sparse)
                similarity_dir = "../experiment/cubesphere_2by10_resnet/results_{}".format(sparse)

            results_name = 'fineptcloud_'
            print('results_dir',results_dir)
            print('similarity_dir',similarity_dir)
            ## If true, then load test results, compute distance matrix 
            ## If false, then load distance matrix computed before  
            if args.first_plot:
                gt_ptcloud, gt_image = load_gt(test_json, args.data_dir)
                
                ########pred_ptcloud, pred_image, _ , pred_codeword = load_pred(args.data_dir, results_dir, args.test_json)
                
                #train_ptcloud, train_image = load_gt(train_json, args.data_dir)
                #pred_ptcloud = load_predpts(results_path = results_dir, results_name = results_name, total_num = 200, origin_test_batch = 100)
                #pred_ptcloud = torch.from_numpy(pred_ptcloud).float()
                sorted_predptcloud = torch.zeros(len(testdata), 1024, 3)
                sorted_gtptcloud = torch.zeros(len(testdata), 1024, 3)
                for i in range(len(testdata)):
                    #sorted_predptcloud[i] = pred_ptcloud[interp_list[i][0]]
                    sorted_gtptcloud[i] = gt_ptcloud[interp_list[i][0]]

                #train_ptcloud = train_ptcloud.cuda()
                #sorted_predptcloud = sorted_predptcloud.cuda()
                #train_image = train_image.cuda()
                sorted_gtptcloud = sorted_gtptcloud.cuda()
                pt_criterion = pt_criterion.cuda()
                img_criterion = img_criterion.cuda()
                train_pt_matrix = compute_squared_EDM_method(train_ptcloud, pt_criterion, True, 
                                                '%s_pt_similarity_matrix.npy'%'train', similarity_dir)
                pred_pt_matrix = compute_squared_EDM_method(sorted_predptcloud, pt_criterion, True, 
                                                '%s_pt_similarity_matrix.npy'%'results', similarity_dir)
                train_img_matrix = compute_squared_EDM_method(train_image, img_criterion, True, 
                                                '%s_img_similarity_matrix.npy'%'train', similarity_dir)
                gt_pt_matrix = compute_squared_EDM_method(sorted_gtptcloud, pt_criterion, True, 		
                                        '%s_pt_similarity_matrix.npy'%'gt', similarity_dir)

            else:
                #pred_ptcloud, pred_image, _ , pred_codeword = load_pred(args.data_dir, results_dir, args.test_json)
                train_pt_matrix = np.load(os.path.join(similarity_dir, '%s_pt_similarity_matrix.npy'%'train'))
                pred_pt_matrix = np.load(os.path.join(similarity_dir, '%s_pt_similarity_matrix.npy'%'results'))
                train_img_matrix = np.load(os.path.join(similarity_dir, '%s_img_similarity_matrix.npy'%'train'))
                gt_pt_matrix = np.load(os.path.join(similarity_dir, '%s_pt_similarity_matrix.npy'%'gt'))
            ## compute modularity

            #gt_part = gt_partition(gt_pt_matrix)
            pred_part = gt_partition(pred_pt_matrix)
            train_pt_mod += [silhouette(train_pt_matrix, partition)]
            train_img_mod += [silhouette(train_img_matrix, partition)]
            #pred_pt_mod += [silhouette(pred_pt_matrix, gt_part)]
            pred_pt_mod += [silhouette(pred_pt_matrix, pred_part)]


        print(idx)

#############################################################################
################################################################################
################################################################################
########Visualtion of Sil... Score##############################		
        plt.subplot(2,2,1+idx*2)
        for i in range(len(train_img_mod)):
            plt.scatter(train_img_mod[i], pred_pt_mod[i], c = colors[i],  s = 600,label = 'dataset index = {}'.format(i+1))
        plt.plot(train_img_mod, pred_pt_mod, linewidth = 7, c= 'k')
        if idx == 1:
            plt.xlabel('Training Image Dataset Silhouette Score',labelpad = 25)
        plt.ylabel('Predicted Point Cloud Results Silhouette Score',labelpad = 22)
        plt.yticks(np.arange(0.25, 0.9, step=0.05))
        #plt.xticks(np.arange(0.5, 1, step=0.1))
        plt.legend(loc = 3, fontsize = 23)
        if mode =='5by20':
            plt.title('Subsampled dataset 1.1')
        if mode =='2by10':
            plt.title('Subsampled dataset 1.2')
        plt.tick_params(labelsize=16)

    
        for x, y in zip(train_img_mod, pred_pt_mod):
            label = "index = {}".format()
            plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,12), # distance from text to points (x,y)
                 ha='center',
                 fontsize=14,
                 fontname='sans')
    
        plt.subplot(2,2,2+idx*2)
        for i in range(len(train_pt_mod)):
            plt.scatter(train_pt_mod[i], pred_pt_mod[i], c = colors[i], s = 600, label = 'dataset index = {}'.format(i+1))
        plt.plot(train_pt_mod, pred_pt_mod, linewidth = 7, c= 'k')
        if idx == 1:
            plt.xlabel('Training Point Cloud Dataset Silhouette Score', labelpad = 25)

        plt.yticks(np.arange(0.25, 0.9, step=0.05))

        if mode =='5by20':
            plt.title('Subsampled dataset 1.1')
        if mode =='2by10':
            plt.title('Subsampled dataset 1.2')
    
        plt.legend(loc = 3, fontsize = 23)
        plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig('../img/cluster/sparse/cluster_score_method1.png')
    '''


#############Visualization of Point cloud#####################################################
##############################################################################################
##############################################################################################

    mode = '5by20'
    # mode = '2by10'

    if mode == '2by10':
        sparselist = [1, 5, 10, 15, 20, 30, 35]
    elif mode == '5by20':
        sparselist = [1, 2, 3, 4, 5, 6, 7]

    #fig.suptitle('Distance Matrices',fontsize=30,fontname="Times New Roman")
    for row, sparse in enumerate(sparselist):
        titles = ["train_img_similarity_matrix", "train_pt_similarity_matrix", "results_pt_similarity_matrix"]
        for col, title in enumerate(titles):
            if mode == '5by20':
                d = np.load("../experiment/cubesphere_5by20_resnet/results_{0}/{1}.npy".format(sparse, title))
            elif mode == '2by10':
                d = np.load("../experiment/cubesphere_2by10_resnet/results_{0}/{1}.npy".format(sparse, title))
            print(np.max(np.max(d,axis =1)))
            #plt.subplot(len(sparselist), 3, row  * 3 + col + 1)
            fig, ax = plt.subplots(figsize=(12, 12))
            im = ax.imshow(d, interpolation='nearest', cmap=plt.cm.get_cmap("jet"))
            # plt.imshow(d, interpolation='nearest', cmap=plt.cm.get_cmap("jet"))
            # plt.xticks([])
            # plt.yticks([])

            ax.set_xticks([])
            ax.set_yticks([])

            aspect = 20
            pad_fraction = 0.5

            #plt.tight_layout()
            if col == 0:
                v = np.linspace(0, 8000, 10, endpoint=True)
                # plt.clim(0, 8000)
                # cb = plt.colorbar(ticks=v)
                # cb = plt.colorbar(im,fraction=0.046, pad=0.04)
                # cb.set_ticks([v])
                # #np.arange(0, 8000, 2)
                # plt.colorbar().ax.set_yticklabels(v,
                #     fontsize=16, weight='bold')
                #plt.colorbar().ax.tick_params(labelsize =14)
                # cb.ax.tick_params(labelsize=16)
                divider = make_axes_locatable(ax)
                width = axes_size.AxesY(ax, aspect=1. / aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                cbar = plt.colorbar(im, cax=cax,  ticks=v)
                cbar.remove()
                # cbar.ax.locator_params(nbins=4)
                # # cbar.set_clim(0, 9000)
                # # cbar.set_ticks([0, 3000, 6000])
                # cbar.ax.tick_params(labelsize=16)
                # cbar.set_ticklabels([0, 4000, 8000])
                # plt.ylabel("dataset index = {0}".format(row+1), fontsize=45, labelpad = 30,fontweight ='bold',fontname="Times New Roman")
            elif col == 1:
                v = np.linspace(0, 0.45, 10, endpoint=True)
                divider = make_axes_locatable(ax)
                width = axes_size.AxesY(ax, aspect=1. / aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                cbar = plt.colorbar(im, cax=cax, ticks=v)
                cbar.remove()
                # # cbar.set_ticks([0, 0.18, 0.36])
                # cbar.ax.tick_params(labelsize=16)
                # cbar.ax.locator_params(nbins=4)
                # #plt.clim(0, 0.6) np.arange(0, 0.6, 2)
                # # plt.colorbar().ax.set_yticklabels(v,
                # #     fontsize=16, weight='bold')
                # #plt.colorbar().ax.tick_params(labelsize =14)
            elif col == 2:
                v = np.linspace(0, 0.45, 10, endpoint=True)
                divider = make_axes_locatable(ax)
                width = axes_size.AxesY(ax, aspect=1. / aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                cbar = plt.colorbar(im, cax=cax, ticks=v) #, ticks=v)
                cbar.remove()
                # cbar.ax.locator_params(nbins=4)
                # cbar.ax.tick_params(labelsize=16)
                # #plt.clim(0, 0.6)np.arange(0, 0.6, 2)
                # # plt.colorbar().ax.set_yticklabels(v,
                # #     fontsize=16, weight='bold')
                # #plt.colorbar().ax.tick_params(labelsize =14)
            '''
            if row == 0 and col ==0:
                plt.title('Training Image',fontsize=48, fontweight = 'bold', fontname="Times New Roman",pad=30)
            if row == 0 and col ==1:
                plt.title('Training Point Cloud',fontsize=48,fontweight = 'bold',fontname="Times New Roman",pad=30)
            if row == 0 and col ==2:
                plt.title('Predicted Point Cloud',fontsize=48,fontweight = 'bold',fontname="Times New Roman",pad=30)
            '''
            plt.savefig('cluster_{}_{}_{}.png'.format(mode, row+1,col+1), bbox_inches='tight')

    print("cluster-level higher means more separate clusters")




    '''
    ## Plot modularity
    plt.figure()
    for i in range(0,7):
        plt.scatter(train_img_mod[i], pred_pt_mod[i],label = '%d'%(i+1))
    plt.plot(train_img_mod, pred_pt_mod)
    plt.legend()
    plt.title('predptmod_vs_train_img')
    plt_space = (max(train_img_mod) - min(train_img_mod))*0.1
    plt.xlim((min(train_img_mod)-plt_space, max(train_img_mod)+plt_space))
    plt.savefig(os.path.join(args.save_dir,'%s.png'%'predptmod_vs_train_img'))

    plt.figure()
    for i in range(0,7):
        plt.scatter(train_pt_mod[i], pred_pt_mod[i],label = '%d'%(i+1))
    plt.plot(train_pt_mod, pred_pt_mod)
    plt.legend()
    plt.title('pred_pt_mod_vs_train_pt')
    plt_space = (max(train_pt_mod) - min(train_pt_mod))*0.1
    plt.xlim((min(train_pt_mod)-plt_space, max(train_pt_mod)+plt_space))
    plt.savefig(os.path.join(args.save_dir,'%s.png'%'pred_pt_mod_vs_train_pt'))


    

    
    fig, ax = plt.subplots(figsize=(12, 25))
    plt.subplot(2, 1, 1)
    plt.imshow(train_pt_matrix, interpolation='nearest')
    plt.title("train_pt_matrix")
#	plt.subplot(3, 1, 2)
#	plt.imshow(pred_pt_matrix, interpolation='nearest')
#	plt.title("pred_pt_matrix")
    plt.subplot(2, 1, 2)
    plt.imshow(train_img_matrix, interpolation='nearest')
    plt.title("train_img_matrix")
    plt.savefig('../img/cluster/sparse/sort7_level.png') 
    '''

if __name__ == '__main__':
    parser = optparse.OptionParser(sys.argv[0], description="interpolation curve")

    parser.add_option("--origin-json",
                      dest="origin_json", type=str,
                      default="CubeSphere_interp_1000.json",
                      help='path of the origin complete json file')
    parser.add_option("--data-dir",
                      dest="data_dir", type=str,
                      default="../../../interp_data/cubesphere_1000",
                      help='data dir name')
    parser.add_option("--train-json",
                      dest="train_json", type=str,
#					  default="cluster_interp_10by10.json",
#					  default = "train_interp_0.143.json",
                      default = "cluster_interp_1by100.json",
#					  default = "cluster_interp_3by33.json",
#					  default = "cluster_interp_4by25.json",
#					  default = "cluster_interp_5by20.json",
                      help='path of the training json file')
    parser.add_option("--test-json",
                      dest="test_json", type=str,
                      default="test_interp_1000.json",
                      help='path of the testing json file')
    parser.add_option("--instance-num",
                      dest ="instance_num", type=int,
                      default = 1000,
                      help = '')
    parser.add_option("--save-dir",
                      dest="save_dir", type=str,
                      default="../img/cluster/sparse",
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
                      default= 100,
                      help='')
    parser.add_option("--imgtemp",
                      dest="imgtemp", type=float,
                      default= 1,
                      help='')
    
    (args, opts) = parser.parse_args()
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)

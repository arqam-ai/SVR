'''
Oracle NN

author: Yefan
created: 2020/3/10 11:21 PM
'''
import os 
import sys
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import numpy as np
import argparse
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.getcwd(), "../../../"))
from mpl_toolkits.mplot3d import Axes3D
from utils.loss import ChamferDistance
import optparse
import json
import cv2
import tqdm
#from torch.utils.data import Dataset, DataLoader, TensorDataset
#import glog as logger
import logging
from dataset.dataset import what3d_dataset_views
from utils.utils import Normalization
import time
import random
from model.what3d.Oracle_NN import Oracle_NN


def search(class_index_dic_path):

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    logger = logging.getLogger()
    file_log_handler = logging.FileHandler(os.path.join(args.save_path, 'Train.log'))
    logger.addHandler(file_log_handler)
    logger.setLevel('INFO')
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s","%Y-%m-%d %H:%M:%S")
    file_log_handler.setFormatter(formatter)
    stderr_log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stderr_log_handler)
    stderr_log_handler.setFormatter(formatter)
    logger.info("Running script for trainning the Oracle NN")


    ptcloud_dic = {'train':{}, 'test':{}}
    class_index_dic = np.load(class_index_dic_path, allow_pickle=True).item()
    assert list(class_index_dic['train'].keys()) == list(class_index_dic['test'].keys()), "class key list error"
    clskey_list = list(class_index_dic['train'])
    for view in list(args.views):
        ptcloud_path = args.ptcloud_path[:-5] + '%s.npz'%view
        ptcloud = np.load(ptcloud_path)
        train_set = torch.from_numpy(ptcloud['train'])  
        test_set = torch.from_numpy(ptcloud['test'])
        train_set = Normalization(train_set, inplace=True, keep_track=False).normalize_unitL2ball()
        test_set = Normalization(test_set, inplace=True, keep_track=False).normalize_unitL2ball()

        for key in clskey_list:
            train_idx = class_index_dic['train'][key][0]
            train_slicer = class_index_dic['train'][key][1]
            test_idx = class_index_dic['test'][key][0]
            test_slicer = class_index_dic['test'][key][1]
            if view == '0':
                ptcloud_dic['train'][key] = train_set[train_idx:train_slicer]
                ptcloud_dic['test'][key] = test_set[test_idx:test_slicer]
            else:
                ptcloud_dic['train'][key] = torch.cat((ptcloud_dic['train'][key], train_set[train_idx:train_slicer]), 0)
                ptcloud_dic['test'][key] =  torch.cat((ptcloud_dic['test'][key], test_set[test_idx:test_slicer]), 0)
    
    model = Oracle_NN(args.device, logger, args.save_path)
    class_loss = {"loss":{}, "index":{}} 

    for order, key in enumerate(clskey_list):
        logger.info("The {}th class {}".format(order, key))
        sum_classloss, class_trainindex = model.forward(ptcloud_dic['train'][key].to(args.device), 
                  ptcloud_dic['test'][key].to(args.device), 0, 0)
        class_loss["loss"][key] = sum_classloss.item()
        class_loss["index"][key] = class_trainindex.cpu().numpy()
    
    sum_loss = 0
    for key in clskey_list:
        sum_loss += class_loss["loss"][key]

    np.save(os.path.join(args.save_path, "class_loss_dic.npy"), class_loss)
    sum_instances = sum([ptcloud_dic['test'][key].shape[0] for key in clskey_list])
    avg = sum_loss/sum_instances
    logger.info('final avg loss: %.6f' % (avg))


def search_one_class(args, class_index_dic_path, class_idx):

    class_index_dic = np.load(class_index_dic_path, allow_pickle=True).item()
    assert list(class_index_dic['train'].keys()) == list(class_index_dic['test'].keys()), "class key list error"
    clskey_list = list(class_index_dic['train'])

    key = clskey_list[class_idx]

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(os.path.join(args.save_path, key)):
        os.mkdir(os.path.join(args.save_path, key))

    logger = logging.getLogger()
    file_log_handler = logging.FileHandler(os.path.join(args.save_path, key, 'Train.log'))
    logger.addHandler(file_log_handler)
    logger.setLevel('INFO')
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s","%Y-%m-%d %H:%M:%S")
    file_log_handler.setFormatter(formatter)
    #stderr_log_handler = logging.StreamHandler(sys.stdout)
    #logger.addHandler(stderr_log_handler)
    #stderr_log_handler.setFormatter(formatter)
    logger.info("Running script for trainning the Oracle NN")
    ptcloud_dic = {'train':{}, 'test':{}}

    for view in list(args.views):
        ptcloud_path = args.ptcloud_path[:-5] + '%s.npz'%view
        ptcloud = np.load(ptcloud_path)
        train_set = torch.from_numpy(ptcloud['train'])  
        test_set = torch.from_numpy(ptcloud['test'])
        train_set = Normalization(train_set, inplace=True, keep_track=False).normalize_unitL2ball()
        test_set = Normalization(test_set, inplace=True, keep_track=False).normalize_unitL2ball()

        train_idx = class_index_dic['train'][key][0]
        train_slicer = class_index_dic['train'][key][1]
        test_idx = class_index_dic['test'][key][0]
        test_slicer = class_index_dic['test'][key][1]
        if view == '0':
            ptcloud_dic['train'][key] = train_set[train_idx:train_slicer]
            ptcloud_dic['test'][key] = test_set[test_idx:test_slicer]
        else:
            ptcloud_dic['train'][key] = torch.cat((ptcloud_dic['train'][key], train_set[train_idx:train_slicer]), 0)
            ptcloud_dic['test'][key] =  torch.cat((ptcloud_dic['test'][key], test_set[test_idx:test_slicer]), 0)
    
    sum_instances = ptcloud_dic['test'][key].shape[0]

    model = Oracle_NN(args.device, logger, os.path.join(args.save_path, key), sum_instances)
    class_loss = {"sumloss":{}, "avgloss":{}, "index":{}, "nums":{}} 

    logger.info("The class {}".format(key))
    sum_classloss, class_trainindex = model.forward(ptcloud_dic['train'][key].to(args.device), 
                ptcloud_dic['test'][key].to(args.device), 0, 0)
    sum_loss = sum_classloss.item()
    avg = sum_loss/sum_instances
    class_loss["sumloss"][key] = sum_loss
    class_loss["index"][key] = class_trainindex.cpu().numpy()
    class_loss["nums"][key] = sum_instances
    class_loss["avgloss"][key] = avg
    
    np.save(os.path.join(args.save_path, key, "class_loss_dic.npy"), class_loss)
    logger.info('class sum loss: %.6f, class instances num: %d, class avg loss: %.6f' % (sum_loss, sum_instances, avg))

def search_one_class_parallel(args, class_index_dic_path, class_idx):

    class_index_dic = np.load(class_index_dic_path, allow_pickle=True).item()
    assert list(class_index_dic['train'].keys()) == list(class_index_dic['test'].keys()), "class key list error"
    clskey_list = list(class_index_dic['train'])

    key = clskey_list[class_idx]

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(os.path.join(args.save_path, key)):
        os.mkdir(os.path.join(args.save_path, key))

    logger = logging.getLogger()
    file_log_handler = logging.FileHandler(os.path.join(args.save_path, key, 'Train_parallel.log'))
    logger.addHandler(file_log_handler)
    logger.setLevel('INFO')
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s","%Y-%m-%d %H:%M:%S")
    file_log_handler.setFormatter(formatter)
    #stderr_log_handler = logging.StreamHandler(sys.stdout)
    #logger.addHandler(stderr_log_handler)
    #stderr_log_handler.setFormatter(formatter)
    logger.info("Running script for trainning the Oracle NN in Python Multiprocessing")
    ptcloud_dic = {'train':{}, 'test':{}}

    trainidx_path = os.path.join(args.save_path, key, "test_mintrainidx.npz")
    if os.path.exists(trainidx_path):
        prev_trainidx = np.load(trainidx_path)["iter_loss"]
        prev_trainidx = prev_trainidx.tolist()
    else:
        prev_trainidx = None

    loss_path = os.path.join(args.save_path, key, "test_minloss.npz")
    if os.path.exists(loss_path):
        prev_minloss = np.load(loss_path)["iter_loss"]
        prev_minloss = prev_minloss.tolist()
    else:
        prev_minloss = None 

    for view in list(args.views):
        print("view", view)
        ptcloud_path = args.ptcloud_path[:-5] + '%s.npz'%view
        ptcloud = np.load(ptcloud_path)
        train_set = torch.from_numpy(ptcloud['train'])  
        test_set = torch.from_numpy(ptcloud['test'])
        train_set = Normalization(train_set, inplace=True, keep_track=False).normalize_unitL2ball()
        test_set = Normalization(test_set, inplace=True, keep_track=False).normalize_unitL2ball()

        train_idx = class_index_dic['train'][key][0]
        train_slicer = class_index_dic['train'][key][1]
        test_idx = class_index_dic['test'][key][0]
        test_slicer = class_index_dic['test'][key][1]
        if view == '0':
            ptcloud_dic['train'][key] = train_set[train_idx:train_slicer]
            ptcloud_dic['test'][key] = test_set[test_idx:test_slicer]
        else:
            ptcloud_dic['train'][key] = torch.cat((ptcloud_dic['train'][key], train_set[train_idx:train_slicer]), 0)
            ptcloud_dic['test'][key] =  torch.cat((ptcloud_dic['test'][key], test_set[test_idx:test_slicer]), 0)
    
    print("Finished Loading DATA-------------------------------------------------")
    sum_instances = ptcloud_dic['test'][key].shape[0]

    model = Oracle_NN(args.device, logger, os.path.join(args.save_path, key), sum_instances, prev_trainidx, prev_minloss,
                       train_set=ptcloud_dic['train'][key].to(args.device), test_set=ptcloud_dic['test'][key].to(args.device),
                       parallel=True)

    class_loss = {"sumloss":{}, "avgloss":{}, "index":{}, "nums":{}} 

    logger.info("The class {}".format(key))
    sum_classloss, class_trainindex = model.forward_multiprc(num_proc=10)
    
    avg = sum_classloss/sum_instances
    class_loss["sumloss"][key] = sum_classloss
    class_loss["index"][key] = class_trainindex
    class_loss["nums"][key] = sum_instances
    class_loss["avgloss"][key] = avg
    
    np.save(os.path.join(args.save_path, key, "class_loss_dic.npy"), class_loss)
    logger.info('class sum loss: %.6f, class instances num: %d, class avg loss: %.6f' % (sum_classloss, sum_instances, avg))



def infer_one_class(args, class_index_dic_path, class_idx):
    img_save_path = os.path.join(args.save_path, "oracle_test")
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    class_index_dic = np.load(class_index_dic_path, allow_pickle=True).item()
    assert list(class_index_dic['train'].keys()) == list(class_index_dic['test'].keys()), "class key list error"
    clskey_list = list(class_index_dic['train'])
    key = clskey_list[class_idx]
    class_loss_dic = np.load(os.path.join(args.save_path, key, "class_loss_dic.npy"), allow_pickle=True).item()
    ptcloud_dic = {'train':{}, 'test':{}}
    chamfer = ChamferDistance().to(args.device)

    for view in list(args.views):
        ptcloud_path = args.ptcloud_path[:-5] + '%s.npz'%view
        ptcloud = np.load(ptcloud_path)
        train_set = torch.from_numpy(ptcloud['train'])  
        test_set = torch.from_numpy(ptcloud['test'])
        train_set = Normalization(train_set, inplace=True, keep_track=False).normalize_unitL2ball()
        test_set = Normalization(test_set, inplace=True, keep_track=False).normalize_unitL2ball()

        train_idx = class_index_dic['train'][key][0]
        train_slicer = class_index_dic['train'][key][1]
        test_idx = class_index_dic['test'][key][0]
        test_slicer = class_index_dic['test'][key][1]
        if view == '0':
            ptcloud_dic['train'][key] = train_set[train_idx:train_slicer]
            ptcloud_dic['test'][key] = test_set[test_idx:test_slicer]
        else:
            ptcloud_dic['train'][key] = torch.cat((ptcloud_dic['train'][key], train_set[train_idx:train_slicer]), 0)
            ptcloud_dic['test'][key] =  torch.cat((ptcloud_dic['test'][key], test_set[test_idx:test_slicer]), 0)

    test_instances = ptcloud_dic['test'][key].shape[0]
    sum_loss = 0
    for i in tqdm.tqdm(range(test_instances), total=test_instances, desc=key):
        prediction = ptcloud_dic['train'][key][class_loss_dic["index"][key][i]]
        gt = ptcloud_dic['test'][key][i].unsqueeze(0)
        sum_loss += chamfer(gt.to(args.device), prediction.to(args.device)).item()

        what3d_dataset_views.data_visualizer(gt, prediction, torch.zeros((1, 3, 224, 224)),  
            split_name = key, path = img_save_path,  
            idx = i)
    print("Class %s, New Inference Avg Loss %.6f, Original Avg Loss in Training %.6f" % (key, sum_loss/test_instances, class_loss_dic["avgloss"][key])) 
    

def infer_summary(args, index_list):
    class_index_dic = np.load(args.class_index_dic_path, allow_pickle=True).item()
    sum_loss = 0
    num_instance = 0
    avg_list = []
    for class_index in index_list:
        clskey_list = list(class_index_dic['train'])
        class_key = clskey_list[class_index]
        class_stats = np.load(os.path.join(args.save_path, class_key, "class_loss_dic.npy"), allow_pickle=True).item()
        #print(class_key, class_stats['avgloss'][class_key])
        sum_loss += class_stats['sumloss'][class_key]
        num_instance += class_stats['nums'][class_key]
        avg_list.append([class_key, class_stats['avgloss'][class_key], class_stats['nums'][class_key]])
    
    print(len(index_list), num_instance)
    print("avg loss {}".format(sum_loss/num_instance))
    avg_list = sorted(avg_list, key = lambda x: x[2])
    print(avg_list[0], avg_list[-1])
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--data-basedir',type=str,default='/home/zyf/What3D',
                    help='path of the jsonfile')
    parser.add_argument('--img-path',type=str,default='renderings',
                    help='path of the jsonfile')
    parser.add_argument("--splits-path",dest="splits_path", type=str, default='splits',
                      help='path of the data folder') 
    parser.add_argument("--ptcloud-path",type=str,default="../../What3D/ptcloud_n.npz",help=' ' )
    parser.add_argument("--class-path", dest="class_path", type=str,default='classes.txt',help="class name list")
    parser.add_argument("--views",dest="views", type=str,default= '01234',
                    help="five view for each instance")
    parser.add_argument("--save-path",dest="save_path", type=str,default= 'exp2',
                    help="")
    parser.add_argument("--class_index_dic_path", type=str, default="/home/zyf/What3D/class_index_dic.npy", help="")
    parser.add_argument("--class_index", type=int, default=0, help="") 
    parser.add_argument("--search", action="store_true", dest="search", default=False)   
    parser.add_argument("--parallel", action="store_true", dest="parallel", default=False)         
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.script_folder = os.path.dirname(os.path.abspath(__file__))
    # if args.search:
    #     print("Running scripts for searching class index %d, save path %s" % (args.class_index, args.save_path))
    #     if args.parallel:
    #         search_one_class_parallel(args, args.class_index_dic_path, args.class_index)
    #     else:
    #         search_one_class(args, args.class_index_dic_path, args.class_index)

    # else:
    #     print("Running scripts for inferring class index %d, save path %s" % (args.class_index, args.save_path))
    #     infer_one_class(args, args.class_index_dic_path, args.class_index)
    
    
    infer_summary(args, index_list = [item for item in range(0,55) if item not in [47, 16, 0, 18, 49]]) 
    
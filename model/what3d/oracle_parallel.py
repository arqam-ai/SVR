'''
This script is for oracle NN parallel processing 
Author: Yefan
'''
import os 
import sys
import torch
import numpy as np
import argparse
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.getcwd(), "../../../"))
from mpl_toolkits.mplot3d import Axes3D
from utils.loss import ChamferDistance
import cv2
import tqdm
import logging
from dataset.dataset import what3d_dataset_views
from utils.utils import Normalization
import time
import random
from multiprocessing import Pool, set_start_method
#from torch.multiprocessing import Pool, Process, set_start_method

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--data-basedir',type=str,default='../../../../What3D',
                help='path of the jsonfile')
parser.add_argument('--img-path',type=str,default='renderings',
                help='path of the jsonfile')
parser.add_argument("--splits-path",dest="splits_path", type=str, default='splits',
                    help='path of the data folder') 
parser.add_argument("--ptcloud-path",type=str,default="../../../../What3D/ptcloud_n.npz",help=' ' )
parser.add_argument("--class-path", dest="class_path", type=str,default='classes.txt',help="class name list")
parser.add_argument("--views",dest="views", type=str,default= '01234',
                help="five view for each instance")
parser.add_argument("--save-path",dest="save_path", type=str,default= 'exp2',
                help="")
parser.add_argument("--class_index_dic_path", type=str, default="../../../../What3D/class_index_dic.npy", help="")
parser.add_argument("--class_index", type=int, default=0, help="") 
parser.add_argument("--num_processors", type=int, default=10, help="")
parser.add_argument("--search", action="store_true", dest="search", default=False) 
parser.add_argument("--start_index", type=int, default=0, help="start index to do oracle nn included")   
parser.add_argument("--end_index", type=int, default=0, help="end index excluded")  
parser.add_argument("--checkpt_path", type=str, default="", help="checkpoint path")     

args = parser.parse_args(sys.argv[1:])
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chamfer = ChamferDistance().to(args.device)

def nn_multiprc(train_set, test_ptcloud, checkpt_path, test_index):
    """Function for search
    train_set      :    torch.tensor
        train point cloud set 
    test_ptcloud   : torch.tensor
        test point cloud set 
    checkpt_path   : str
    test_index     : int
    """
    min_idx = 0
    min_loss = 1000
    test_ptcloud = test_ptcloud.unsqueeze(0)
    # enumerate through train set， compare and  find the minimum loss 
    for idx in range(train_set.shape[0]):
        loss = chamfer(train_set[idx].unsqueeze(0), test_ptcloud).item() 
        if loss < min_loss:
            min_idx = idx
            min_loss = loss
    logger.info('test index:%d, loss:%.6f, predicted idx in trainset:%d' % (test_index, min_loss, min_idx))
    np.save(os.path.join(checkpt_path, "checkpt_{}.npy".format(test_index)), np.array([test_index, min_loss, min_idx]))

class_index_dic = np.load(args.class_index_dic_path, allow_pickle=True).item()

assert list(class_index_dic['train'].keys()) == list(class_index_dic['test'].keys()), "class key list error"
clskey_list = list(class_index_dic['train'])
##key is the class name
key = clskey_list[args.class_index]

#####################Make Logger#############################
logger = logging.getLogger()
file_log_handler = logging.FileHandler(os.path.join(args.save_path, key, 'Train_parallel.log'))
logger.addHandler(file_log_handler)
logger.setLevel('INFO')
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s","%Y-%m-%d %H:%M:%S")
file_log_handler.setFormatter(formatter)
stderr_log_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stderr_log_handler)
stderr_log_handler.setFormatter(formatter)
logger.info("Running script for trainning the Oracle NN in parallel")
logger.info("Class Name {}, start index: {}, end index: {}".format(key, args.start_index, args.end_index))



if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
if not os.path.exists(os.path.join(args.save_path, key)):
    os.mkdir(os.path.join(args.save_path, key))

checkpt_path = os.path.join(args.save_path, key, args.checkpt_path)
if not os.path.exists(checkpt_path):
    os.mkdir(checkpt_path)

####################Load DATA######################
ptcloud_dic = {'train':{}, 'test':{}}
for view in list(args.views):
    logger.info("view {}".format(view))
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

train_set = ptcloud_dic['train'][key]
test_set = ptcloud_dic['test'][key]

logger.info("Finished Loading DATA-------------------------------------------------")
for test_index in range(args.start_index, args.end_index):
    nn_multiprc(train_set.to(args.device), test_set[test_index].to(args.device), checkpt_path, test_index)


#sum_classloss = sum([item[1] for item in prev_minloss])
#class_trainindex = [item[1] for item in prev_trainidx]

#class_loss = {"sumloss":{}, "avgloss":{}, "index":{}, "nums":{}} 

#avg = sum_classloss/sum_instances
#class_loss["sumloss"][key] = sum_classloss
#class_loss["index"][key] = class_trainindex
#class_loss["nums"][key] = sum_instances
#class_loss["avgloss"][key] = avg

#np.save(os.path.join(args.save_path, key, "class_loss_dic.npy"), class_loss)
#logger.info('class sum loss: %.6f, class instances num: %d, class avg loss: %.6f' % (sum_classloss, sum_instances, avg))





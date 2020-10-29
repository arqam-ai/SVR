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
import logging
from dataset.dataset import what3d_dataset_views
from utils.utils import Normalization, find_median, normalize_bbox
from utils.loss import evaluate_voxel_prediction
import time
import random
import utils.binvox_rw as binvox_rw
import utils.binvox_torch as binvox_torch
import matplotlib.pyplot as plt
from utils.f_score import threshold_list, f_score_list, BBOX
from voxel.compute_miou import convert_array_to_dataframe, convert_ptc_to_voxel


#from encoders import Encoder

class Stats(object):
    def __init__(self):
        self.iter_loss = []

    def push_loss(self, iter, loss):
        self.iter_loss.append([iter, loss])

    def push(self, iter, loss):
        self.push_loss(iter, loss)

    def save(self, file):
        np.savez_compressed(
            file,
            iter_loss=np.asarray(self.iter_loss))

def class_counter(args, split_name):
    f_class = open(os.path.join(args.data_basedir, args.splits_path, args.class_path),"r")
    
    class_num = 0
    class_dic = {}           # class_name : num of instance in this class
    class_index = {}         # class_name : class_index     e.g. airplane:0
    class_list = []          # 55 class
    data_class = []          # airairairchairchair
    color = []
    for line in f_class:
        index = line.find(' ')
        clname = line[:index]
        class_dic[clname] = 0
        class_list += [clname]
        class_index[clname] = class_num
        class_num += 1
        
    instance_num = 0
    for clname in tqdm.tqdm(class_list,total= len(class_list), desc = '%s'%split_name):
        f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt'%split_name),"r")
        for x in f:
            class_dic[clname] += 1
            instance_num += 1
            data_class += [clname]
            color += [class_index[clname]]
    
    #print(instance_num,class_dic, data_class) 
    return class_list, instance_num, class_dic, data_class, color


def dict_Avg(Dict) :
    L = len(Dict)						
    S = sum(Dict.values())				
    A = S / L
    return A

class Oracle_NN_infer(object):

    def __init__(self, device, num_instance):
        super(Oracle_NN_infer,self).__init__()
        self.test_index = 0
        self.device = device
        self.pbar = tqdm.tqdm(total=num_instance)
        self.total_loss = 0
        self.criterion = ChamferDistance()
        self.criterion = self.criterion.to(device)
        self.prediction = torch.zeros(num_instance, 1024, 3).to(device)

    def forward(self, image_path, train_gt, test_gt, pred_trainindex, img_save_path):
        
        for local_test_index, train_index in enumerate(pred_trainindex):
            prediction = train_gt[train_index]
            self.prediction[self.test_index] = prediction
            gt = test_gt[local_test_index]
            prediction = prediction.unsqueeze(0)
            gt = gt.unsqueeze(0)
            
            self.total_loss += self.criterion(prediction, gt).item()
            image = cv2.imread(image_path[local_test_index]).transpose((2, 0, 1))
            image = np.expand_dims(image, axis=0)
            gt = gt.cpu().numpy()
            prediction = prediction.cpu().numpy()

            #what3d_dataset_views.data_visualizer(gt, prediction, image,  
            # split_name = "test", path = img_save_path,  
            # idx = self.test_index)
            self.test_index += 1
            self.pbar.update(1)


def infer(args, train_index_path, img_save_path):

    logger = logging.getLogger()
    file_log_handler = logging.FileHandler('Train.log')
    logger.addHandler(file_log_handler)
    logger.setLevel('INFO')
    formatter = logging.Formatter()
    file_log_handler.setFormatter(formatter)
    #stderr_log_handler = logging.StreamHandler(sys.stdout)
    #logger.addHandler(stderr_log_handler)
    #stderr_log_handler.setFormatter(formatter)
    logger.info("Running script for testing the Oracle NN")

    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    class_list, _, _, _,_ = class_counter(args, "test")
    image_path_list = list()
    img_view_list = list()

    ## Load the total instance path
    for clname in class_list:
        f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt'%"test"),"r")
        for x in f:
            instance_id = x[:-1]
            image_path_list.append(os.path.join(args.data_basedir, args.img_path, clname, instance_id))

    for view_idx, view in enumerate(args.views):
        for i in range(len(image_path_list)):
            img_view_list.append(os.path.join(image_path_list[i], '%s.png'%view))
    
    _, _, test_class_dic, _, _ = class_counter(args, 'test')
    _, _, train_class_dic, _, _ = class_counter(args, 'train')
    model = Oracle_NN_infer( args.device, len(img_view_list))
    pred_trainindex = np.load(train_index_path)[:,0]
    # the class order of loaded ptcloud is consistent with train/test_class_dic's key 
    ptcloud = np.load(args.ptcloud_path)
    train_set = ptcloud['train']         
    test_set = ptcloud['test']
    train_set = torch.from_numpy(train_set).to(args.device)
    test_set = torch.from_numpy(test_set).to(args.device)
    assert train_set.shape[0] == sum(train_class_dic.values())
    assert test_set.shape[0] == sum(test_class_dic.values())
    train_set = Normalization(train_set, inplace=True, keep_track=False).normalize_unitL2ball()
    test_set = Normalization(test_set, inplace=True, keep_track=False).normalize_unitL2ball()
    
    train_idx = 0 # front index of a train class
    test_idx = 0 # front index of a test class
    train_slicer = 0 # back index of a train class
    test_slicer = 0 # back index of a test class
    class_loss = {} 

    for idx, key in enumerate(train_class_dic):
        train_slicer += train_class_dic[key]
        test_slicer += test_class_dic[key]
        #print('key:{},train_idx:{},train_slicer:{},num:{}'.format(key, train_idx, train_slicer, train_class_dic[key]))
        #print('key:{},test_idx:{},test_slicer:{},num:{}'.format(key, test_idx, test_slicer, test_class_dic[key]))
        # input specific class train set and test set to model.forward 
        model.forward(img_view_list[test_idx:test_slicer], train_set[train_idx:train_slicer], 
              test_set[test_idx:test_slicer], pred_trainindex[test_idx:test_slicer], img_save_path)
        train_idx += train_class_dic[key]
        test_idx += test_class_dic[key]

    np.save(os.path.join(args.save_path, "pred_ptcloud.npy"), model.prediction.cpu().numpy())
    logger.info("Test Loss: %.4f"% (model.total_loss/len(img_view_list)))


def simple_infer(args, train_index_path, img_save_path, if_ptcloud=False, if_image=False, point_normalize="unitL2ball", voxel_normalize=False):
    """
    Params:
    ----------
    args: 
    train_index_path: str
        path to predict index of Oracle NN
    img_save_path: str
        path to image save path
    if_ptcloud: boolean
        if evaluate pointcloud, if True use pointcloud and Chamfer Distance, else use voxel and mIoU 
    if_image: boolean
        if use image 
    """
    class_list, _, _, _,_ = class_counter(args, "test")
    voxel_path_dic = {"train":[], 'test':[]}
    voxel_save_path_dic = {"train":[], 'test':[]}
    pred_indexes = np.load(train_index_path)

    if if_image:
        image_path_list = list()
        img_view_list = list()
        if not os.path.exists(img_save_path):
            os.mkdir(img_save_path)

        ## Load the total instance path
        for clname in class_list:
            f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt'%"test"),"r")
            for x in f:
                instance_id = x[:-1]
                image_path_list.append(os.path.join(args.data_basedir, args.img_path, clname, instance_id))
        
        for view_idx, view in enumerate(args.views):
            for i in range(len(image_path_list)):
                img_view_list.append(os.path.join(image_path_list[i], '%s.png'%view))

    if if_ptcloud:
        chamfer = ChamferDistance()
        ptcloud = np.load(args.ptcloud_path)
        if point_normalize == "unitL2ball":
            train_set = torch.from_numpy(ptcloud['train']).to(args.device)   
            test_set = torch.from_numpy(ptcloud['test']).to(args.device)
            train_set = Normalization(train_set, inplace=True, keep_track=False).normalize_unitL2ball()
            test_set = Normalization(test_set, inplace=True, keep_track=False).normalize_unitL2ball()
        elif point_normalize == "bbox":
            train_set = normalize_bbox(ptcloud['train'], BBOX, isotropic=True)
            test_set = normalize_bbox(ptcloud['test'], BBOX, isotropic=True)
            train_set = torch.from_numpy(train_set).to(args.device)   
            test_set = torch.from_numpy(test_set).to(args.device)
    else:
        for split in ['train', 'test']:
            for clname in class_list:
                f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt'%split),"r")
                for x in f:
                    instance_id = x[:-1]
                    voxel_path_dic[split].append(os.path.join(args.data_basedir, args.voxel_path, clname, instance_id, "voxel_object.binvox"))
                    voxel_save_path_dic[split].append(os.path.join(args.data_basedir, args.voxel_path, clname, instance_id, "normalized_voxel_cords.npy"))
                    
        assert pred_indexes.shape[0] == len(voxel_path_dic['test']), "voxel loading error"
        assert bool(set(voxel_path_dic['train']) & set(voxel_path_dic['test'])) == False, "overlap in train test"
        print("train set instances num: {}, test set instances num: {}".format(len(voxel_path_dic['train']), len(voxel_path_dic['test'])))

    loss_sum = 0
    loss_list = []
    fscore = {"f": {}, "p": {}, "r": {}}
    for key in fscore:
        for th in threshold_list:
            fscore[key].update({f"{th}":[]})
    for test_idx in tqdm.tqdm(range(pred_indexes.shape[0]), total=pred_indexes.shape[0]): #
        pred_trainindex = pred_indexes[test_idx, 0]
        if if_ptcloud:
            gt = test_set[test_idx].unsqueeze(0)
            prediction = train_set[pred_trainindex].unsqueeze(0)
            tmp = chamfer(gt, prediction).item()
            loss_list.append(tmp)
            loss_sum += tmp
            gt = gt.cpu().numpy()
            prediction = prediction.cpu().numpy()

            ##################  F-SCORE    ###############
            for th in threshold_list:
                ############## Oracle NN don't need to do f-score renormalization ##################
                f_p_r = f_score_list(th=th, pred_set=prediction, gt_set=gt, normalize="unitL2ball")
                fscore["f"][f"{th}"] += f_p_r[0]
                fscore["p"][f"{th}"] += f_p_r[1]
                fscore["r"][f"{th}"] += f_p_r[2]

        else:
            with open(voxel_path_dic['test'][test_idx], 'rb') as f:
                gt_object = binvox_rw.read_as_coord_array(f)
                gt = gt_object.data
            gt = gt.transpose(1, 0)
            if voxel_normalize:
                gt_u2ball = Normalization(torch.from_numpy(gt), inplace=True, keep_track=False).normalize_unitL2ball()
                gt_u2ball = gt_u2ball.numpy()[0]
                gt_normed_voxel_cords, gt_normed_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(gt_u2ball))
            else:
                gt_normed_voxel_cords, gt_normed_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(gt))
            
            with open(voxel_path_dic['train'][pred_trainindex], 'rb') as f:
                prediction_object = binvox_rw.read_as_coord_array(f)
                prediction = prediction_object.data
            prediction = prediction.transpose(1, 0)
            if voxel_normalize:
                prediction_u2ball = Normalization(torch.from_numpy(prediction), inplace=True, keep_track=False).normalize_unitL2ball()
                prediction_u2ball = prediction_u2ball.numpy()[0]
                prediction_normed_voxel_cords, prediction_normed_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(prediction_u2ball))
            else:
                prediction_normed_voxel_cords, prediction_normed_voxel = convert_ptc_to_voxel(convert_array_to_dataframe(prediction))

            tmp = evaluate_voxel_prediction(prediction_normed_voxel, gt_normed_voxel)
            loss_list.append(tmp)
            loss_sum += tmp
            print("inferred instances %d/%d; iou:%.4f" % (test_idx, pred_indexes.shape[0], tmp))
        if if_image:
            if test_idx % 200 == 0:
                image = cv2.imread(img_view_list[test_idx]).transpose((2, 0, 1))
                image = np.expand_dims(image, axis=0)
                what3d_dataset_views.data_visualizer(torch.from_numpy(gt_normed_voxel_cords.transpose(1, 0)), 
                torch.from_numpy(prediction_normed_voxel_cords.transpose(1, 0)), image, "oracle_normed_voxel", img_save_path, test_idx, loss=tmp, type="voxel_cords")
    
    if if_ptcloud:
        np.save(os.path.join(args.save_path, "f_score_{}.npy".format(point_normalize)), fscore)
        print("Chamfer max %.6f, Chamfer min %.6f, Chamfer mean %.6f"%(max(loss_list), min(loss_list), loss_sum/pred_indexes.shape[0]))
        for key in fscore:
            print(f"{key}: ")
            score_list = ""
            th_list = ""
            for th in fscore[key]:
                score_list += "%.5f  " % (sum(fscore[key][th])/len(fscore[key][th]))
                th_list += "%s   "% th
            print(th_list)
            print(score_list)
            print("----------------------------")
    else:
        print("mIoU max %.6f, mIoU min %.6f, mIoU mean %.6f"%(max(loss_list), min(loss_list), loss_sum/pred_indexes.shape[0]))
        np.save(os.path.join(args.save_path, "iou_list_norm_{}.npy".format(voxel_normalize)), np.array(loss_list))
        loss_list.sort()
        middle = len(loss_list)//2
        # lower quartile
        lower_quartile = find_median(loss_list[:middle])
        # median
        median = find_median(loss_list)
        # upper quartile
        upper_quartile = find_median(loss_list[middle:])
        print("lower_quartile %.4f, median %.4f, upper_quartile %.4f" %(lower_quartile, median, upper_quartile))


def class_index_dic(pred_trainindex_path):
    train_index = np.load(pred_trainindex_path)
    train_idx = 0       # front index of a train class
    test_idx = 0        # front index of a test class
    train_slicer = 0    # end index of a train class
    test_slicer = 0     # end index of a test class
    _, _, test_class_dic, _, _ = class_counter(args, 'test')
    _, _, train_class_dic, _, _ = class_counter(args, 'train')
    class_index_dic = {"train":{}, "test":{}}
    for idx, key in enumerate(train_class_dic):
        train_slicer += train_class_dic[key]
        test_slicer += test_class_dic[key]
        class_index_dic["train"][key] = [train_idx, train_slicer, train_class_dic[key]]
        class_index_dic["test"][key] = [test_idx, test_slicer, test_class_dic[key]]
        train_idx += train_class_dic[key]
        test_idx += test_class_dic[key]
    np.save("/home/zyf/What3D/class_index_dic.npy", class_index_dic)

def print_pral(test_idx):
    print(test_idx)

class Oracle_NN():

    def __init__(self, device, logger, save_path, total_num, prev_trainidx=None, prev_minloss=None,
                  train_set=None, test_set=None, parallel=False):
        super(Oracle_NN,self).__init__()
        self.criterion = ChamferDistance()				  #
        self.criterion = self.criterion.to(device)
        self.device = device
        self.test_index = 0

        self.test_mintrainidx = Stats()  
        if not prev_trainidx is None:
            self.test_mintrainidx.iter_loss = prev_trainidx
            self.test_index = self.test_mintrainidx.iter_loss[-1][0] + 1
            
        self.test_minloss = Stats()
        if not prev_minloss is None:
            self.test_minloss.iter_loss = prev_minloss

        if not train_set is None:
            self.train_set = train_set
            self.search_range = self.train_set.shape[0]
        if not test_set is None:
            self.test_set = test_set

        if parallel:
            self.checkpt_path = os.path.join(save_path, "checkpoint")
            if not os.path.exists(self.checkpt_path):
                os.mkdir(self.checkpt_path)
        self.epoch_callbacks = [self.save_stats]
        self.logger = logger
        self.start_time = time.time() 
        self.save_path = save_path
        self.total_num = total_num


    def invoke_epoch_callback(self):
        if len(self.epoch_callbacks)>0:
            for ith, cb in enumerate(self.epoch_callbacks):
                try:
                    cb()
                except:
                    self.logger.warn('epoch_callback[{}] failed.'.format(ith))

    def save_stats(self):
        self.test_minloss.save(os.path.join(self.save_path, 'test_minloss_testmulti.npz'))
        self.test_mintrainidx.save(os.path.join(self.save_path, 'test_mintrainidx_testmulti.npz'))

    def NN(self, train_set, test_ptcloud, train_idx, test_idx):
        if self.test_index % 200 == 0:
            self.logger.info("Trained instances %d/%d, Time passed %.6f min" % (self.test_index, self.total_num, (time.time()-self.start_time)/60))
        
        B, ptnum,_ = train_set.shape
        min_idx = 0
        min_loss = torch.tensor([1000.0], dtype=torch.float64).to(self.device)
        # enumerate through train set， compare and  find the minimum loss 
        for idx in range(B):
            loss = self.criterion(train_set[idx].unsqueeze(0), test_ptcloud) 
            if loss < min_loss:
                min_idx = idx
                min_loss = loss
        #prediction = train_set[min_idx].unsqueeze(0)
        ## add offset to index 
        min_idx = train_idx + min_idx
        self.test_minloss.push(self.test_index, loss = min_loss.item())
        self.test_mintrainidx.push(self.test_index, loss = min_idx)
        self.logger.info('test index:%d, loss:%.6f, predicted idx in trainset:%d' % (self.test_index, min_loss, min_idx))
        self.test_index += 1
        return min_idx, min_loss

    def forward(self, train_gt, test_gt, train_idx, test_idx):
        B, ptnum, dim = test_gt.shape
        # construct a empty predicted ptcloud 
        class_trainindex = torch.zeros([B, 1], dtype=torch.int32).to(self.device)
        # container to collect loss
        loss_all = torch.zeros([B], dtype=torch.float64)
        # enumerate through test set
        for index in range(B):
            class_trainindex[index], loss_all[index] = self.NN(train_gt, test_gt[index].unsqueeze(0), train_idx, test_idx) 
            self.invoke_epoch_callback()
        return torch.sum(loss_all), class_trainindex
    

    @staticmethod
    def test_OracleNN(args, pred_trainindex_path, class_index_dic_path):
        pred_trainidx = np.load(pred_trainindex_path)
        class_index_dic = np.load(class_index_dic_path, allow_pickle=True).item()
        test_nums = sum([item[2] for item in class_index_dic["test"].values()])
        ptcloud = np.load(args.ptcloud_path)
        train_set = ptcloud['train']         
        test_set = ptcloud['test']
        train_set = torch.from_numpy(train_set)
        test_set = torch.from_numpy(test_set)
        train_set = Normalization(train_set, inplace=True, keep_track=False).normalize_unitL2ball()
        test_set = Normalization(test_set, inplace=True, keep_track=False).normalize_unitL2ball()

        class_list, _, _, _,_ = class_counter(args, "test")
        image_path_list = list()
        img_view_list = list()

        ## Load the total instance path
        for clname in class_list:
            f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt'%"test"),"r")
            for x in f:
                instance_id = x[:-1]
                image_path_list.append(os.path.join(args.data_basedir, args.img_path, clname, instance_id))

        for view_idx, view in enumerate(args.views):
            for i in range(len(image_path_list)):
                img_view_list.append(os.path.join(image_path_list[i], '%s.png'%view))

        test_num = 100

        for test_order in range(test_num):
            test_random_index = random.randint(0, test_nums-1)
            for key in class_index_dic['test']:
                if class_index_dic['test'][key][0] <= test_random_index and class_index_dic['test'][key][1] > test_random_index:
                    train_index = class_index_dic["train"][key][0]
                    train_slicer = class_index_dic["train"][key][1]
                    class_to_test = key
            
            test_ptcloud = test_set[test_random_index].to(args.device)
            image = cv2.imread(img_view_list[test_random_index]).transpose((2, 0, 1))
            image = np.expand_dims(image, axis=0)    
            class_train_set = train_set[train_index:train_slicer].to(args.device)
            criterion = ChamferDistance().to(args.device)
            B, ptnum, _ = class_train_set.shape
            min_idx = 0 
            min_loss = torch.tensor([1000.0], dtype=torch.float64).to(args.device)
            # enumerate through train set， compare and find the minimum loss 
            for idx in range(B):
                loss = criterion(class_train_set[idx].unsqueeze(0), test_ptcloud.unsqueeze(0)) 
                if loss < min_loss:
                    min_idx = idx
                    min_loss = loss
            ## add offset to index 
            new_min_idx = train_index + min_idx
            origin_min_idx = pred_trainidx[test_random_index][0]
            what3d_dataset_views.data_visualizer(test_ptcloud.unsqueeze(0).cpu().numpy(), train_set[origin_min_idx].unsqueeze(0).cpu().numpy(), image, 'oracle_test', 
                        "/home/zyf/SVR/experiment/Oracle_NN/Oracle_object/oracle_test", test_order)
            if new_min_idx == origin_min_idx:
                print("Test {} PASS, Class: {}, test index: {}, pred train idx: {}".format(test_order, class_to_test, test_random_index, new_min_idx))
            else:
                print("Test {} FAIL, Class: {}, test index: {}, new pred train idx: {}, origin pred train idx: {}".format(test_order, class_to_test, test_random_index, new_min_idx,
                      pred_trainidx[test_random_index][0]))
        

def search(args):
    
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
    
    _, _, test_class_dic, _, _ = class_counter(args, 'test')
    _, _, train_class_dic, _, _ = class_counter(args, 'train')
    
    # the class order of loaded ptcloud is consistent with train/test_class_dic's key 
    ptcloud = np.load(args.ptcloud_path)
    train_set = ptcloud['train']         
    test_set = ptcloud['test']
    assert train_set.shape[0] == sum(train_class_dic.values())
    assert test_set.shape[0] == sum(test_class_dic.values())

    train_set = torch.from_numpy(train_set).to(args.device)
    test_set = torch.from_numpy(test_set).to(args.device)

    train_set = Normalization(train_set, inplace=True, keep_track=False).normalize_unitL2ball()
    test_set = Normalization(test_set, inplace=True, keep_track=False).normalize_unitL2ball()
    
    model = Oracle_NN(args.device, logger, args.save_path)

    train_idx = 0 # front index of a train class
    test_idx = 0 # front index of a test class
    train_slicer = 0 # back index of a train class
    test_slicer = 0 # back index of a test class
    class_loss = {} 

    for idx, key in enumerate(train_class_dic):
        train_slicer += train_class_dic[key]
        test_slicer += test_class_dic[key]
        logger.info('key:{},train_idx:{},train_slicer:{},num:{}'.format(key, train_idx, train_slicer, train_class_dic[key]))
        logger.info('key:{},test_idx:{},test_slicer:{},num:{}'.format(key, test_idx, test_slicer, test_class_dic[key]))
        # input specific class train set and test set to model.forward 
        sum_classloss, class_trainindex = model.forward(train_set[train_idx:train_slicer], 
                  test_set[test_idx:test_slicer], train_idx, test_idx)
        if idx == 0:
            predict_trainindex = class_trainindex
        else: 
            predict_trainindex = torch.cat((predict_trainindex, class_trainindex),0)
        train_idx += train_class_dic[key]
        test_idx += test_class_dic[key]
        class_loss[key] = sum_classloss.item()
        logger.info('key:{}, class sum loss:{}'.format(key, class_loss[key]))

    sum_loss = 0
    for clname in test_class_dic:
        sum_loss += class_loss[clname]

    avg = sum_loss/test_set.shape[0]
    logger.info('final avg loss: %.6f' % (avg))
    predict_trainindex = predict_trainindex.cpu().numpy()
    np.save(os.path.join(args.save_path, 'predict_trainindex.npy'), predict_trainindex)
    np.save(os.path.join(args.save_path, 'class_loss.npy'), class_loss)
    

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
    parser.add_argument("--class-path", dest="class_path", type=str,default='classes.txt',help="class name list")
    parser.add_argument("--views",dest="views", type=str,default= '0',
                    help="five view for each instance")
    parser.add_argument("--save-path",dest="save_path", type=str,default= 'exp2',
                    help="")  
    parser.add_argument("--voxel_normalize", action="store_true", dest="voxel_normalize", 
                     default=False, help="if unit2ball normalize gt")
    parser.add_argument("--point_normalize", type=str, dest="point_normalize", 
                     default="unitL2ball", help="normalization method on pointcloud")                 
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    print(str(args))

    #Oracle_NN.test_OracleNN(args, "/home/zyf/SVR/experiment/Oracle_NN/Oracle_object/{}/predict_trainindex.npy".format(args.save_path),
    #             "/home/zyf/What3D/class_index_dic.npy")
    #search(args)
    #infer(args, "/home/zyf/SVR/experiment/Oracle_NN/Oracle_object/exp1/predict_trainindex.npy", img_save_path="/home/zyf/SVR/experiment/Oracle_NN/Oracle_object/exp1/final_vis")

    ############ Voxel mIoU calculation on pointcloud-based Oracle NN search ############
    # simple_infer(args, "/home/zyf/SVR/experiment/Oracle_NN/Oracle_object/{}/predict_trainindex.npy".format(args.save_path), 
    # img_save_path="/home/zyf/SVR/experiment/Oracle_NN/Oracle_object/{}/normed_voxel_vis".format(args.save_path), if_ptcloud=True, if_image=False, voxel_normalize=args.voxel_normalize)

    ############## F-SCORE and Chamfer Distance based on pointcloud-based Oralce NN search #################
    simple_infer(args, "{}/predict_trainindex.npy".format(args.save_path), 
    img_save_path="{}/normed_voxel_vis".format(args.save_path), if_ptcloud=True, if_image=False, point_normalize=args.point_normalize)

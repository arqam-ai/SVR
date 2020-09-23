'''
Oracle NN

author: Yefan
created: 2020/3/10 11:21 PM
'''
import os 
import sys
#import torch
#import torch.nn as nn
#import torch.nn.functional as Functional
import numpy as np
import argparse
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.getcwd(), os.pardir, "../"))
from mpl_toolkits.mplot3d import Axes3D
#from utils.loss import ChamferDistance
import optparse
import json
import cv2
import tqdm
#from torch.utils.data import Dataset, DataLoader, TensorDataset
#import glog as logger
import logging

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


def instance_load(data_dir, data, idx, select):

    if select == 'All':
        tmp_ptcloud = np.load(os.path.join(data_dir,data[idx]['ptcloud_path']))
        tmp_image   = cv2.imread(os.path.join(data_dir,data[idx]['isometric_path']))
        tmp_image   = image_preprocessing(tmp_image)
        tmp_ptcloud = np.expand_dims(tmp_ptcloud, axis=0)
        tmp_image = np.expand_dims(tmp_image, axis=0)
        return tmp_ptcloud, tmp_image
    if select == 'ptcloud':
        tmp_ptcloud = np.load(os.path.join(data_dir,data[idx]['ptcloud_path']))
        tmp_ptcloud = np.expand_dims(tmp_ptcloud, axis=0)
        return tmp_ptcloud

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

    for i in tqdm.tqdm(range(len(data)),total= len(range(len(data))), desc = "load ground truth"):
        tmp_ptcloud = instance_load(data_dir, data, i, 'ptcloud')
        if i == 0:
            gt_ptcloud = tmp_ptcloud
#			gt_image = tmp_image
#			log_image = tmp_image
            log_ptcloud = tmp_ptcloud
        else:
            gt_ptcloud = np.concatenate((gt_ptcloud,tmp_ptcloud),axis=0)
#			gt_image = np.concatenate((gt_image,tmp_image),axis=0)

    gt_ptcloud = torch.from_numpy(gt_ptcloud).float()
#	gt_image = torch.from_numpy(gt_image).float()
    print("gt_ptcloud", gt_ptcloud.shape)
    
    return gt_ptcloud

def ratio_to_jsonname(ratio, json_name, index):
    '''
    Input: 
        ratio: float
        name: str
    Output:
        name: str
    '''
    train_json = list(json_name)
    train_json.insert(index,str(ratio))
    name = ''.join(train_json)
    return name 

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
    return instance_num, class_dic, data_class, color

def experiment(train_json, test_json, data_basedir, device):
    '''
    experiment 
    '''
    train_ratio_list = [1000, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]
    testjson_name = args.test_json
    test_gt  = load_gt(testjson_name, args.data_basedir)
    model = Oracle_NN(args.device)
    loss_list = []
    for item in train_ratio_list:
        trainjson_name = ratio_to_jsonname(item, args.train_json, -5)
        train_gt = load_gt(trainjson_name, args.data_basedir)
        train_gt = train_gt.cuda()
        test_gt = test_gt.cuda()
        loss_list.append(model.forward(train_gt, test_gt).item())
        
def experiment1():
    train_ratio_list = [1000, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]

    testjson_name = args.test_json
    test_gt  = load_gt(testjson_name, args.data_basedir)
    model = Oracle_NN(args.device)
    loss_list = []
    for item in train_ratio_list:
        trainjson_name = ratio_to_jsonname(item, args.train_json, -5)
        train_gt = load_gt(trainjson_name, args.data_basedir)
        train_gt = train_gt.cuda()
        test_gt = test_gt.cuda()
        loss_list.append(model.forward(train_gt, test_gt).item())
    #print(loss_list)
    np.save(os.path.join(args.save_dir,'Oracle_NN.npy'), np.asarray(loss_list))
#	train_ptcloud, gt_image = load_gt(args.train_json, args.data_basedir)

def dict_Avg(Dict) :
    L = len(Dict)						
    S = sum(Dict.values())				
    A = S / L
    return A

'''
class Oracle_NN(nn.Module):

    def __init__(self, device, logger):
        super(Oracle_NN,self).__init__()
        self.criterion = ChamferDistance()				  #
        self.criterion = self.criterion.to(device)
        self.device = device
        self.test_index = 0
        self.test_minloss = Stats()
        self.test_mintrainidx = Stats()  
        self.epoch_callbacks = [self.save_stats]
        self.logger = logger


    def invoke_epoch_callback(self):
        if len(self.epoch_callbacks)>0:
            for ith, cb in enumerate(self.epoch_callbacks):
                try:
                    cb()
                except:
                    self.logger.warn('epoch_callback[{}] failed.'.format(ith))

    def save_stats(self):
        self.test_minloss.save(os.path.join('test_minloss.npz'))
        self.test_mintrainidx.save(os.path.join('test_mintrainidx.npz'))

    def NN(self, train_set, test_ptcloud):

        B, ptnum,_ = train_set.shape

        min_idx = 0
        min_loss = torch.tensor([1000]).to(self.device)
        # enumerate through train set， compare and  find the minimum loss 
        for idx in range(B):
            loss = self.criterion(train_set[idx].unsqueeze(0), test_ptcloud) 
            if loss < min_loss:
                min_idx = idx
                min_loss = loss
        #prediction = train_set[min_idx].unsqueeze(0)
        self.test_minloss.push(self.test_index, loss = min_loss.item())
        self.test_mintrainidx.push(self.test_index, loss = min_idx)
        self.test_index += 1
        self.logger.info('loss:{}, idx:{}'.format(min_loss, min_idx))
        return min_idx, min_loss

    def forward(self, train_gt, test_gt):
        
        B, ptnum, dim = test_gt.shape
        # construct a empty predicted ptcloud 
        class_trainindex = torch.zeros([B, 1], dtype=torch.int32).to(self.device)
        # container to collect loss
        loss_all = torch.zeros([B], dtype=torch.float64)
        # enumerate through test set
        for index in tqdm.tqdm(range(B), total= len(range(B)), desc = "Oracle_NN"):
            class_trainindex[index], loss_all[index] = self.NN(train_gt, test_gt[index].unsqueeze(0)) 
            self.invoke_epoch_callback()
        return torch.mean(loss_all), class_trainindex
    
    def test(self, gt_ptcloud, trainidx_path):
        pass
    '''

def visualizer(gt, prediction, path, split_name, idx):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=5, alpha=0.5)
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    ax.set_title("GT")
    plt.axis("off")
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(prediction[:, 0], prediction[:, 1], prediction[:, 2], s=5, alpha=0.5)
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    ax.set_title("Prediction")
    plt.axis("off")
    ax = fig.add_subplot(133)
    image = np.ones((224,224,3))
    ax.imshow(image)
    plt.axis("off")
    title = os.path.join(path, "%s_%d" % (split_name,idx))
    fig.savefig(title, bbox_inches='tight',pad_inches = 0, dpi = 200)
    plt.close()

class Oracle_NN_infer(object):

    def __init__(self, device, logger):
        super(Oracle_NN_infer,self).__init__()
        self.test_index = 0

    def forward(self, train_gt, test_gt, pred_trainindex):
        
        for local_test_index, train_index in enumerate(pred_trainindex):
            if self.test_index % 200 == 0:
                prediction = train_gt[train_index]
                gt = test_gt[local_test_index]
                visualizer(gt, prediction, "../experiment/Oracle_NN/Oracle_object/final_vis", "test", self.test_index/200)
            self.test_index += 1


def infer(args, train_index_path):
    _, test_class_dic, _, _ = class_counter(args, 'test')
    _, train_class_dic, _, _ = class_counter(args, 'train')
    model = Oracle_NN_infer(None, None)
    kwargs = {'num_workers':4, 'pin_memory':True}
    pred_trainindex = np.load(train_index_path)[:,0]
    # the class order of loaded ptcloud is consistent with train/test_class_dic's key 
    ptcloud = np.load(args.ptcloud_path)
    train_set = ptcloud['train']         
    test_set = ptcloud['test']
    assert train_set.shape[0] == sum(train_class_dic.values())
    assert test_set.shape[0] == sum(test_class_dic.values())
    
    train_idx = 0 # front index of a train class
    test_idx = 0 # front index of a test class
    train_slicer = 0 # back index of a train class
    test_slicer = 0 # back index of a test class
    class_loss = {} 

    for idx, key in enumerate(train_class_dic):
        train_slicer += train_class_dic[key]
        test_slicer += test_class_dic[key]
        print('key:{},train_idx:{},train_slicer:{},num:{}'.format(key, train_idx, train_slicer, train_class_dic[key]))
        print('key:{},test_idx:{},test_slicer:{},num:{}'.format(key, test_idx, test_slicer, test_class_dic[key]))
        # input specific class train set and test set to model.forward 
        model.forward(train_set[train_idx:train_slicer], 
              test_set[test_idx:test_slicer], pred_trainindex[test_idx:test_slicer])
        train_idx += train_class_dic[key]
        test_idx += test_class_dic[key]


def search(args):
    
    logger = logging.getLogger()
    file_log_handler = logging.FileHandler('Train.log')
    logger.addHandler(file_log_handler)
    logger.setLevel('INFO')
    formatter = logging.Formatter()
    file_log_handler.setFormatter(formatter)
    #stderr_log_handler = logging.StreamHandler(sys.stdout)
    #logger.addHandler(stderr_log_handler)
    #stderr_log_handler.setFormatter(formatter)
    logger.info("Running script for trainning the Oracle NN")
    # class_dic : {'class name ': instance number of test/train set}
    _, test_class_dic, _, _ = class_counter(args, 'test')
    _, train_class_dic, _, _ = class_counter(args, 'train')
    
    kwargs = {'num_workers':4, 'pin_memory':True}
    # the class order of loaded ptcloud is consistent with train/test_class_dic's key 
    ptcloud = np.load(args.ptcloud_path)
    train_set = ptcloud['train']         
    test_set = ptcloud['test']
    assert train_set.shape[0] == sum(train_class_dic.values())
    assert test_set.shape[0] == sum(test_class_dic.values())
    train_set = torch.from_numpy(train_set).to(args.device)
    test_set = torch.from_numpy(test_set).to(args.device)
    model = Oracle_NN(args.device, logger)

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
        avg_loss, class_trainindex = model.forward(train_set[train_idx:train_slicer], test_set[test_idx:test_slicer])
        if idx == 0:
            predict_trainindex = class_trainindex
        else:
            predict_trainindex = torch.cat((predict_trainindex, class_trainindex),0)
        train_idx += train_class_dic[key]
        test_idx += test_class_dic[key]
        class_loss[key] = avg_loss.item()
        logger.info('key:{}, avg loss:{}'.format(key, class_loss[key]))

    weight_loss = 0
    for clname in test_class_dic:
        weight_loss += class_loss[clname] * test_class_dic[clname]

    avg = weight_loss/sum(test_class_dic.values())
    logger.info('final avg loss: %.3f' % (avg))
    predict_trainindex = predict_trainindex.cpu().numpy()
    np.save('predict_trainindex.npy', predict_trainindex)
    np.save('class_loss.npy', class_loss)
    
    
    

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--data-basedir',type=str,default='../../What3D',
                    help='path of the jsonfile')
    parser.add_argument('--img-path',type=str,default='renderings',
                    help='path of the jsonfile')
    parser.add_argument('--splits-path',type=str,default='splits',
                    help='path of the jsonfile')
    parser.add_argument('--class-path',type=str,default='classes.txt',
                    help='path of the jsonfile')
    parser.add_argument("--ptcloud-path",type=str,
                        default="../../What3D/ptcloud_object.npz",
                        help=' ' )
    args = parser.parse_args(sys.argv[1:])
    #args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    print(str(args))
    #search(args)
    infer(args, "../experiment/Oracle_NN/Oracle_object/predict_trainindex.npy")
    
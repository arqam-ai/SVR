""" The script is used to evaluate viewer center oracle NN f-score
    Author: Yefan Zhou
"""

import os
import sys
import argparse
import torch
import numpy as np
import tqdm
sys.path.append("../../../")
from utils.utils import Normalization
from utils.loss import ChamferDistance
from utils.f_score import f_score

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument("--class_index_dic_path", type=str, default="../../../../What3D/class_index_dic.npy", help="")
parser.add_argument("--save-path",dest="save_path", type=str,default= 'exp2', help="")
parser.add_argument("--views",dest="views", type=str,default= '01234',help="five view for each instance")
parser.add_argument("--ptcloud-path", type=str,default="../../../../What3D/ptcloud_n.npz",help=' ' )
parser.add_argument("--normalization", action="store_true", default=False)
args = parser.parse_args(sys.argv[1:])
print(args)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chamfer = ChamferDistance().to(args.device)
class_index_dic = np.load(args.class_index_dic_path, allow_pickle=True).item()
class_eval_stats = np.load(os.path.join(args.save_path, "class_eval_stats.npy"), allow_pickle=True).item()
class_eval_index = class_eval_stats["index"]
class_name_list = list(class_index_dic['train'].keys())

## Split each view ptcloud to class wise layout 
ptcloud_dic = {'train':{}, 'test':{}}
ptcloud_view_list = [] 
for view in list(args.views):
    print(f"Loading point cloud view: {view}")
    ptcloud_tmp = {"train":[], "test":[]}
    ptcloud_path = args.ptcloud_path[:-5] + '%s.npz'%view
    ptcloud_npz = np.load(ptcloud_path)
    train_set = torch.from_numpy(ptcloud_npz['train'])
    test_set = torch.from_numpy(ptcloud_npz['test'])
    if args.normalization:
        ptcloud_tmp['train'] = Normalization(train_set, inplace=True, keep_track=False).normalize_unitL2ball()
        ptcloud_tmp['test'] = Normalization(test_set, inplace=True, keep_track=False).normalize_unitL2ball()
    else:
        ptcloud_tmp['train'] = train_set
        ptcloud_tmp['test'] = test_set

    ptcloud_view_list.append(ptcloud_tmp)

gt_set = []
pred_set = []

for class_index, class_name in enumerate(class_name_list):
    for view_idx, view in enumerate(list(args.views)):
        train_set = ptcloud_view_list[view_idx]['train']
        test_set = ptcloud_view_list[view_idx]['test']
        train_idx = class_index_dic['train'][class_name][0]
        train_slicer = class_index_dic['train'][class_name][1]
        test_idx = class_index_dic['test'][class_name][0]
        test_slicer = class_index_dic['test'][class_name][1]
        if view == '0':
            ptcloud_dic['train'][class_name] = train_set[train_idx:train_slicer]
            ptcloud_dic['test'][class_name] = test_set[test_idx:test_slicer]
        else:
            ptcloud_dic['train'][class_name] = torch.cat((ptcloud_dic['train'][class_name], train_set[train_idx:train_slicer]), 0)
            ptcloud_dic['test'][class_name] =  torch.cat((ptcloud_dic['test'][class_name], test_set[test_idx:test_slicer]), 0)

    ptcloud_dic['train'][class_name] = ptcloud_dic['train'][class_name].numpy()
    ptcloud_dic['test'][class_name] = ptcloud_dic['test'][class_name].numpy()

    for test_idx in range(ptcloud_dic['test'][class_name].shape[0]):
        gt_set.append(ptcloud_dic['test'][class_name][test_idx])
        train_idx = class_eval_index[class_name][test_idx]
        pred_tmp = ptcloud_dic['train'][class_name][train_idx]
        if len(pred_tmp.shape) > 2:
            pred_tmp = np.squeeze(pred_tmp)
        pred_set.append(pred_tmp)

gt_set = np.array(gt_set)
pred_set = np.array(pred_set)
assert gt_set.shape == pred_set.shape

#threshold_list, thre_mean = f_score(th=None, pred_set=pred_set, gt_set=gt_set, out_path=None)
#print(threshold_list)
#print(thre_mean)
np.save("gt_set.npy", gt_set)
np.save("pred_set.npy", pred_set)

###################Chamfer Distance Recheck#######################
# gt_set = torch.from_numpy(gt_set).to(args.device)
# pred_set = torch.from_numpy(pred_set).to(args.device)
# num_instance = gt_set.shape[0]
# sum_loss = 0
# for idx in tqdm.tqdm(range(num_instance), total=num_instance):
#     sum_loss += chamfer(gt_set[[idx]], pred_set[[idx]]).item()
# print("avg loss %.4f" % (sum_loss/num_instance))





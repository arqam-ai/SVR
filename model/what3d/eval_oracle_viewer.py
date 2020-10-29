""" The script is used to evaluate viewer center oracle NN Chamfer Distance
    Author: Yefan Zhou
"""
import os
import sys
import argparse
import torch
import numpy as np
import tqdm
sys.path.append("../../../")
from utils.loss import ChamferDistance
from dataset.dataset import load_class_ptcloud, what3d_dataset_views
from utils.utils import Normalization
parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument("--ptcloud-path",type=str,default="../../../../What3D/ptcloud_n.npz",help=' ' )
parser.add_argument("--save-path",dest="save_path", type=str,default= 'exp2', help="")
parser.add_argument("--class_index_dic_path", type=str, default="../../../../What3D/class_index_dic.npy", help="")
parser.add_argument("--checkpt_path", type=str, default="checkpoint")
parser.add_argument("--views",dest="views", type=str,default= '01234',help="five view for each instance")
parser.add_argument("--img_save_path", type=str, default="exp2_vis")
args = parser.parse_args(sys.argv[1:])
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(save_path, class_name, class_index_dic):

    ####################Load Checkpoints######################
    trainidx_path = os.path.join(save_path, class_name, "test_mintrainidx.npz")
    trainloss_path = os.path.join(save_path, class_name, "test_minloss.npz")
    if os.path.exists(trainidx_path):
        prev_trainidx = np.load(trainidx_path)["iter_loss"]
        prev_trainidx = prev_trainidx.tolist()
        prev_trainloss = np.load(trainloss_path)["iter_loss"]
        prev_trainloss = prev_trainloss.tolist()
        start_testidx = prev_trainidx[-1][0] + 1
    else:
        prev_trainidx = []
        start_testidx = 0
    
    instance_num = 5 * class_index_dic['test'][class_name][2]
    return start_testidx, instance_num, prev_trainidx, prev_trainloss

### visualize the top 10 and last 10, eval_loss_list
def visualize_test(eval_loss_list, train_set, test_set, class_trainindex, class_name):
    sorted_index_list = sorted(range(len(eval_loss_list)), key=lambda i: eval_loss_list[i], reverse=True)
    for image_id, test_index in enumerate(sorted_index_list[:10]):
        what3d_dataset_views.data_visualizer(test_set[test_index].cpu(), train_set[class_trainindex[test_index]].cpu(), 
                image=torch.zeros(1, 3, 224, 224), split_name="{}_bad".format(class_name), path=args.img_save_path, idx=image_id,
                  loss=eval_loss_list[test_index])

    for image_id, test_index in enumerate(sorted_index_list[-10:]):
         what3d_dataset_views.data_visualizer(test_set[test_index].cpu(), train_set[class_trainindex[test_index]].cpu(), 
                image=torch.zeros(1, 3, 224, 224), split_name="{}_top".format(class_name), path=args.img_save_path, idx=image_id,
                  loss=eval_loss_list[test_index])
        

chamfer = ChamferDistance().to(args.device)
hugeclass_index_list = [47, 16, 0, 18, 49]
class_index_dic = np.load(args.class_index_dic_path, allow_pickle=True).item()
assert list(class_index_dic['train'].keys()) == list(class_index_dic['test'].keys()), "class key list error"
class_name_list = list(class_index_dic['train'].keys())
print("Class List:")
print(class_name_list)
print("------------------------------------------")
## Load ptcloud as numpy array and normalize to torch tensor
ptcloud_view_list = [] 
for view in list(args.views):
    print(f"Loading point cloud view: {view}")
    ptcloud_tmp = {"train":[], "test":[]}
    ptcloud_path = args.ptcloud_path[:-5] + '%s.npz'%view
    ptcloud_npz = np.load(ptcloud_path)
    train_set = torch.from_numpy(ptcloud_npz['train']).to(args.device)
    test_set = torch.from_numpy(ptcloud_npz['test']).to(args.device)
    ptcloud_tmp['train'] = Normalization(train_set, inplace=True, keep_track=False).normalize_unitL2ball()
    ptcloud_tmp['test'] = Normalization(test_set, inplace=True, keep_track=False).normalize_unitL2ball()
    ptcloud_view_list.append(ptcloud_tmp)

## Split each view ptcloud to class wise layout 
ptcloud_dic = {'train':{}, 'test':{}}
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

print(f"Spliting point cloud class Done ------------------------------------------")

class_eval_stats = {"index": {}, "sumloss": {}, "nums": {}}
sum_loss = 0
sum_num = 0
for class_index, class_name in enumerate(class_name_list):
    eval_loss_list = []
    print(f"Starting evaluate class {class_name} again")
    if class_index not in hugeclass_index_list:
        class_loss = np.load(os.path.join(args.save_path, class_name, "class_loss_dic.npy"), allow_pickle=True).item()
        sum_classloss = class_loss["sumloss"][class_name] 
        class_trainindex = class_loss["index"][class_name] 
        sum_instances = class_loss["nums"][class_name] 
        avg = class_loss["avgloss"][class_name]
        assert ptcloud_dic['test'][class_name].shape[0] == sum_instances
        for test_idx in range(sum_instances):
            eval_loss_list.append(chamfer(ptcloud_dic['test'][class_name][test_idx].unsqueeze(0), ptcloud_dic['train'][class_name][class_trainindex[test_idx]]).item())
        #if not (sum_classloss == sum(eval_loss_list) and avg == sum(eval_loss_list)/sum_instances):

    else:
        start_testidx, sum_instances, prev_trainidx, prev_trainloss = load_checkpoint(args.save_path, class_name, class_index_dic)
        for load_test_idx in range(start_testidx, sum_instances):
            tmp_stats = np.load(os.path.join(args.save_path, class_name, args.checkpt_path, "checkpt_{}.npy".format(load_test_idx)))
            prev_trainidx.append([int(tmp_stats[0]), int(tmp_stats[2])])
            prev_trainloss.append([int(tmp_stats[0]), tmp_stats[1]])

        check_index_consistent = [int(pair[0]) for pair in prev_trainidx]
        assert check_index_consistent == list(range(sum_instances)), "huge class test index inconsistent"
        assert len(prev_trainidx) == sum_instances
        assert len(prev_trainloss) == sum_instances

        class_trainindex = [loss_pair[1] for loss_pair in prev_trainidx]
        sum_classloss = sum([loss_pair[1] for loss_pair in prev_trainloss])
        avg = sum_classloss / sum_instances
        for idx_pair in tqdm.tqdm(prev_trainidx, total=len(prev_trainidx)):
            eval_loss_list.append(chamfer(ptcloud_dic['test'][class_name][idx_pair[0]].unsqueeze(0), ptcloud_dic['train'][class_name][idx_pair[1]].unsqueeze(0)).item())

    print("original sum loss %.4f, reeval sum loss %.4f" %(sum_classloss, sum(eval_loss_list))) 
    print("original avg loss %.4f, reeval avg loss %.4f" %(avg, sum(eval_loss_list)/sum_instances))
    print("----------------------------------------------------------")

    sum_loss += sum_classloss
    sum_num += sum_instances
    #visualize_test(eval_loss_list, ptcloud_dic['train'][class_name], ptcloud_dic['test'][class_name], class_trainindex, class_name)

    #class_eval_stats['index'][class_name] = class_trainindex
    #class_eval_stats['sumloss'][class_name] = sum_classloss
    #class_eval_stats['nums'][class_name] = sum_instances

print("Average Chamfer Loss: %.4f" % (sum_loss/sum_num))
#np.save(os.path.join(args.save_path, "class_eval_stats.npy"), class_eval_stats)


import torch

import os
import sys
sys.path.append("../")

import numpy as np
import pickle
import argparse
import tqdm
from datetime import datetime
from utils.plot_image_grid import visuaize_pts
from utils.utils import check_exist_or_mkdirs
from utils.loss import ChamfersDistance3
import matplotlib.pyplot as plt
from collections import defaultdict
import collections
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def class_counter(data_basedir, splits_path, class_path, split_name):
    """return What 3D dataset class and instance information

    Params:
    ------------------
    data_basedir : string
        path to data base folder
    splits_path : string
        data folder name to contain splits info
    class_path : string
        class info file name
    splits_name : string
        "train" or "test"

    Returns:
    ------------------
    instance_num : int
        total number of instances
    class_dic: python dictionary
        {class_name: num of instance in this class}
    data_class: list  length = instance_num
        list of all sample classname i.e. [airplane, chair]
    fileid_list: list of string
        list of "classname/fileid"
    """
    f_class = open(os.path.join(data_basedir, splits_path, class_path), "r")

    class_num = 0
    class_dic = {}  # class_name : num of instance in this class
    class_index = {}  # class_name : class_index     i.e. airplane:0
    class_list = []  # 55 class
    data_class = []  # list of all sample classname i.e. [airplane airplane chair]
    # color = []
    for line in f_class:
        index = line.find(' ')
        clname = line[:index]
        class_dic[clname] = 0
        class_list += [clname]
        class_index[clname] = class_num
        class_num += 1
    fileid_list = []
    instance_num = 0
    for clname in tqdm.tqdm(class_list, total=len(class_list), desc='%s' % split_name):
        f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt' % split_name), "r")
        for x in f:
            fileid_list.append(clname + '/' + x[:-1])
            class_dic[clname] += 1
            instance_num += 1
            data_class += [clname]
    #        color += [class_index[clname]]
    assert instance_num == len(data_class), "the class list length mismatch instances num"
    return instance_num, class_dic, data_class, fileid_list


def load_what3d_assignment(txtfile_path, gt_fileid_list):
    """convert cluster assignment txt file to list

    Params:
    ------------------
    txtfile_path: string
        path to cluster assignment
    gt_fileid_list: string
        file id list loaded from split info

    Return:
    ------------------
    fileid_list: list
        list of tuple (file_name, cluster_idx)
    cluster: python dictionary
        dictionary {cluster_id:[sample_idx]}
    """
    ## Convert txt file to list of tuple tuple(file_name, cluster_idx)
    f = open(txtfile_path)
    fileid_list = []  # file id list
    cluster = {}
    ##
    for idx, x in enumerate(f):
        if idx % 5 == 0:
            x = x[:-1]
            # append tuple(filename, cluster_idx)
            fileid_list.append((x[70:x.find(' ') - 6], x[x.find(' ') + 1:]))

            ## Check if the what3d file order match file order from split info
    for i, what3d_id in enumerate(fileid_list):
        if what3d_id[0] != gt_fileid_list[i]:
            print("One mismatch detected, the index is ", i)
            del fileid_list[i]
            break
    ## After remove mismatch instance, check it again
    for t, what3d_id in enumerate(fileid_list):
        assert what3d_id[0] == gt_fileid_list[t]

    print("Number of Instance is ", len(fileid_list))

    ## convert list to dictionary
    for sample_index, tuple in enumerate(fileid_list):
        if int(tuple[1]) not in cluster.keys():
            cluster[int(tuple[1])] = []
            cluster[int(tuple[1])] += [sample_index]
        else:
            cluster[int(tuple[1])].append(sample_index)

    return fileid_list, cluster

def load_prediction(filepath):
    """load prediction index
    Params:
    ------------------
    filepath : string
        path to prediction index file

    Returns:
    ------------------
    pred_list : list
        list of cluster idx (prediction)
    """
    f = open(filepath)
    pred_list = []
    counter = 0
    for idx, pred in enumerate(f):
        if idx % 5 == 0:
            pred_list.append(int(pred[:-1]))

    return pred_list


def main(args):
    check_exist_or_mkdirs(args.matrix_save_path)

    ## LOAD class information
    test_num, test_class_dic, test_class_list, test_fileid_list = class_counter(data_basedir=args.data_basedir,
                                                                                splits_path=args.splits_path,
                                                                                class_path=args.class_path,
                                                                                split_name='test')

    # with open("test-data_filename.pkl", "wb") as f:
    #     pickle.dump(test_fileid_list, f)

    train_num, train_class_dic, train_class_list, train_fileid_list = class_counter(data_basedir=args.data_basedir,
                                                                                    splits_path=args.splits_path,
                                                                                    class_path=args.class_path,
                                                                                    split_name='train')
                                                     
    ## LOAD point cloud
    ptcloud = np.load(args.ptcloud_path)
    train_set = ptcloud['train']
    test_set = ptcloud['test']

    ## CHECK the point cloud set size corresponding to instance number in class info
    assert train_set.shape[0] == sum(
        train_class_dic.values()), 'point cloud number mismatches the class dictionary counter'
    assert test_set.shape[0] == sum(
        test_class_dic.values()), 'point cloud number mismatches the class dictionary counter'

    ## LOAD train set cluster assignment and testset prediction given by what3d author
    train_cluster_list, cluster_dic = load_what3d_assignment(
        txtfile_path=os.path.join(args.clusterfile_path, 'train.txt'), gt_fileid_list=train_fileid_list)
    pred_list = load_prediction(args.prediction_path)

    total_cluster_counts = {}

    ## Prediction Pipeline
    test_cluster_data_index = defaultdict(list)
    test_cluster_type = defaultdict(list)
    for idx in tqdm.tqdm(range(test_set.shape[0]), desc='Predicting in clustering method'):
        class_type = test_fileid_list[idx].split("/")[0]
        cluster_idx = pred_list[idx]  # cluster_idx 0~499
        # trainset_idx = cluster_dic[cluster_idx]  # list of idx train set 0 ~ 36757
        # pred_cluster = train_set[trainset_idx]  # all the samples of the corresponding cluster
        test_cluster_type[cluster_idx].append(class_type)
        test_cluster_data_index[cluster_idx].append(idx)

    with open("clustering_data_index.pkl", "wb") as f:
        pickle.dump(test_cluster_data_index, f)


    for idx, cluster_typ in enumerate(test_cluster_type):
        # print("Cluster {} .............................................".format(idx))
        class_list = test_cluster_type[cluster_typ]
        counter = collections.Counter(class_list)
        print(idx, counter)
        # for keys in counter.keys():
        #     print(keys, counter[keys])
            # if keys not in total_cluster_counts:
            #     total_cluster_counts[keys] = counter[keys]
            # else:
            #     total_cluster_counts[keys] += counter[keys]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--data-basedir',type=str,default='/home/yirus/Datasets/what3d',
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
                        default="/home/yirus/Datasets/what3d/ptcloud_0.npz",
                        help=' ' )
    parser.add_argument("--matrix-save-path",type=str,
                        default="cluster_matrix",
                        help=' ' )
    parser.add_argument("--batch-size", type=int,
                        default=1,
                        help=' ' )
    parser.add_argument("--save-results", action = 'store_true',
                        default=False,
                        help=' ' )
    args = parser.parse_args(sys.argv[1:])

    print(torch.cuda.is_available())

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    print(str(args))

    print("================================")
    print(args.device)

    main(args)



'''
Clustering
This is a script to implement a method to infer pointcloud.
The method is adapted from paper "what do single-view 3D reconstruction do?" Clustering method
link: https://arxiv.org/abs/1905.03678
First use sklearn.kMedoids to calculate K mean shape (actually most representative data in each cluster) in training set,
Then in test, search nearest shape in K mean shape set. 
'''
import os
import sys
sys.path.append("../")
import torch
import numpy as np
import argparse
import tqdm
import random
from utils.plot_image_grid import visuaize_pts
from utils.utils import check_exist_or_mkdirs
from utils.loss import ChamfersDistance3
import matplotlib.pyplot as plt 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
    f_class = open(os.path.join(data_basedir, splits_path, class_path),"r")
    
    class_num = 0
    class_dic = {}           # class_name : num of instance in this class
    class_index = {}         # class_name : class_index     i.e. airplane:0
    class_list = []          # 55 class
    data_class = []          # list of all sample classname i.e. [airplane airplane chair]
    #color = []
    for line in f_class:
        index = line.find(' ')
        clname = line[:index]
        class_dic[clname] = 0
        class_list += [clname]
        class_index[clname] = class_num
        class_num += 1
    fileid_list = []
    instance_num = 0
    for clname in tqdm.tqdm(class_list,total= len(class_list), desc = '%s'%split_name):
        f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt'%split_name),"r")
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
    fileid_list = []     #  file id list
    cluster = {}
    ## 
    for idx, x in enumerate(f):
        if idx % 5 == 0:
            x = x[:-1]
            # append tuple(filename, cluster_idx)
            fileid_list.append((x[70:x.find(' ')-6], x[x.find(' ')+1: ]))  

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

def compute_ptcloud_dismatrix(X1, X2, distance_metric, title=None, results_dir=None, ifsave=False):
    """return distance matrix between ptcloud X1 ptcloud X2
    Params:
    ----------------------------------
    X1: (N, ptnum,3) torch.tensor
        point cloud set 1
    X2: (N, ptnum,3) torch.tensor
        point cloud set 2
    distance_metric: func
        metric to measure the distance of two point cloud
    ifsave: boolean
        if to save the distance matrix to disk
    title: string
        name of the saved matrix
    results_dir:string
        the path to save the distance matrix
        
    Returns:
    ----------------------------------
    D: (ptnum, ptnum) torch.tensor
        distance matrix
    """
    N = X1.shape[0]
    # initialize distance matrix
    D = torch.zeros([N, N])
    # iterate over one group of ptcloud
    for i in range(N):	
        for j in range(N):
            D[i,j] = distance_metric(X1[i].unsqueeze(0), X2[j].unsqueeze(0))
    if ifsave:
        D = D.cpu().numpy()
        np.save(os.path.join(results_dir,title), D)
        print("saved to " + os.path.join(results_dir, title)) 
    return D

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



def main(args):
    check_exist_or_mkdirs(args.matrix_save_path)
    stats_test_loss = Stats()

    ## LOAD class information 
    test_num, test_class_dic, test_class_list, test_fileid_list  = class_counter(data_basedir = args.data_basedir, 
            splits_path = args.splits_path, class_path = args.class_path, split_name ='test')
    train_num, train_class_dic, train_class_list, train_fileid_list  = class_counter(data_basedir = args.data_basedir, 
            splits_path = args.splits_path, class_path = args.class_path, split_name ='train')
    ## LOAD point cloud 
    ptcloud = np.load(args.ptcloud_path)
    train_set = ptcloud['train']         
    test_set = ptcloud['test']

    ## CHECK the point cloud set size corresponding to instance number in class info
    assert train_set.shape[0] == sum(train_class_dic.values()), 'point cloud number mismatches the class dictionary counter'
    assert test_set.shape[0] == sum(test_class_dic.values()), 'point cloud number mismatches the class dictionary counter'
    
    ## LOAD train set cluster assignment and testset prediction given by what3d author
    train_cluster_list, cluster_dic = load_what3d_assignment(
            txtfile_path = os.path.join(args.clusterfile_path,'train.txt'), gt_fileid_list = train_fileid_list)
    pred_list = load_prediction(args.prediction_path)

    ## Move point cloud and chamfer function to GPU
    train_set = torch.from_numpy(train_set).to(args.device)
    test_set = torch.from_numpy(test_set).to(args.device)
    chamfer = ChamfersDistance3().to(args.device)
    
    ## Prediction Pipeline
    computed_cluster_idx = []                           # log which cluster matrix has been computed
    pred_ptcloud_set = torch.zeros_like(test_set)       # make space for prediction set 
    sum_loss = 0                                            # initialize the loss
    for idx in tqdm.tqdm(range(test_set.shape[0]), desc='Predicting in clustering method'):
        cluster_idx = pred_list[idx]                    # cluster_idx 0~499
        trainset_idx = cluster_dic[cluster_idx]         # list of idx train set 0 ~ 36757
        pred_cluster = train_set[trainset_idx]     # all the samples of the corresponding cluster
        if cluster_idx in computed_cluster_idx:         # if the cluster matrix has been computed, load it directly         
            pred_matrix = np.load(os.path.join(args.matrix_save_path, 'cluster_matrix_{:03}.npy'.format(cluster_idx)))
        else:                                           # otherwise compute it 
            pred_matrix = compute_ptcloud_dismatrix(X1=pred_cluster, X2=pred_cluster, 
                                        distance_metric=chamfer, title='cluster_matrix_{:03}.npy'.format(cluster_idx), 
                                    results_dir=args.matrix_save_path, ifsave=True)
            computed_cluster_idx.append(cluster_idx)    # log the cluster has been computed 

        ## find one sample in a cluster which has the minimum sum of distance to all the samples in the cluster
        index = np.argmin(np.sum(pred_matrix, axis=1, keepdims=True), axis=0)
        pred_ptcloud = pred_cluster[index]
        ## save it to prediction set
        pred_ptcloud_set[idx] = pred_ptcloud[0]
        ## compute the chamfer loss between prediction and GT
        loss = chamfer(pred_ptcloud, test_set[idx].unsqueeze(0))
        sum_loss += loss
        ## log test loss
        stats_test_loss.push(idx, loss = loss.item())
        stats_test_loss.save(os.path.join(args.matrix_save_path, 'stats_test_loss.npz'))
    pred_ptcloud_set = pred_ptcloud_set.cpu().numpy()
    np.save(os.path.join(args.matrix_save_path, 'clustering_prediction.npy'), pred_ptcloud_set)
    print('average test loss: ', sum_loss.item()/test_set.shape[0])    
    
    



def test():
    #### A test to check the class order of point cloud set is identical to Variable train_class_list########
    ##############################################################################################################
    '''
    random_idx = [random.choice(range(train_set.shape[0])) for _ in range(20)]
    random_set = train_set[random_idx]
    corres_classname = ''
    for idx in random_idx:
        corres_classname += train_class_list[idx]
        corres_classname += '\t'
    visuaize_pts(random_set, title = corres_classname, rows=4, cols =5)
    plt.savefig('test_sample_classname.png')
    '''
    ##############################################################################################################
    #### A test to check the prediction given by author of what3d is visually close to gt #######################
    ##############################################################################################################
    '''
    print('cluster index: ', min(list(cluster_dic.keys())), max(list(cluster_dic.keys())))
    for idx, pred in enumerate(pred_list):
        gt = test_set[[idx]]
        pred_ptcloud = train_set[cluster_dic[pred]]
        if idx in [0, 100, 1000, 6000, 4000, 2000, 8000, 10000]:
            visuaize_pts(np.concatenate((gt,pred_ptcloud[random.sample(range(pred_ptcloud.shape[0]), 10)]), axis =0), title = 'class: {}, first is gt, following is pred'.format(test_class_list[idx]))
            plt.savefig('../img/check_prediction_{}.png'.format(idx))
    '''
    ##############################################################################################################





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
    parser.add_argument("--batch-size", type=int,
                        default=1,
                        help=' ' )
    parser.add_argument("--save-results", action = 'store_true',
                        default=False,
                        help=' ' )
    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    print(str(args))
    main(args)
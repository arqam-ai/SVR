import open3d as o3d
import argparse
import os
import sys
import tqdm
import numpy as np
import random

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument("--ptcloud-path",dest="ptcloud_path", type=str,default="points_object_1024_upbound",
                        help='path of the ptcloud')
parser.add_argument("--img-path", dest="img_path", type=str, default='renderings',
                    help='path of the image')
parser.add_argument("--label-path", dest="label_path", type=str,default='label.npz',
                    help='path of the image')
parser.add_argument("--data-basedir",dest="data_basedir", type=str,default='/home/../../public/zyf/What3D',
                    help='path of the data folder') 
parser.add_argument("--splits-path",dest="splits_path", type=str, default='splits',
                    help='path of the data folder') 
parser.add_argument("--class-path", dest="class_path", type=str,default='classes.txt',
                    help="class name list")
parser.add_argument("--image-size",dest="image_size", type=int,default = 224,
                    help="image size for network")
parser.add_argument("--sample-ratio",dest="sample_ratio", type=float, default = 0.001,
                    help="ratio to sample the dataset")
parser.add_argument("--views",dest="views", type=str,default= '0',help="five view for each instance")
parser.add_argument("--pts-num",dest="pts_num", type=int,default=1024,
                    help="number of points in a ptcloud")
parser.add_argument("--mode", dest="mode", type=str,default="viewer", 
                    help="['viewer', 'object']")
parser.add_argument("--read-view", action="store_true",dest="read_view",
                    default= False, help="if adding the view information in training", )
parser.add_argument("--train-batch-size",dest="train_batch_size", type=int,
                    default=1, help='training batch size')
args = parser.parse_args(sys.argv[1:])
data_mode = ['train','test','val']

def compress(args, origin_name = '1.ply', compressed_name = 'ptcloud_1.npz'):
    class_list, _ = class_counter(os.path.join(args.data_basedir, args.splits_path, args.class_path))
    ptcloud_path_list = {'train':[], 'test':[], 'val':[]}
    ptcloud_set_dic = {'train':[], 'test':[], 'val':[]}

    for split_name in data_mode:
        for clname in class_list:
            f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt'%split_name),"r")
            for x in f:
                instance_id = x[:-1]
                ptcloud_path_list[split_name].append(os.path.join(args.data_basedir, args.ptcloud_path, clname, instance_id))

    for split_name in data_mode:
        pt_index = 0
        split_num =len(ptcloud_path_list[split_name])
        pbar = tqdm.tqdm(total=split_num, desc='%s'%split_name)
        ptcloud_set = np.zeros((split_num, 1024, 3)) 
        for folder in ptcloud_path_list[split_name]:
            pcd = o3d.io.read_point_cloud(os.path.join(folder, origin_name))
            ptcloud_set[pt_index] = np.asarray(pcd.points)
            pbar.update(1)
            pt_index += 1
        ptcloud_set_dic[split_name] = ptcloud_set

    np.save(os.path.join(args.data_basedir, compressed_name), ptcloud_set_dic['test'])
    #np.savez(os.path.join(args.data_basedir, compressed_name), train = ptcloud_set_dic['train'], 
    #                                        val = ptcloud_set_dic['val'], test = ptcloud_set_dic['test'])

def check(args, view):
    '''check if the '''

    for mode in ['train', 'test', 'val']:
        if mode == 'train':
            index = random.sample(range(30000), 1)[0]   #10432 5241
        elif mode == 'test':
            index = random.sample(range(10432), 1)[0]   #10432 5241
        elif mode == 'val':
            index = random.sample(range(5241), 1)[0]   #10432 5241
        ptcloud = np.load(os.path.join(args.data_basedir, 'ptcloud_%s.npz'%'object'))
        ptcloud_npz = ptcloud[mode][index]
        class_list, _ = class_counter(os.path.join(args.data_basedir, args.splits_path, args.class_path))
        ptcloud_path_list = {'train':[], 'test':[], 'val':[]}

        for split_name in data_mode:
            for clname in class_list:
                f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt'%split_name),"r")
                for x in f:
                    instance_id = x[:-1]
                    ptcloud_path_list[split_name].append(os.path.join(args.data_basedir, args.ptcloud_path, clname, instance_id))

        pcd = o3d.io.read_point_cloud(os.path.join(ptcloud_path_list[mode][index], '%s.ply'%view))
        ptcloud_ply = np.asarray(pcd.points)
        print(mode, view, index, np.allclose(ptcloud_ply, ptcloud_npz))


def class_counter(indexfile):
    f_class = open(indexfile,"r")
    class_num = 0
    class_dic = {}
    class_list = []
    instance_num = 0
    for line in f_class:
        index = line.find(' ')
        clname = line[:index]
        class_dic[clname] = class_num
        class_list += [clname]
        class_num += 1
    return class_list, class_dic

compress(args, origin_name = 'ptcloud.ply', compressed_name='pts1024_upbound_test.npy')

#check(args)
'''
for view in range(0,5):
    #compress(args, origin_name='%d.ply'%view, compressed_name='ptcloud_%d.npz'%view) 
    check(args, view)
'''
# for view in ['ptcloud']:
#     #compress(args, origin_name='%d.ply'%view, compressed_name='ptcloud_%d.npz'%view) 
#     check(args, view)
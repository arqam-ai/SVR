'''
PointCloudDataset

author : Yefan
created : 8/9/19 10:21PM
'''
import sys
import open3d as o3d
from torch.utils.data import Dataset
import torch
import argparse
import os
import tqdm
import cv2
from torchvision import transforms
import numpy as np
import random
import matplotlib.pyplot as plt
sys.path.append('../')
from utils.utils import Normalization

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

class what3d_dataset_singleview(Dataset):

    def __init__(self, data_basedir, ptcloud_path, img_path, label_path, splits_path, split_name, class_path, image_size, view, read_view = False):

            
        self.image_size = image_size
        self.transforms = ToTensor()
        self.data_basedir = data_basedir
        self.cached_data_pt = np.load(os.path.join(data_basedir, ptcloud_path))[split_name]
        self.cached_data_label = np.load(os.path.join(data_basedir, label_path))[split_name]
        assert self.cached_data_pt.shape[0] == self.cached_data_label.shape[0]
        self.length = self.cached_data_label.shape[0]

        self.cached_data_img = np.zeros((self.length,  3,  self.image_size, self.image_size)) 
        self.cached_data_view = np.zeros((self.length, 2))
        
        convertor  = transforms.ToPILImage()
        self.class_list, _ = class_counter(os.path.join(data_basedir, splits_path, class_path))

        idx = 0
        for clname in tqdm.tqdm(self.class_list,total= len(self.class_list), desc = '%s img loading...'%split_name):
            f = open(os.path.join(data_basedir, splits_path, 'lists', clname, '%s.txt'%split_name),"r")
            for x in f:
                instance_id = x[:-1]
   
                image = cv2.imread(os.path.join(data_basedir, img_path, clname, instance_id, '%s.png'%view))   
                image = convertor(image)
                image = transforms.functional.resize(image, [self.image_size,self.image_size])
                image = np.array(image) / 255.0 
                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W
                image = image.transpose((2, 0, 1))        
                self.cached_data_img[idx] = image
                
                # Added one part of reading the view information
                if read_view:
                    fview = open(os.path.join(data_basedir, img_path, clname, instance_id, 'viewpoints.txt'), 'r')
                    for x in fview:
                        self.cached_data_view[idx,0] = float(x.split()[0][2:])
                        self.cached_data_view[idx,1] = float(x.split()[1][2:])
                        break
                    fview.close()
                    
                idx += 1
        assert idx == self.length, "Img loader go wrong"
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        image = self.cached_data_img[idx]
        ptcloud = self.cached_data_pt[idx]
        label = self.cached_data_label[idx]
        view = self.cached_data_view[idx]
        sample = {'image': image,'ptcloud':ptcloud,\
                      'label':label, 'view':view}
        sample = self.transforms(sample)

        return sample

class what3d_dataset(Dataset):
    def __init__(self, data_basedir, 
                       ptcloud_path, 
                       img_path, 
                       label_path, 
                       splits_path, 
                       split_name, 
                       class_path, 
                       views,
                       sample_ratio=1.0,
                       image_height=224,
                       image_width=224,
                       points_num=1024, 
                       read_view=False,
                       mode = 'viewer'):
        self.image_height = image_height
        self.image_width = image_width
        self.data_basedir = data_basedir
        self.class_list, _ = class_counter(os.path.join(self.data_basedir, splits_path, class_path))
        #ptcloud_path_list = list()
        image_path_list = list()
        ## Load the total instance path
        for clname in self.class_list:
            f = open(os.path.join(data_basedir, splits_path, 'lists', clname, '%s.txt'%split_name),"r")
            for x in f:
                instance_id = x[:-1]
                #ptcloud_path_list.append(os.path.join(data_basedir, ptcloud_path, clname, instance_id))
                image_path_list.append(os.path.join(data_basedir, img_path, clname, instance_id))

        self.data_ptcloud = np.load(os.path.join(data_basedir, ptcloud_path))[split_name]
        ## if sample ratio < 1.0, then sample the path list 
        total_instance_num = len(image_path_list)
        if abs(sample_ratio - 1.0) > 1e-6:
            self.instance_num = int(total_instance_num * sample_ratio)
            ## Sample the instance path 
            sample_index = random.sample(range(total_instance_num), self.instance_num)
            #ptcloud_path_list = [ptcloud_path_list[index] for index in sample_index]
            image_path_list = [image_path_list[index] for index in sample_index]
            self.data_ptcloud = self.data_ptcloud[sample_index]
        else:
            self.instance_num = total_instance_num

        ## Initialize the data structure 
        total_view_num = len(views) * self.instance_num
        self.data_img = np.zeros((total_view_num,  3,  self.image_height, self.image_width)) 
        #self.data_img = torch.zeros(total_view_num, 3, self.image_height, self.image_width)

        if read_view:
            self.data_view = np.zeros((total_view_num, 2))

        convertor = transforms.ToPILImage()
        pbar = tqdm.tqdm(total = total_view_num, desc = '%s data loading'%split_name)
        idx = 0
        for i in range(self.instance_num):
            for view in views:
                image = cv2.imread(os.path.join(image_path_list[i], '%s.png'%view)) 
                if image.shape[0] != self.image_height and image.shape[1] != self.width:
                    image = convertor(image)
                    image = transforms.functional.resize(image, [self.image_height, self.image_width])
                #image = (np.array(image)-np.min(image))/np.max(image)
                image = np.array(image)/255.0
                # swap image axis from (numpy image: H x W x C) to  (torch image: C X H X W)
                image = image.transpose((2, 0, 1))          
                self.data_img[idx] = image
                # read one view of point cloud
                # Added one part of reading the view information
                if read_view:
                    fview = open(os.path.join(data_basedir, img_path, clname, instance_id, 'viewpoints.txt'), 'r')
                    view_num = 0
                    for x in fview:
                        self.data_view[idx,0] = float(x.split()[0][2:])
                        self.data_view[idx,1] = float(x.split()[1][2:])
                        view_num += 1
                        if view_num >= len(views):
                            break
                    fview.close()
                idx += 1
                pbar.update(1)

        self.data_ptcloud = torch.from_numpy(self.data_ptcloud).float()
        self.data_img = torch.from_numpy(self.data_img).float()

    def __len__(self):
        return self.instance_num

    def __getitem__(self, idx):
        image = self.data_img[idx]
        ptcloud = self.data_ptcloud[idx]
        sample = {'image': image,'ptcloud':ptcloud}
        return sample

    @staticmethod
    def data_visualizer(ptcloud, image, split_name, path, idx):

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(ptcloud[0, :, 0], ptcloud[0, :, 1], ptcloud[0, :, 2], marker='o')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax = fig.add_subplot(122)
        image = image[0].transpose(0,1)
        image = image.transpose(1,2)
        ax.imshow(image)
        #index = np.argmax(label)
        #print(index)
        #class_list,_ = class_counter(os.path.join('../../../What3D', 'splits', 'classes.txt'))
        #class_name = class_list[index]
        #ax.set_title('%s'%class_name,fontsize=12,color='r')
        title = os.path.join(path, "%s_%d" % (split_name,idx))
        fig.savefig(title)
        plt.close()




class what3d_dataset_views(Dataset):
    def __init__(self, data_basedir, 
                       ptcloud_path, 
                       img_path, 
                       label_path, 
                       splits_path, 
                       split_name, 
                       class_path, 
                       views,
                       sample_ratio=1.0,
                       image_height=224,
                       image_width=224,
                       points_num=1024, 
                       read_view=False,
                       mode = 'viewer'):
        self.image_height = image_height
        self.image_width = image_width
        self.data_basedir = data_basedir
        self.mode = mode
        self.img_view_list = list()
        self.class_list, _ = class_counter(os.path.join(self.data_basedir, splits_path, class_path))
        image_path_list = list()

        ## Load the total instance path
        for clname in self.class_list:
            f = open(os.path.join(data_basedir, splits_path, 'lists', clname, '%s.txt'%split_name),"r")
            for x in f:
                instance_id = x[:-1]
                image_path_list.append(os.path.join(data_basedir, img_path, clname, instance_id))
        
        if self.mode == 'viewer':
            for idx, view in enumerate(views):
                ptcloud_path = ptcloud_path[:-5] + '%s.npz'%view
                if idx == 0:
                    self.data_ptcloud = np.load(os.path.join(data_basedir, ptcloud_path))[split_name]
                else:
                    self.data_ptcloud = np.concatenate((self.data_ptcloud, np.load(os.path.join(data_basedir, ptcloud_path))[split_name]), axis=0)
        elif self.mode == 'object':
            ptcloud_path = ptcloud_path[:-5] + '%s.npz'%"object"
            self.data_ptcloud = np.load(os.path.join(data_basedir, ptcloud_path))[split_name]
            self.ptcloud_num = self.data_ptcloud.shape[0]
        ## if sample ratio < 1.0, then sample the path list 
        total_instance_num = len(image_path_list)
        
        if abs(sample_ratio - 1.0) > 1e-6:
            sampled_instance_num = int(total_instance_num * sample_ratio)
            ## Sample the instance path 
            sample_index = random.sample(range(total_instance_num), sampled_instance_num)
            image_path_list = [image_path_list[index] for index in sample_index]
            self.data_ptcloud = self.data_ptcloud[what3d_dataset_views.ptcloud_index(sample_index, total_instance_num, len(views))]
        else:
            sampled_instance_num = total_instance_num
        self.instance_num = len(views) * sampled_instance_num

        self.convertor = transforms.ToPILImage()
        for view_idx, view in enumerate(views):
            for i in range(sampled_instance_num):
                self.img_view_list.append(os.path.join(image_path_list[i], '%s.png'%view))

        self.data_ptcloud = torch.from_numpy(self.data_ptcloud).float()
        self.data_ptcloud = Normalization(self.data_ptcloud, inplace=True, keep_track=False).normalize_unitL2ball()
        
        
    def __len__(self):
        return self.instance_num

    def __getitem__(self, idx):
        if self.mode == 'viewer':
            ptcloud = self.data_ptcloud[idx]
        elif self.mode == 'object':
            ptcloud = self.data_ptcloud[idx % self.ptcloud_num]

        image = cv2.imread(self.img_view_list[idx]) 
        if image.shape[0] != self.image_height and image.shape[1] != self.width:
            image = self.convertor(image)
            image = transforms.functional.resize(image, [self.image_height, self.image_width])
        image = np.array(image)/255.0
        # swap image axis from (numpy image: H x W x C) to  (torch image: C X H X W)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        sample = {'image': image,'ptcloud': ptcloud}
        return sample

    @staticmethod
    def ptcloud_index(sample_index, offset, view_num):
        res_index = sample_index.copy()
        for i in range(1, view_num):
            res_index+= [index + i * offset for index in sample_index]
            
        return res_index


def validtion(args):
    '''verify that if dataset contain multi view data'''
    kwargs = {'num_workers':1, 'pin_memory':True}
    train_loader = torch.utils.data.DataLoader(
                what3d_dataset_views(data_basedir=args.data_basedir, ptcloud_path=args.ptcloud_path, 
                img_path=args.img_path, label_path=args.label_path, 
                splits_path=args.splits_path, split_name='val', 
                class_path=args.class_path, sample_ratio=args.sample_ratio,
                image_height=args.image_size, image_width=args.image_size, 
                views=list(args.views), read_view=args.read_view,
                points_num = args.pts_num, mode = args.mode),
                batch_size=args.train_batch_size, shuffle=False,**kwargs)
    
    pbar = tqdm.tqdm(total = len(train_loader), desc = 'batch')
    instance_num = int(len(train_loader)/5) 
    index = random.sample(range(instance_num), 1)
    multiview_index = what3d_dataset_views.ptcloud_index(index, offset=instance_num, view_num=5)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx in multiview_index:
            what3d_dataset.data_visualizer(batch['ptcloud'], batch['image'], 'val', "../img/dataset/test", batch_idx)
        pbar.update(1)



def main(args):
    test_type = 'train'
    kwargs = {'num_workers':16, 'pin_memory':True}
    train_loader = torch.utils.data.DataLoader(
                what3d_dataset_views(data_basedir=args.data_basedir, ptcloud_path=args.ptcloud_path, 
                img_path=args.img_path, label_path=args.label_path, 
                splits_path=args.splits_path, split_name=test_type, 
                class_path=args.class_path, sample_ratio=args.sample_ratio,
                image_height=args.image_size, image_width=args.image_size, 
                views=list(args.views), read_view=args.read_view,
                points_num = args.pts_num, mode = args.mode),
                batch_size=args.train_batch_size, shuffle=False,**kwargs)
    
    pbar = tqdm.tqdm(total = len(train_loader), desc = 'batch')
    instance_num = int(len(train_loader)/5) 
    #print(instance_num)
    index = random.sample(range(instance_num), 2)
    multiview_index = what3d_dataset_views.ptcloud_index(index, offset=instance_num, view_num=5)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx in multiview_index:
            what3d_dataset.data_visualizer(batch['ptcloud'], batch['image'], test_type, "../img/dataset/test/object", batch_idx)
        pbar.update(1)
    
    
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(sys.argv[0])

    parser.add_argument("--ptcloud-path",dest="ptcloud_path", type=str,default="ptcloud_0.npz",
                         help='path of the ptcloud')

    parser.add_argument("--img-path", dest="img_path", type=str, default='renderings',
                      help='path of the image')

    parser.add_argument("--label-path", dest="label_path", type=str,default='label.npz',
                      help='path of the image')
    
    parser.add_argument("--data-basedir",dest="data_basedir", type=str, default='/home/zyf/What3D',#default='/home/../../public/zyf/What3D',
                      help='path of the data folder') 

    parser.add_argument("--splits-path",dest="splits_path", type=str, default='splits',
                      help='path of the data folder') 

    parser.add_argument("--class-path", dest="class_path", type=str,default='classes.txt',
                      help="class name list")

    parser.add_argument("--image-size",dest="image_size", type=int,default = 224,
                      help="image size for network")

    parser.add_argument("--sample-ratio",dest="sample_ratio", type=float, default = 1.0,
                      help="ratio to sample the dataset")

    parser.add_argument("--views",dest="views", type=str,default= '01234',
                    help="five view for each instance")

    parser.add_argument("--pts-num",dest="pts_num", type=int,default=1024,
                      help="number of points in a ptcloud")

    parser.add_argument("--mode", dest="mode", type=str,default="object", 
                      help="['viewer', 'object']")

    parser.add_argument("--read-view", action="store_true",dest="read_view",
                      default= False, help="if adding the view information in training", )

    parser.add_argument("--train-batch-size",dest="train_batch_size", type=int,
                       default=1, help='training batch size')
    args = parser.parse_args(sys.argv[1:])
    args.cuda = torch.cuda.is_available()
    print(str(args))

    main(args)



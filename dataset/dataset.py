'''
PointCloudDataset

author : Yefan
created : 8/9/19 10:21PM
'''
import sys
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
from utils.visualize_pts import draw_pts, colormap2d

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
        ## if sample ratio < 1.0, then sample the path list 
        total_instance_num = len(image_path_list)
        
        if abs(sample_ratio - 1.0) > 1e-6:
            sampled_instance_num = int(total_instance_num * sample_ratio)
            ## Sample the instance path 
            sample_index = random.sample(range(total_instance_num), sampled_instance_num)
            image_path_list = [image_path_list[index] for index in sample_index]
            if self.mode == 'viewer':
                self.data_ptcloud = self.data_ptcloud[what3d_dataset_views.ptcloud_index(sample_index, total_instance_num, len(views))]
            elif self.mode == 'object':
                self.data_ptcloud = self.data_ptcloud[sample_index]
        else:
            sampled_instance_num = total_instance_num

        self.ptcloud_num = self.data_ptcloud.shape[0]
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

    @staticmethod
    def data_visualizer(ptcloud, prediction, image, split_name, path, idx, loss=0.0, type='pt'):

        fig = plt.figure(figsize=(20, 4))

        ax = fig.add_subplot(151)
        if isinstance(image, np.ndarray):
            image = image[0].transpose((1, 2, 0))
        else: 
            image = image[0].transpose(0,1)
            image = image.transpose(1,2)

        ax.imshow(image)
        plt.axis("off")
        
        if type=='pt':
            if len(ptcloud.shape) == 2:
                ptcloud = ptcloud.unsqueeze(0)
            if len(prediction.shape) == 2:
                prediction = prediction.unsqueeze(0)
            ax = fig.add_subplot(152, projection='3d')
            ax.scatter(ptcloud[0,:,2], ptcloud[0,:,0], ptcloud[0,:,1], s= 2)
            ax.set_xlim([-0.5,0.5])
            ax.set_ylim([-0.5,0.5])
            ax.set_zlim([-0.5,0.5])
            ax.set_title("GT view (0,0)")
            plt.axis("off")
            ax.view_init(0, 0)

            ax = fig.add_subplot(153, projection='3d')
            ax.scatter(ptcloud[0,:,2], ptcloud[0,:,0], ptcloud[0,:,1], s= 2)
            ax.set_xlim([-0.5,0.5])
            ax.set_ylim([-0.5,0.5])
            ax.set_zlim([-0.5,0.5])
            ax.set_title("GT view (30,135)")
            plt.axis("off")
            ax.view_init(30, 135)

            ax = fig.add_subplot(154, projection='3d')
            ax.scatter(prediction[0,:,2], prediction[0,:,0], prediction[0,:,1], s= 2)
            ax.set_xlim([-0.5,0.5])
            ax.set_ylim([-0.5,0.5])
            ax.set_zlim([-0.5,0.5])
            ax.set_title("prediction view (0,0)")
            plt.axis("off")
            ax.view_init(0, 0)

            ax = fig.add_subplot(155, projection='3d')
            ax.scatter(prediction[0,:,2], prediction[0,:,0], prediction[0,:,1], s= 2)
            ax.set_xlim([-0.5,0.5])
            ax.set_ylim([-0.5,0.5])
            ax.set_zlim([-0.5,0.5])
            ax.set_title("prediction view (30,135)")
            plt.axis("off")
            ax.view_init(30, 135)

        elif type=='voxel':
            ax = fig.add_subplot(152, projection='3d')
            ax.voxels(ptcloud)
            ax.set_title("GT view (0,0)")
            ax.view_init(0, 0)

            ax = fig.add_subplot(153, projection='3d')
            ax.voxels(ptcloud)
            ax.set_title("GT view (30,135)")
            ax.view_init(30, 135)

            ax = fig.add_subplot(154, projection='3d')
            ax.voxels(prediction)
            ax.set_title("prediction view (0,0)")
            ax.view_init(0, 0)

            ax = fig.add_subplot(155, projection='3d')
            ax.voxels(prediction)
            ax.set_title("prediction view (30,135)")
            ax.view_init(30, 135)
        
        elif type=='voxel_cords':
            if len(ptcloud.shape) == 2:
                ptcloud = ptcloud.unsqueeze(0)
            if len(prediction.shape) == 2:
                prediction = prediction.unsqueeze(0)
            ax = fig.add_subplot(152, projection='3d')
            ax.scatter(ptcloud[0,:,2], ptcloud[0,:,0], ptcloud[0,:,1], s= 2)
            ax.set_xlim([0,128])
            ax.set_ylim([0,128])
            ax.set_zlim([0,128])
            ax.set_title("GT view (0,0)")
            ax.view_init(0, 0)

            ax = fig.add_subplot(153, projection='3d')
            ax.scatter(ptcloud[0,:,2], ptcloud[0,:,0], ptcloud[0,:,1], s= 2)
            ax.set_xlim([0,128])
            ax.set_ylim([0,128])
            ax.set_zlim([0,128])
            ax.set_title("GT view (30,135)")
            ax.view_init(30, 135)

            ax = fig.add_subplot(154, projection='3d')
            ax.scatter(prediction[0,:,2], prediction[0,:,0], prediction[0,:,1], s= 2)
            ax.set_xlim([0,128])
            ax.set_ylim([0,128])
            ax.set_zlim([0,128])
            ax.set_title("prediction view (0,0)")
            ax.view_init(0, 0)

            ax = fig.add_subplot(155, projection='3d')
            ax.scatter(prediction[0,:,2], prediction[0,:,0], prediction[0,:,1], s= 2)
            ax.set_xlim([0,128])
            ax.set_ylim([0,128])
            ax.set_zlim([0,128])
            ax.set_title("prediction view (30,135)")
            ax.view_init(30, 135)
            plt.title("mIoU is %.4f"%loss)

        
        title = os.path.join(path, "%s_%d" % (split_name,idx))
        fig.savefig(title, bbox_inches='tight')
        plt.close()



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
            what3d_dataset_views.data_visualizer(batch['ptcloud'], batch['ptcloud'], batch['image'], 'val', "../img/dataset/test", batch_idx)
        pbar.update(1)



def main(args):
    test_type = 'val'
    
    kwargs = {'num_workers':16, 'pin_memory':True}
    train_loader = torch.utils.data.DataLoader(
                what3d_dataset_views(data_basedir=args.data_basedir, ptcloud_path=args.ptcloud_path, 
                img_path=args.img_path, label_path=args.label_path, 
                splits_path=args.splits_path, split_name=test_type, 
                class_path=args.class_path, sample_ratio=args.sample_ratio,
                image_height=args.image_size, image_width=args.image_size, 
                views=list(args.views), points_num = args.pts_num, mode = args.mode),
                batch_size=args.train_batch_size, shuffle=False,**kwargs)
    
    
    # dataset = what3d_dataset_views(data_basedir=args.data_basedir, ptcloud_path=args.ptcloud_path, 
    #             img_path=args.img_path, label_path=args.label_path, 
    #             splits_path=args.splits_path, split_name=test_type, 
    #             class_path=args.class_path, sample_ratio=args.sample_ratio,
    #             image_height=args.image_size, image_width=args.image_size, 
    #             views=list(args.views), read_view=args.read_view,
    #             points_num = args.pts_num, mode = args.mode)

    
    pbar = tqdm.tqdm(total = len(train_loader), desc = 'batch')
    instance_num = int(len(train_loader)/5) 
    print(instance_num)
    index = random.sample(range(instance_num), 1)
    multiview_index = what3d_dataset_views.ptcloud_index(index, offset=instance_num, view_num=5)
    multiview_index = [0] + multiview_index
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx in multiview_index:
            what3d_dataset_views.data_visualizer(batch['ptcloud'].numpy(), batch['ptcloud'].numpy(), batch['image'], test_type, "../img/dataset/test/viewer", batch_idx)
        pbar.update(1)

    
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(sys.argv[0])

    parser.add_argument("--ptcloud-path",dest="ptcloud_path", type=str,default="ptcloud_n.npz",
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

    parser.add_argument("--image-size",dest="image_size", type=int, default = 224,
                      help="image size for network")

    parser.add_argument("--sample-ratio",dest="sample_ratio", type=float, default = 1.0,
                      help="ratio to sample the dataset")

    parser.add_argument("--views",dest="views", type=str,default= '01234',
                    help="five view for each instance")

    parser.add_argument("--pts-num",dest="pts_num", type=int,default=1024,
                      help="number of points in a ptcloud")

    parser.add_argument("--mode", dest="mode", type=str,default="viewer", 
                      help="['viewer', 'object']")

    parser.add_argument("--train-batch-size",dest="train_batch_size", type=int,
                       default=1, help='training batch size')
    args = parser.parse_args(sys.argv[1:])
    args.cuda = torch.cuda.is_available()
    print(str(args))

    main(args)



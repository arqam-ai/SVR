"""
    Auther: Yefan Zhou
    The script is used to test the consistency of what3d clustering evaluation and our training data path 
"""
import os
from dataset import class_counter
import tqdm
data_basedir = "../../What3D"
splits_path = "splits"
class_path = "classes.txt"
views = "01234"
cluster_path = "clustering"
import re

# class list 
class_list, _ = class_counter(os.path.join(data_basedir, splits_path, class_path))

## Train split
### SVR
split_name = "train"
image_path_list = []
for clname in class_list:
    f = open(os.path.join(data_basedir, splits_path, 'lists', clname, '%s.txt'%split_name),"r")
    for x in f:
        instance_id = x[:-1]
        image_path_list.append(os.path.join(clname, instance_id))

dataloader_img_view_list = []
for i in range(len(image_path_list)):
    for view_idx, view in enumerate(views):
        dataloader_img_view_list.append(os.path.join(image_path_list[i], '%s.png'%view))

### Cluster
cluster_img_view_list = []
f = open(os.path.join(data_basedir, cluster_path, "train_clusters.txt"),"r")
for x in f:
    cluster_img_view_list.append(x[:[i.start() for i in re.finditer(' ', x)][0]])

### Check two path list is consistent 
print("dataloader train instances {}, set {}".format(len(dataloader_img_view_list), len(set(dataloader_img_view_list))))
print("cluster train instances {}, set {}".format(len(cluster_img_view_list), len(set(cluster_img_view_list))))

num_miss = 0
for item in cluster_img_view_list:
    if item not in dataloader_img_view_list:
        print("item in cluster but not in dataloader {}".format(item))
        num_miss += 1
print("dataloader train miss match {}".format(num_miss))
num_miss = 0 

for item in dataloader_img_view_list:
    if item not in cluster_img_view_list:
        print("item in dataloader but not in cluster {}".format(item))
        num_miss += 1
print("cluster train miss match {}".format(num_miss))



## Test split
### SVR
split_name = "test"
image_path_list = []
for clname in class_list:
    f = open(os.path.join(data_basedir, splits_path, 'lists', clname, '%s.txt'%split_name),"r")
    for x in f:
        instance_id = x[:-1]
        image_path_list.append(os.path.join(clname, instance_id))

dataloader_img_view_list = []
for i in range(len(image_path_list)):
    for view_idx, view in enumerate(views):
        dataloader_img_view_list.append(os.path.join(image_path_list[i], '%s.png'%view))

### Cluster
cluster_img_view_list = []
f = open(os.path.join(data_basedir, cluster_path, "{}_images.txt".format(split_name)),"r")
for x in f:
    cluster_img_view_list.append(x[:[i.start() for i in re.finditer('/', x)][1] + 6])

### Check two path list is consistent 
print("dataloader test instances {}, set {}".format(len(dataloader_img_view_list), len(set(dataloader_img_view_list))))
print("cluster test instances {}, set {}".format(len(cluster_img_view_list), len(set(cluster_img_view_list))))

num_miss = 0
for item in cluster_img_view_list:
    if item not in dataloader_img_view_list:
        print("item in cluster but not in dataloader {}".format(item))
        num_miss += 1
print("dataloader test miss match {}".format(num_miss))
num_miss = 0 

for item in dataloader_img_view_list:
    if item not in cluster_img_view_list:
        print("item in dataloader but not in cluster {}".format(item))
        num_miss += 1
print("cluster test miss match {}".format(num_miss))



## Val split
### SVR
split_name = "val"
image_path_list = []
for clname in class_list:
    f = open(os.path.join(data_basedir, splits_path, 'lists', clname, '%s.txt'%split_name),"r")
    for x in f:
        instance_id = x[:-1]
        image_path_list.append(os.path.join(clname, instance_id))

dataloader_img_view_list = []
for i in range(len(image_path_list)):
    for view_idx, view in enumerate(views):
        dataloader_img_view_list.append(os.path.join(image_path_list[i], '%s.png'%view))

### Cluster
cluster_img_view_list = []
f = open(os.path.join(data_basedir, cluster_path, "{}_clusters.txt".format(split_name)),"r")
for x in f:
    cluster_img_view_list.append(x[:[i.start() for i in re.finditer(' ', x)][0]])

### Check two path list is consistent 
print("dataloader val instances {}, set {}".format(len(dataloader_img_view_list), len(set(dataloader_img_view_list))))
print("cluster val instances {}, set {}".format(len(cluster_img_view_list), len(set(cluster_img_view_list))))

num_miss = 0
for item in cluster_img_view_list:
    if item not in dataloader_img_view_list:
        print("item in cluster but not in dataloader {}".format(item))
        num_miss += 1
print("dataloader val miss match {}".format(num_miss))
num_miss = 0 

for item in dataloader_img_view_list:
    if item not in cluster_img_view_list:
        print("item in dataloader but not in cluster {}".format(item))
        num_miss += 1
print("cluster val miss match {}".format(num_miss))

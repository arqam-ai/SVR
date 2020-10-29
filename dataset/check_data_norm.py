import sys
import numpy as np
import torch
import tqdm
from dataset import what3d_dataset_views
sys.path.append("../")
from utils.utils import Normalization, normalize_bbox
from utils.loss import ChamferDistance

ptcloud_dic = np.load("../../What3D/ptcloud_0.npz")
ptcloud_set = ptcloud_dic["train"]
ptcloud_set = np.concatenate((ptcloud_set, ptcloud_dic["test"]), axis=0)
ptcloud_set = np.concatenate((ptcloud_set, ptcloud_dic["val"]), axis=0)
#ptcloud = ptcloud_set[7]
#ptcloud = np.expand_dims(ptcloud, 0)
chamfer = ChamferDistance().to("cuda")
'''
###Original BBOX Normalization 
min_vals = np.min(ptcloud, 1, keepdims=True)
max_vals = np.max(ptcloud, 1, keepdims=True)
print("original ptcloud", min_vals, max_vals, max_vals-min_vals)

###PyTorch BBOX Normalization 
ptcloud_torch = Normalization(torch.from_numpy(ptcloud.copy())).normalize_bounding_box()
min_vals = torch.min(ptcloud_torch, 1, keepdim=True)
max_vals = torch.max(ptcloud_torch, 1, keepdim=True)
print("pytorch3d", min_vals[0], max_vals[0], max_vals[0]-min_vals[0])


###NUMPY BBOX Normalization 
bbox = np.array([[[0, 0, 0], [1.0, 1.0, 1.0]]])
ptcloud_norm = normalize_bbox(ptcloud, bbox, isotropic=True)
min_vals = np.min(ptcloud_norm, 1, keepdims=True)
max_vals = np.max(ptcloud_norm, 1, keepdims=True)
print("numpy version", min_vals, max_vals, max_vals-min_vals)
chamfer = ChamferDistance()
loss = chamfer(torch.from_numpy(ptcloud_norm), torch.from_numpy(ptcloud))
what3d_dataset_views.data_visualizer(torch.from_numpy(ptcloud), torch.from_numpy(ptcloud_norm), torch.zeros(1, 3, 224, 224)
, split_name="test_bbox_norm", path="../img/dataset/bbox_norm", idx=0, loss=loss, type="pt_bbox")
'''

origin_max_list = []
origin_min_list = []


# bbox_max_list = []
# bbox_min_list = []
# loss_sum = 0
# bbox = np.array([[[0, 0, 0], [1.0, 1.0, 1.0]]])
# for i in tqdm.tqdm(range(ptcloud_set.shape[0])):

#     ptcloud = np.expand_dims(ptcloud_set[i], axis=0)
#     ptcloud = normalize_bbox(ptcloud, bbox)
#     min_vals = np.min(ptcloud, 1, keepdims=True)
#     max_vals = np.max(ptcloud, 1, keepdims=True)


    
#     ptcloud_unit2ball = Normalization(torch.from_numpy(ptcloud.copy())).normalize_unitL2ball()
#     ptcloud_unit2ball = ptcloud_unit2ball.numpy()
#     #min_vals = np.min(ptcloud_unit2ball, 1, keepdims=True)
#     #max_vals = np.max(ptcloud_unit2ball, 1, keepdims=True)

#     ptcloud_bbox = normalize_bbox(ptcloud_unit2ball, bbox)
#     min_vals_bbox = np.min(ptcloud_bbox, 1, keepdims=True)
#     max_vals_bbox = np.max(ptcloud_bbox, 1, keepdims=True)

#     #print("Origin bounding box         [%.4f %.4f  %.4f] [%.4f %.4f %.4f]" % (min_vals[0,0,0], min_vals[0,0,1], min_vals[0,0,2], max_vals[0,0,0], max_vals[0,0,1], max_vals[0,0,2]))
#     #print("Renormalized bounding box   [%.4f %.4f  %.4f] [%.4f %.4f %.4f]" % (min_vals_bbox[0,0,0], min_vals_bbox[0,0,1], min_vals_bbox[0,0,2], max_vals_bbox[0,0,0], max_vals_bbox[0,0,1], max_vals_bbox[0,0,2]))
    
    
#     loss_sum += chamfer(torch.from_numpy(ptcloud_bbox).to("cuda"), torch.from_numpy(ptcloud).to("cuda")).item()
#     #print(loss)
#     #print("Chamfer Distance %.4f" % loss)
#     #what3d_dataset_views.data_visualizer(torch.from_numpy(ptcloud), torch.from_numpy(ptcloud_bbox), torch.zeros(1, 3, 224, 224)
#     #, split_name="test_bbox_norm", path="../img/dataset/bbox_norm", idx = i , loss = loss, type = "pt_bbox")

# print(loss_sum/ptcloud_set.shape[0])



sample_set = ptcloud_set[0:100]
sample_set_unitL2ball = Normalization(torch.from_numpy(sample_set.copy())).normalize_unitL2ball()
bbox = np.array([[[0, 0, 0], [1.0, 1.0, 1.0]]])
ptcloud_bbox = normalize_bbox(sample_set_unitL2ball.numpy(), bbox)
for i in range(ptcloud_bbox.shape[0]):
    min_vals_bbox = np.min(ptcloud_bbox[[i]], 1, keepdims=True)
    max_vals_bbox = np.max(ptcloud_bbox[[i]], 1, keepdims=True)
    print("Renormalized bounding box   [%.4f %.4f  %.4f] [%.4f %.4f %.4f]" % (min_vals_bbox[0,0,0], min_vals_bbox[0,0,1], min_vals_bbox[0,0,2], max_vals_bbox[0,0,0], max_vals_bbox[0,0,1], max_vals_bbox[0,0,2]))

sample_set = normalize_bbox(sample_set, bbox)
loss = chamfer(torch.from_numpy(sample_set).to("cuda"), torch.from_numpy(ptcloud_bbox).to("cuda")).item()
print("%.4f" % loss)


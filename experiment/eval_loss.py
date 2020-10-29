"""
    This is script to eval Chamfer Loss
    Author: Yefan Zhou
"""
import os
import sys
import argparse
import logging
sys.path.append("../")
from utils.loss import ChamferDistance 
import numpy as np
import torch
import tqdm
from utils.utils import Normalization

chamfer = ChamferDistance().to("cuda")
gt = np.load("../../What3D/ptcloud_object.npz")['test']
oracle_nn_pred = np.load("Oracle_NN/Oracle_object/exp1/pred_ptcloud.npy")
foldingres_path = "object_models_v2/foldingres6_FCBN/final_vis"

B = gt.shape[0]

for i in tqdm.tqdm(range(0, 261), total=261):
    if i == 0:
        folding_ptcloud = np.load(os.path.join(foldingres_path, "fineptcloud_%04d.npy"%i))
    else:
        folding_ptcloud = np.concatenate((folding_ptcloud, np.load(os.path.join(foldingres_path, "fineptcloud_%04d.npy"%i))), axis=0)   

assert folding_ptcloud.shape[0] == B * 5


#folding_ptcloud = folding_ptcloud[0:B]
folding_ptcloud = torch.from_numpy(folding_ptcloud).to("cuda")
gt = torch.from_numpy(gt).to("cuda")
#oracle_nn_pred = torch.from_numpy(oracle_nn_pred).to("cuda")
gt = Normalization(gt, inplace=True, keep_track=False).normalize_unitL2ball()
#oracle_nn_pred = Normalization(oracle_nn_pred, inplace=True, keep_track=False).normalize_unitL2ball()
#folding_ptcloud = Normalization(folding_ptcloud, inplace=True, keep_track=False).normalize_unitL2ball()

print(gt.shape)
print(oracle_nn_pred.shape)
oracle_loss_stats = torch.zeros((B))
folding_loss_stats = torch.zeros((B*5))
sum_loss = 0

# for idx in tqdm.tqdm(range(gt.shape[0]), total=gt.shape[0]):
#     #oracle_loss_stats[idx] = chamfer(gt[idx].unsqueeze(0), oracle_nn_pred[idx].unsqueeze(0)).item()
#     folding_loss_stats[idx] = chamfer(gt[idx].unsqueeze(0), folding_ptcloud[idx].unsqueeze(0)).item()
#     #sum_loss += oracle_loss_stats[idx] 

for idx in tqdm.tqdm(range(B * 5), total=B * 5):
    #oracle_loss_stats[idx] = chamfer(gt[idx].unsqueeze(0), oracle_nn_pred[idx].unsqueeze(0)).item()
    folding_loss_stats[idx] = chamfer(gt[idx % B].unsqueeze(0), folding_ptcloud[idx].unsqueeze(0)).item()
    #sum_loss += oracle_loss_stats[idx] 

#np.save("object_models_v2/foldingres6_FCBN/loss_stats.npy", folding_loss_stats.numpy())
print(torch.mean(folding_loss_stats))
'''
np.save("Oracle_NN/Oracle_object/exp1/loss_stats.npy", loss_stats.numpy())
print(sum_loss/gt.shape[0])
print(torch.max(loss_stats))
print(torch.min(loss_stats))
print(torch.mean(loss_stats))
'''
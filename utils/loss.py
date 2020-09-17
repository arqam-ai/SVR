import sys
import numpy as np
import torch
import torch.nn as nn
import csv
#from utils.PyTorchEMD.emd import earth_mover_distance
# Define Chamfer Loss
#from pytorch3d.loss import chamfer_distance
#import utils.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
#from utils.ChamferDistancePytorch.chamfer_python import distChamfer
#from model.im2mesh.eval import distance_p2p
class ChamferDistance(nn.Module):

	def forward(self, input1, input2, mode='nonsquared'):
		# input1, input2: BxNxK, BxMxK, K = 3
		B, N, K = input1.shape
		_, M, _ = input2.shape

		# Repeat (x,y,z) M times in a row
		input11 = input1.unsqueeze(2)           # BxNx1xK
		input11 = input11.expand(B, N, M, K)    # BxNxMxK
		# Repeat (x,y,z) N times in a column
		input22 = input2.unsqueeze(1)           # Bx1xMxK
		input22 = input22.expand(B, N, M, K)    # BxNxMxK
		# compute the distance matrix
		D = input11 - input22                   # BxNxMxK
		D = torch.norm(D, p=2, dim=3)           # BxNxM
		if mode == 'squared':
			D = torch.norm(D, p=2, dim=3)**2    # for L2 
			
		dist0, _ = torch.min( D, dim=1 )        # BxM
		dist1, _ = torch.min( D, dim=2 )        # BxN

		loss = torch.mean(dist0, 1) + torch.mean(dist1, 1)  # B
		loss = torch.mean(loss)                             # 1
		return loss

		# pytorch3d 
		#loss_ptc_fine_pytorch3d, _ = chamfer_distance(ptcloud_pred_fine, ptcloud)
		# atlasnet python
		#dist1, dist2,_,_ = distChamfer(ptcloud_pred_fine, ptcloud)
		#loss_ptc_fine_atlaspython = dist1.mean() + dist2.mean()
		
# atlasnet c cuda
class ChamferDistanceL2(nn.Module):
	def __init__(self):
		super(ChamferDistanceL2, self).__init__()
		self.chamLoss = dist_chamfer_3D.chamfer_3DDist()

	def forward(self, prediction, gt):
		dist1, dist2, idx1, idx2 = self.chamLoss(prediction, gt)
		loss_ptc_fine_atlas = torch.mean(dist1) + torch.mean(dist2)
		return loss_ptc_fine_atlas

	
def emd(pred, gt):
	""" earth mover distance
	pred  : torch.tensor (N, ptnum, 3)
		predicted pointcloud 
	gt    : torch.tensor (N, ptnum, 3)
		ground truth pointcloud
	"""
	ptnum = pred.shape[1]
	emd = earth_mover_distance(gt, pred, transpose=False)
	emd /= ptnum

	return emd

   
class MaskedL1(nn.Module):
	def __init__(self):
		super(MaskedL1, self).__init__()
		self.criterion = nn.L1Loss(reduction="sum")

	def forward(self, gt, pred, mask):
		loss = self.criterion(gt*mask, pred*mask)
		loss /= (mask==1.0).sum()
		return loss

    
class ProjChamfer(nn.Module):
	def __init__(self,img_center):
		super(ProjChamfer, self).__init__()
		self.chamfer = ChamfersDistance3()
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
		self.centersize =  img_center
 
        
	def projection(self, input, proMatrix):
		B = input.shape[0]
		ptnum = input.shape[1]
		infill = torch.ones(B, ptnum, 1).to(self.device)          #BxNx1
		input_4by1 = torch.transpose(torch.cat((input,infill),2),1,2)               #BxNx4 -> Bx4xN 
		posed_ptcloud = torch.bmm(proMatrix, input_4by1)                            #Bx4xNi
		x_z = torch.div(posed_ptcloud[:,0,:],posed_ptcloud[:,2,:]).unsqueeze(2)
		y_z = torch.div(posed_ptcloud[:,1,:],posed_ptcloud[:,2,:]).unsqueeze(2)
		#constant_z =  torch.cuda.FloatTensor(torch.ones(B, ptnum, 1).to(self.device))   
		#squeezed_ptcloud = torch.cat((x_z, y_z, constant_z),2)
		squeezed_ptcloud = torch.cat((x_z, y_z),2)
		return squeezed_ptcloud  
    
		#squeezed_ptcloud = posed_ptcloud.clone()       
		#squeezed_ptcloud[:,0,:] = torch.div(posed_ptcloud[:,0,:],posed_ptcloud[:,2,:])    
		#squeezed_ptcloud[:,1,:] = torch.div(posed_ptcloud[:,1,:],posed_ptcloud[:,2,:])
		#squeezed_ptcloud[:,2,:] = torch.ones(B, ptnum)

    
    
	def normalize(self, input):
		B = input.shape[0]
		ptnum = input.shape[1]        
		center_xy = self.centersize * torch.ones(B, ptnum, 2).to(self.device)    
		#center_z =torch.zeros(B, ptnum, 1)
		#center = torch.cat((center_xy,center_z),2).to(self.device)
		center = center_xy       
		#print(input.shape)
		#print(center.shape)       
		return (input - center)/(2 * self.centersize)
        
        
	def forward(self, pred, gt, proMatrix):
		B = pred.shape[0]        
		ptnum = pred.shape[1]      
		gt_ptplane = self.projection(gt, proMatrix)
		pred_ptplane = self.projection(pred, proMatrix)
		#gt_ptplane = self.normalize(gt_ptplane)
		#pred_ptplane = self.normalize(pred_ptplane)

		loss = self.chamfer(gt_ptplane,pred_ptplane)     
		return loss
    
    
class ProjChamfer_test(nn.Module):
	def __init__(self,img_center):
		super(ProjChamfer_test, self).__init__()
		self.chamfer = ChamfersDistance3()
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")   
		self.centersize =  img_center
		self.x_max = list()
		self.x_min = list()
		self.y_max = list()
		self.y_min = list()
        
        
	def projection(self, input, proMatrix):
		B = input.shape[0]
		ptnum = input.shape[1]
		infill = torch.ones(B, ptnum, 1).to(self.device)          #BxNx1
		input_4by1 = torch.transpose(torch.cat((input,infill),2),1,2)               #BxNx4 -> Bx4xN 
		posed_ptcloud = torch.bmm(proMatrix, input_4by1)                            #Bx4xNi
		x_z = torch.div(posed_ptcloud[:,0,:],posed_ptcloud[:,2,:]).unsqueeze(2)
		y_z = torch.div(posed_ptcloud[:,1,:],posed_ptcloud[:,2,:]).unsqueeze(2)
		#constant_z =  torch.cuda.FloatTensor(torch.ones(B, ptnum, 1).to(self.device))   
		#squeezed_ptcloud = torch.cat((x_z, y_z, constant_z),2)
		squeezed_ptcloud = torch.cat((x_z, y_z),2)
		return squeezed_ptcloud  
    
		#squeezed_ptcloud = posed_ptcloud.clone()       
		#squeezed_ptcloud[:,0,:] = torch.div(posed_ptcloud[:,0,:],posed_ptcloud[:,2,:])    
		#squeezed_ptcloud[:,1,:] = torch.div(posed_ptcloud[:,1,:],posed_ptcloud[:,2,:])
		#squeezed_ptcloud[:,2,:] = torch.ones(B, ptnum)

    
    
	def normalize(self, input):
		B = input.shape[0]
		ptnum = input.shape[1]        
		center_xy = self.centersize * torch.ones(B, ptnum, 2).to(self.device)    
		#center_z =torch.zeros(B, ptnum, 1)
		#center = torch.cat((center_xy,center_z),2).to(self.device)
		center = center_xy       
		#print(input.shape)
		#print(center.shape)       
		return (input - center)/(2 * self.centersize)
        
	def text_save(self, filename, data):#
		file = open(filename,'a')
		for i in range(len(data)):            
			if i == 3:            
				s = str(data[i]) + '\n'
			else:
				s = str(data[i]) + '\t'    #    
			file.write(s)
		file.close()
		
    
    
	def forward(self, pred, gt, proMatrix):
		B = pred.shape[0]        
		ptnum = pred.shape[1]      
		gt_ptplane = self.projection(gt, proMatrix)
		tmp1 = torch.max(gt_ptplane[:,:,0])
		tmp2 = torch.min(gt_ptplane[:,:,0])
		tmp3 = torch.max(gt_ptplane[:,:,1])
		tmp4 = torch.min(gt_ptplane[:,:,1])     
		x_max = int(tmp1.item())
		x_min = int(tmp2.item())
		y_max = int(tmp3.item())
		y_min = int(tmp4.item()) 
		self.x_max.append(x_max)
		self.x_min.append(x_min)
		self.y_max.append(y_max)
		self.y_min.append(y_min)
		data = [x_min, x_max,y_min,y_max]
		print(min(self.x_min)," ",max(self.x_max)," ",min(self.y_min)," ",max(self.y_max))         
		self.text_save('datarange.txt',data)        
		pred_ptplane = self.projection(pred, proMatrix)
		gt_ptplane = self.normalize(gt_ptplane)
		pred_ptplane = self.normalize(pred_ptplane)

		loss = self.chamfer(gt_ptplane,pred_ptplane)     
		return loss
    

# def laplace_coord(input, lap_idx, block_id, use_cuda = True):
#
#     # Inputs :
#     # input : nodes Tensor, size (n_pts, n_features)
#     # lap_idx : laplace index matrix Tensor, size (n_pts, 10)
#     #
#     # Returns :
#     # The laplacian coordinates of input with respect to edges as in lap_idx
#
#
#     vertex = torch.cat((input, torch.zeros(1, 3).cuda()), 0) if use_cuda else torch.cat((input, torch.zeros(1, 3)), 0)
#
#     indices = torch.tensor(lap_idx[block_id][:, :8])
#     weights = torch.tensor(lap_idx[block_id][:,-1], dtype = torch.float32)
#
#     if use_cuda:
#         indices = indices.cuda()
#         weights = weights.cuda()
#
#     weights = torch.reciprocal(weights).reshape((-1, 1)).repeat((1, 3))
#
#     num_pts, num_indices = indices.shape[0], indices.shape[1]
#     indices = indices.reshape((-1,))
#     vertices = torch.index_select(vertex, 0, indices)
#     vertices = vertices.reshape((num_pts, num_indices, 3))
#
#     laplace = torch.sum(vertices, 1)
#     laplace = input - torch.mul(laplace, weights)
#
#     return laplace
#
# def laplace_loss(input1, input2, lap_idx, block_id, use_cuda = True):
#
#     # Inputs :
#     # input1, input2 : nodes Tensor before and after the deformation
#     # lap_idx : laplace index matrix Tensor, size (n_pts, 10)
#     # block_id : id of the deformation block (if different than 1 then adds
#     # a move loss as in the original TF code)
#     #
#     # Returns :
#     # The Laplacian loss, with the move loss as an additional term when relevant
#
#     lap1 = laplace_coord(input1, lap_idx, block_id, use_cuda)
#     lap2 = laplace_coord(input2, lap_idx, block_id, use_cuda)
#     laplace_loss = torch.mean(torch.sum(torch.pow(lap1 - lap2, 2), 1)) * 1500
#     move_loss = torch.mean(torch.sum(torch.pow(input1 - input2, 2), 1)) * 100
#
#     if block_id == 0:
#         return laplace_loss
#     else:
#         return laplace_loss + move_loss
#
#
#
# def edge_loss(pred, gt_pts, edges, block_id, use_cuda = True):
#
# 	# edge in graph
#     #nod1 = pred[edges[block_id][:, 0]]
#     #nod2 = pred[edges[block_id][:, 1]]
#     idx1 = torch.tensor(edges[block_id][:, 0]).long()
#     idx2 = torch.tensor(edges[block_id][:, 1]).long()
#
#     if use_cuda:
#         idx1 = idx1.cuda()
#         idx2 = idx2.cuda()
#
#     nod1 = torch.index_select(pred, 0, idx1)
#     nod2 = torch.index_select(pred, 0, idx2)
#     edge = nod1 - nod2
#
# 	# edge length loss
#     edge_length = torch.sum(torch.pow(edge, 2), 1)
#     edge_loss = torch.mean(edge_length) * 300
#
#     return edge_loss
#
#
# def L1Tensor(img1, img2) :
# 	""" input shoudl be tensor and between 0 and 1"""
# 	mae = torch.mean(torch.abs(img2 - img1))
# 	return mae
#
#
# def L2Tensor(img1, img2) :
# 	""" input shoudl be tensor and between 0 and 1"""
# 	mse = torch.mean((img2 - img1) ** 2)
# 	return mse
#
# def chamfer_distance(gt, pred):
#
# 	dist1, dist2 = distChamfer(gt, pred) # BxN
# 	my_chamfer_loss = torch.mean(dist1, 1) + torch.mean(dist2, 1) # B
# 	loss_cd = torch.mean(my_chamfer_loss) # 1
# 	return loss_cd
#
#
# def total_pts_loss(pred_pts_list, pred_feats_list, gt_pts, ellipsoid, use_cuda = True):
#     """
#     pred_pts_list: [x1, x1_2, x2, x2_2, x3]
#     """
#
#     my_chamfer_loss, my_edge_loss, my_lap_loss = 0., 0., 0.
#     lap_const = [0.2, 1., 1.]
#
#     for i in range(3):
#         dist1, dist2 = distChamfer(gt_pts, pred_pts_list[i].unsqueeze(0))
#         my_chamfer_loss += torch.mean(dist1) + torch.mean(dist2)
#         my_edge_loss += edge_loss(pred_pts_list[i], gt_pts, ellipsoid["edges"], i, use_cuda)
#         my_lap_loss += lap_const[i] * laplace_loss(pred_feats_list[i], pred_pts_list[i], ellipsoid["lap_idx"], i, use_cuda)
#
#     my_pts_loss = 100 * my_chamfer_loss + 0.1 * my_edge_loss + 0.3 * my_lap_loss
#
#     return my_pts_loss
#
#
#
# def total_img_loss(pred_img, gt_img):
#
#     my_rect_loss = torch.nn.functional.binary_cross_entropy(pred_img, gt_img, size_average = False)
#     my_l1_loss = L1Tensor(pred_img, gt_img)
#
#     img_loss = my_rect_loss + my_l1_loss
#
#     return img_loss




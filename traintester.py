import os
import sys
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.utils import check_exist_or_mkdirs
from torch.utils.tensorboard import SummaryWriter
from torch import autograd
import torchvision.utils as vutils
import tqdm
from pathlib import Path
import time
import utils.plot_image_grid as plot_image_grid
from utils.plot_log import plot_log as plot_log
from dataset.dataset import what3d_dataset_views
#from utils.f_score import threshold_list, f_score_list
from utils.loss import ChamferDistanceReNorm


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

class BN_Stats(object):
    def __init__(self, layer_name):
        self.stastics = {"mean":[], "var":[]}
        self.layer_name = layer_name
    def __call__(self, module, module_in, module_out):
        self.stastics["mean"].append(torch.mean(module_in,dim=[0,2]).detach().cpu().numpy().astype(np.float16))
        self.stastics["var"].append(torch.var(module_in,dim=[0,2],unbiased=False).detach().cpu().numpy().astype(np.float16))
    def clear(self):
        self.stastics = {"mean":[], "var":[]}
    def save(self, path):
        np.savez_compressed(path,
                mean = np.array(self.stastics["mean"]), 
                var = np.array(self.stastics["var"]))
        self.clear()


class TrainTester(object):

    def __init__(self, netG, criterion_G, optmizer_G, lr_scheduler_G, logger, args):
        self.netG= netG
        self.model = args.model
        self.criterion_G = criterion_G
        self.optimizer_G = optmizer_G
        self.logger = logger
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_schedulers = [self.lr_scheduler_G]

        self.total_epochs = args.total_epochs
        self.log_dir, self.verbose_per_n_batch = args.log_dir, args.verbose_per_n_batch
        self.test_per_n_epoch = args.test_per_n_epoch
        self.done = False
        self.train_iter = 0

        self.stats_finecd_itertrain = Stats()  
        self.stats_finecd_epochtrain = Stats()
        self.stats_finecd_epochtest = Stats()
        self.stats_finecd_epochval = Stats()
        self.stats_finecd_itertest = Stats()
        self.stats_lr_itertrain = Stats()
        self.finalchamferloss = Stats()

        self.fscore_renorm = args.fscore_renorm
        self.fscore = {"f": {}, "p": {}, "r": {}}
        # for key in self.fscore:
        #     for th in threshold_list:
        #         self.fscore[key].update({f"{th}":[]})
        # self.chamReNorm = ChamferDistanceReNorm(device=args.device)

        self.running_loss = None
        self.running_factor = 0.9
        self.epoch_callbacks = [self.save_stats]
        self.train_loss = float("inf")
        self.val_loss = float("inf")
        
        self.device = args.device
        self.lambda_loss_primitive = args.lambda_loss_primitive
        self.lambda_loss_fine = args.lambda_loss_fine
        self.save_results = args.save_results
        self.log_dir = args.log_dir
        self.normalize = args.normalize
        self.vis_dir = os.path.join(args.log_dir,'final_vis')
        self.tensorboard_dir = os.path.join(self.log_dir,'../../sum_runs')
        self.stats_dir = os.path.join(self.log_dir,'stats')
        self.plot_graph = False

        self.folding_twice = args.folding_twice
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vis_dir).mkdir(parents=True, exist_ok=True)
        Path(self.stats_dir).mkdir(parents=True, exist_ok=True)

        self.if_train = args.train
        self.tensorboard = args.tensorboard
        if self.tensorboard and self.if_train:
            if args.model == 'atlasnet':
                tensor_comment = "{}_{}_BS{}_noBN{}_LR{}_NumL{}_{}{}_Width{}_latent{}_NumP{}_TypeP_{}".format(args.model, args.mode, args.train_batch_size, 
                                                    args.remove_all_batchNorms, args.lr_G, 
                                                    args.num_layers, args.template_type, args.nb_primitives, 
                                                    args.hidden_neurons, args.bottleneck_size, args.nb_primitives, args.template_type)
            else:
                tensor_comment = "{}_{}_BS{}_Block_{}_noBN{}_LR{}_NumL{}_Width{}_latent{}".format(args.model, args.mode, args.train_batch_size, args.decoderBlock, 
                                                    args.remove_all_batchNorms, args.lr_G, args.num_layers, 
                                                    args.hidden_neurons, args.bottleneck_size) 
            self.writer = SummaryWriter(comment=tensor_comment)
        self.checkpoint_model = args.checkpoint_model
        self.checkpoint_solver = args.checkpoint_solver

        if self.checkpoint_model:
            self.logger.info("Loading model checkpoint ...")
            self.netG.load_state_dict(torch.load(self.checkpoint_model))

        if self.checkpoint_solver:
            self.logger.info("Loading solver checkpoint ...")
            self.optimizer_G.load_state_dict(torch.load(self.checkpoint_solver))

        # self.if_BNstats = args.if_BNstats
        # if self.if_BNstats:
        #     self.bn1_stats = BN_Stats("bn1")
        #     self.bn6_stats = BN_Stats("bn6")
        #     self.bn_stats_epoch = range(0, self.total_epochs, args.bnstats_step)
        #     self.hk_handles = []


    def invoke_epoch_callback(self):
        if len(self.epoch_callbacks)>0:
            for ith, cb in enumerate(self.epoch_callbacks):
                try:
                    cb()
                except:
                    self.logger.warn('epoch_callback[{}] failed.'.format(ith))

    def adjust_lr_linear(self, step, total_step):
        base_lr = self.solver.defaults['lr']
        lr = base_lr * (total_step - step + 1.) / total_step
        for param_group in self.solver.param_groups:
            param_group['lr'] = lr

    def hook_bn(self, hook_fn=None, mode="enable", target_block = 'G1', target_layer=["bn1"]):
        if mode == "enable":
            for name, layer in self.netG._modules.items():
                if name == target_block:
                    for name, layer in layer._modules.items():
                        if isinstance(layer, nn.BatchNorm1d) and name in target_layer:
                            self.hk_handles.append(layer.register_forward_hook(hook_fn))
        elif mode == "disable":
            for handle in self.hk_handles:
                handle.remove()
            self.hk_handles = []
    
    def save_stats(self):
        self.stats_finecd_epochval.save(os.path.join(self.stats_dir, 'stats_finecd_epochval.npz'))
        self.stats_finecd_itertrain.save(os.path.join(self.stats_dir, 'stats_finecd_itertrain.npz'))
        self.stats_finecd_epochtrain.save(os.path.join(self.stats_dir, 'stats_finecd_epochtrain.npz'))
        self.stats_finecd_epochtest.save(os.path.join(self.stats_dir, 'stats_finecd_epochtest.npz'))
        self.stats_lr_itertrain.save(os.path.join(self.stats_dir, 'stats_lr_itertrain.npz'))


    def train(self, epoch, loader):
        # import ipdb; ipdb.set_trace()
        loss_sum, loss_sum_ptc, loss_sum_fineCD, loss_sum_class, loss_sum_primitiveCD = 0.0, 0.0, 0.0, 0.0, 0.0
        batch_loss, batch_fineCD_loss, batch_primitiveCD_loss = 0.0, 0.0, 0.0
        starter_time = time.time()
        self.netG.train()
        for batch_idx, batch in enumerate(loader):
            for param_group in self.optimizer_G.param_groups:
                self.stats_lr_itertrain.push(self.train_iter, param_group['lr'])
                lr = param_group['lr']

            image, ptcloud = batch['image'], batch['ptcloud']

            image, ptcloud  = \
                Variable(image).to(self.device),\
                Variable(ptcloud).to(self.device),\

            ###################################################
            # (1) Update Image&PCG network:                   #
            ###################################################
            self.netG.zero_grad()
            if self.model == "foldingres":
                ptcloud_pred_primitive, ptcloud_pred_fine, _ , _ = self.netG(image)
                loss_ptc_fine = self.criterion_G(ptcloud_pred_fine, ptcloud)

                batch_fineCD_loss = self.lambda_loss_fine * loss_ptc_fine.item()
                loss_all = self.lambda_loss_fine * loss_ptc_fine

                if self.folding_twice:
                    loss_ptc_primitive = self.criterion_G(ptcloud_pred_primitive, ptcloud)
                    loss_all += self.lambda_loss_primitive * loss_ptc_primitive

            elif self.model == 'psgn':
                ptcloud_pred_fine, codeword = self.netG(image)
                loss_ptc_fine = self.criterion_G(ptcloud_pred_fine, ptcloud)
                batch_fineCD_loss = self.lambda_loss_fine * loss_ptc_fine.item()
                loss_all = self.lambda_loss_fine * loss_ptc_fine
            
            elif self.model == 'atlasnet':
                ptcloud_pred_fine, codeword = self.netG(image, train=True)
                ptcloud_pred_fine = ptcloud_pred_fine.transpose(2, 3).contiguous()
                ptcloud_pred_fine = ptcloud_pred_fine.view(ptcloud_pred_fine.size(0), -1, 3)
                #ptcloud_pred_fine = ptcloud_pred_fine.view(ptcloud_pred_fine.size(0), 1, -1, ptcloud_pred_fine.size(3)).squeeze(1)
                
                loss_ptc_fine = self.criterion_G(ptcloud_pred_fine, ptcloud)
                batch_fineCD_loss = self.lambda_loss_fine * loss_ptc_fine.item()
                loss_all = self.lambda_loss_fine * loss_ptc_fine

            if not self.plot_graph:
                self.writer.add_graph(self.netG, image)
                self.plot_graph = True

            loss_all.backward()
            self.optimizer_G.step()

            # stastically log
            batch_loss = loss_all.item()   
            loss_sum_fineCD += batch_fineCD_loss
            loss_sum += batch_loss

            if self.running_loss is None:
                self.running_loss = batch_loss
            else:
                self.running_loss = self.running_factor*self.running_loss \
                                    + (1-self.running_factor)*batch_loss

            # collect stats
            self.train_iter += 1
            self.stats_finecd_itertrain.push(self.train_iter, loss = batch_fineCD_loss)
            self.writer.add_scalar('Loss/train/iter', batch_fineCD_loss, self.train_iter)
            # logger
            if self.verbose_per_n_batch>0 and batch_idx % self.verbose_per_n_batch==0:
                self.logger.info((
                    'Epoch={:<3d} [{:3.0f}%/{:<5d}] '
                    'AVGloss (B,R)=({:.6f},{:.6f}) '
                    'FineCDloss = {:.6f} '
                    'lr = {:.5f} ').format(
                    epoch, 100.*batch_idx/loader.__len__(), len(loader.dataset),
                    batch_loss, self.running_loss,
                    batch_fineCD_loss, lr))

        self.stats_finecd_epochtrain.push(epoch, loss = loss_sum_fineCD/float(len(loader)))
        self.writer.add_scalar('Loss/train/epoch', loss_sum_fineCD/float(len(loader)), epoch)
        self.logger.info("======== Epoch {:<3d} ========".format(epoch))
        self.logger.info("Train: avg overall={:.4f}, avg fineCD = {:.4f}, time = {:.3f}".format(
            loss_sum / float(len(loader)),
            loss_sum_fineCD/float(len(loader)),
            time.time()- starter_time))
        return loss_sum_fineCD / float(loader.__len__())


    def test(self, epoch, loader, type):
        # import ipdb; ipdb.set_trace()
        # self.netI.eval()
        self.netG.eval()
        chamfer_loss = 0.
        bbox_chamfer_loss = 0.
        test_loss = 0.
        counter = 0.              
        batch_idx = 0

        for batch in tqdm.tqdm(loader, total=len(loader)):
            image, ptcloud = batch['image'], batch['ptcloud']
            image, ptcloud  = \
                Variable(image).to(self.device), \
                Variable(ptcloud).to(self.device), \

            with torch.set_grad_enabled(False):
                if self.model == "foldingres":
                    _, ptcloud_pred_fine, codeword, _ = self.netG(image)
                elif self.model == 'psgn':
                    ptcloud_pred_fine, codeword = self.netG(image)
                elif self.model == 'atlasnet':
                    ptcloud_pred_fine, codeword = self.netG(image, train=False)
                    ptcloud_pred_fine = ptcloud_pred_fine.transpose(2, 3).contiguous()
                    ptcloud_pred_fine = ptcloud_pred_fine.view(ptcloud_pred_fine.size(0), -1, 3)

                loss_ptc_fine = self.criterion_G(ptcloud_pred_fine, ptcloud)
            
            ########### Extra operation in last test epoch ###########
            if epoch == self.total_epochs + 1:
                pc_orig, pc2 = \
                ptcloud.cpu().numpy(), ptcloud_pred_fine.cpu().numpy()
                code = codeword.cpu().numpy()
                code = np.squeeze(code)
                #self.finalchamferloss.push(batch_idx, loss = loss_ptc_fine.item())
                img = image.cpu()
                what3d_dataset_views.data_visualizer(pc_orig, pc2, img, type, self.vis_dir, batch_idx, type=f'pt_{self.normalize}')
                if self.save_results:
                    np.save('%s/oriptcloud_%04d.npy' % (self.vis_dir, batch_idx), pc_orig)
                    #np.save('%s/codeword_%04d.npy'%(self.vis_dir, batch_idx), code)
                    #np.save('%s/fineptcloud_%04d.npy'%(self.vis_dir, batch_idx), pc2)
                
                ##################  F-SCORE    ###############
                # for th in threshold_list:
                #    f_p_r = f_score_list(th=th, pred_set=pc2, gt_set=pc_orig, normalize=self.fscore_renorm)
                #    self.fscore["f"][f"{th}"] += f_p_r[0]
                #    self.fscore["p"][f"{th}"] += f_p_r[1]
                #    self.fscore["r"][f"{th}"] += f_p_r[2]
                
                # ################ BBox Chamfer ##############
                # bbox_chamfer_loss += self.chamReNorm(pc_orig, pc2, renorm="bbox").item()

            batch_idx += 1
            chamfer_loss += loss_ptc_fine.item()     
            test_loss += self.lambda_loss_fine * loss_ptc_fine.item()
            counter += 1

        test_loss = test_loss / float(counter)
        chamfer_loss = chamfer_loss / float(counter)
        
    
        if type == 'test' and self.if_train:
            self.stats_finecd_itertest.push(self.train_iter, loss=chamfer_loss)
            self.stats_finecd_epochtest.push(epoch, loss = chamfer_loss)
            self.writer.add_scalar('Loss/test', chamfer_loss, epoch)
        elif type == 'val':
            self.stats_finecd_epochval.push(epoch, loss = chamfer_loss)
            self.writer.add_scalar('Loss/val', chamfer_loss, epoch)
            self.invoke_epoch_callback()
            plot_log(self.stats_dir, ["stats_finecd_epochval.npz", "stats_finecd_itertrain.npz",
                "stats_finecd_epochtrain.npz"])
        self.logger.info('{} set (epoch={:<3d}): AverageLoss={:.6f}: ChamferLoss={:.6f}: '.format(type, epoch, test_loss, chamfer_loss))
        # if epoch == self.total_epochs + 1:
        #     bbox_chamfer_loss = bbox_chamfer_loss / float(counter)
        #     self.logger.info('{} set (epoch={:<3d}): BBox ChamferLoss={:.6f}: '.format(type, epoch, bbox_chamfer_loss))
        #     np.save(os.path.join(self.stats_dir, "f_score_{}.npy".format(self.fscore_renorm)), self.fscore)
        #     for key in self.fscore:
        #         self.logger.info(f"{key}: ")
        #         score_list = ""
        #         th_list = ""
        #         for th in self.fscore[key]:
        #             score_list += "%.5f  " % (sum(self.fscore[key][th])/len(self.fscore[key][th]))
        #             th_list += "%s   "% th
        #         self.logger.info(th_list)
        #         self.logger.info(score_list)
        #         self.logger.info("----------------------------")

        return chamfer_loss

    def generate_mesh(self, epoch, loader, type):
        self.netG.eval()
        chamfer_loss = 0.
        test_loss = 0.
        counter = 0.              
        batch_idx = 0
        for batch in tqdm.tqdm(loader,total=len(loader)):
            image, ptcloud = batch['image'], batch['ptcloud']
            image, ptcloud  = \
                Variable(image).to(self.device), \
                Variable(ptcloud).to(self.device), \
            
            with torch.set_grad_enabled(False):
                if self.model == 'atlasnet':
                    pred_mesh = self.netG.generate_mesh(image)
                    np.save(os.path.join(self.vis_dir, "mesh_vertices_%d.npy" % (batch_idx)), pred_mesh.vertices)
                    np.save(os.path.join(self.vis_dir, "mesh_faces_%d.npy" % (batch_idx)), pred_mesh.faces)

            batch_idx += 1


    def run(self, train_loader, test_loader, val_loader):
        
        # add a hook_fn as a member, then when epoch condition enable hook otherwise remove, 
        # consider save it as use numpy key, 'epoch': "mean" 2x2350x512 "var"2x2350x4   
        # torch.mean(batch)  torch.mean/max/var/min(channel)

        for epoch in range(1, self.total_epochs+1):
#########################    Train    ############################
            new_train_loss = self.train(
                epoch=epoch,
                loader=train_loader,
            )
######################### Validation #############################            
            new_val_loss = self.test(
                epoch=epoch,
                loader=val_loader,
                type = 'val'
            )
################## Save Model best in validation ######################## 
            if new_val_loss < self.val_loss:
                self.logger.info('Epoch %d saving checkpoint .... val epoch loss %f' % (epoch, new_val_loss))
                torch.save(self.netG.state_dict(), os.path.join(self.log_dir, "model_train_best.pth"))
                torch.save(self.optimizer_G.state_dict(), os.path.join(self.log_dir,"solver_train_best.pth"))
                self.val_loss = new_val_loss

############################ Test ########################
            if epoch % self.test_per_n_epoch == 0 or epoch == 1:
                self.test(
                    epoch=epoch,
                    loader=test_loader,
                    type = 'test'
                )
                self.invoke_epoch_callback()
                plot_log(self.stats_dir, ["stats_finecd_epochtest.npz"])
            self.lr_scheduler_G.step()

        self.writer.close()
        self.save_stats()
        self.done = True


# #######################  BN statistics ##########################
#             if epoch in self.bn_stats_epoch and self.if_BNstats:
#                 self.hook_bn(self.bn1_stats, mode='enable',
#                                 target_block='G1', target_layer=["bn1"])
#                 self.hook_bn(self.bn6_stats, mode='enable',
#                                 target_block='G1', target_layer=["bn6"])


# ######################## BN statistics END #######################                
#             if epoch in self.bn_stats_epoch and self.if_BNstats:
#                 self.hook_bn(mode='disable')
#                 self.bn1_stats.save(os.path.join(self.stats_dir, "epoch%d_layer_%s_input.npz"%(epoch,"bn1")))
#                 self.bn6_stats.save(os.path.join(self.stats_dir, "epoch%d_layer_%s_input.npz"%(epoch,"bn6")))
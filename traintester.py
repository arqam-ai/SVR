import os
import sys
import glog as logger
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
from dataset.dataset import what3d_dataset

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

class TrainTester(object):

    def __init__(self, netG, criterion_G,  criterion_C, criterion_M, optmizer_G, lr_scheduler_G, alpha, logger, args):
        self.netG= netG
        self.model = args.model
        self.criterion_G = criterion_G
        self.criterion_C = criterion_C
        self.criterion_M = criterion_M
        self.optimizer_G = optmizer_G
        self.logger = logger
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_schedulers = [self.lr_scheduler_G]

        self.total_epochs = args.total_epochs
        self.log_dir, self.verbose_per_n_batch = args.log_dir, args.verbose_per_n_batch
        self.test_per_n_epoch = args.test_per_n_epoch
        self.done = False
        self.train_iter = 0
        #self.stats_train_batch = Stats()
        #self.stats_train_running = Stats()
#		self.stats_class_itertrain = Stats()
#		self.stats_class_epochval = Stats()
#		self.stats_class_epochtest = Stats()
#		self.stats_class_epochtrain = Stats()
#		self.stats_classacc_epochtest = Stats()
#		self.stats_classacc_epochval = Stats()

        self.stats_finecd_itertrain = Stats()  
        self.stats_finecd_epochtrain = Stats()
        self.stats_finecd_epochtest = Stats()
        self.stats_finecd_epochval = Stats()
        self.stats_finecd_itertest = Stats()

#		self.stats_manifold_itertrain = Stats()
#		self.stats_manifold_epochtest = Stats()
#		self.stats_manifold_epochtrain = Stats()
#		self.stats_manifold_epochval = Stats()

        self.stats_lr_itertrain = Stats()
        self.finalchamferloss = Stats()

        self.use_manifold = args.use_manifold
        self.running_loss = None
        self.running_factor = 0.9
        self.epoch_callbacks = [self.save_stats]
        self.train_loss = float("inf")
        self.val_loss = float("inf")
        #self.batch_size = args.train_batch_size
        self.device = args.device
        self.mean = args.mean
        self.stddev = args.stddev
        self.lambda_loss_classification = args.lambda_loss_classification
        self.lambda_loss_primitive = args.lambda_loss_primitive
        self.lambda_loss_fine = args.lambda_loss_fine
        self.lambda_loss_manifold = args.lambda_loss_manifold
        self.save_results = args.save_results
        self.snapshot_dir = args.snapshot_dir
        self.log_dir = args.log_dir
        self.output_dir = args.output_dir
        self.vis_dir = os.path.join(args.output_dir,'final_vis')
        self.tensorboard_dir = os.path.join(self.log_dir,'scalar')
#        self.aug_dir = os.path.join(args.output_dir,'aug_test')
#        self.alpha = alpha
        self.minloss = {}
        self.read_view = args.read_view
        self.folding_twice = args.folding_twice
        Path(self.snapshot_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vis_dir).mkdir(parents=True, exist_ok=True)
#        Path(self.aug_dir).mkdir(parents=True, exist_ok=True)
        
        self.image_saver = plot_image_grid.NumpytoPNG(self.vis_dir)	
        self.tensorboard = args.tensorboard
        if self.tensorboard:
            self.writer = SummaryWriter(self.tensorboard_dir)
        self.checkpoint_model = args.checkpoint_model
        self.checkpoint_solver = args.checkpoint_solver
        self.testing = args.test
#		self.minloss['testloss'] = (0, float("+inf"))
#		self.minloss['trainloss'] = (0, float("+inf"))

        if self.checkpoint_model:
            print("Loading model checkpoint ...")
            self.netG.load_state_dict(torch.load(self.checkpoint_model))

        if self.checkpoint_solver:
            print("Loading solver checkpoint ...")
            self.optimizer_G.load_state_dict(torch.load(self.checkpoint_solver))

    def invoke_epoch_callback(self):
        if len(self.epoch_callbacks)>0:
            for ith, cb in enumerate(self.epoch_callbacks):
                try:
                    cb()
                except:
                    logger.warn('epoch_callback[{}] failed.'.format(ith))

    def adjust_lr_linear(self, step, total_step):
        base_lr = self.solver.defaults['lr']
        lr = base_lr * (total_step - step + 1.) / total_step
        for param_group in self.solver.param_groups:
            param_group['lr'] = lr


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
            if self.model == 'foldingnet':
                ptcloud_pred_primitive, ptcloud_pred_fine, _ , _ = self.netG(image)
                loss_ptc_fine = self.criterion_G(ptcloud_pred_fine, ptcloud)

                batch_fineCD_loss = self.lambda_loss_fine * loss_ptc_fine.item()
                loss_all = self.lambda_loss_fine * loss_ptc_fine

                if self.folding_twice:
                    loss_ptc_primitive = self.criterion_G(ptcloud_pred_primitive, ptcloud)
#                    batch_primitiveCD_loss = self.lambda_loss_primitive * loss_ptc_primitive.item()
                    loss_all += self.lambda_loss_primitive * loss_ptc_primitive

            elif self.model == 'psgn':
                ptcloud_pred_fine, codeword = self.netG(image)
                loss_ptc_fine = self.criterion_G(ptcloud_pred_fine, ptcloud)
                batch_fineCD_loss = self.lambda_loss_fine * loss_ptc_fine.item()
                loss_all = self.lambda_loss_fine * loss_ptc_fine
            
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
            self.writer.add_scalar('Loss/train', batch_fineCD_loss, self.train_iter)
            # logger
            if self.verbose_per_n_batch>0 and batch_idx % self.verbose_per_n_batch==0:
                self.logger.info((
                    'Epoch={:<3d} [{:3.0f}%/{:<5d}] '
                    'AVGloss (B,R)=({:.3f},{:.3f}) '
                    'FineCDloss = {:.4f} '
                    'lr = {:.5f} ').format(
                    epoch, 100.*batch_idx/loader.__len__(), len(loader.dataset),
                    batch_loss, self.running_loss,
                    batch_fineCD_loss, lr))

        self.stats_finecd_epochtrain.push(epoch, loss = loss_sum_fineCD/float(len(loader)))

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
        test_loss = 0.
        counter = 0.              
        batch_idx = 0

        for batch in tqdm.tqdm(loader,total=len(loader)):
            image, ptcloud = batch['image'], batch['ptcloud']
            image, ptcloud  = \
                Variable(image).to(self.device), \
                Variable(ptcloud).to(self.device), \

            with torch.set_grad_enabled(False):
                if self.model == 'foldingnet':
                    _, ptcloud_pred_fine, codeword, _ = self.netG(image)
                elif self.model == 'psgn':
                    ptcloud_pred_fine, codeword = self.netG(image)

                loss_ptc_fine = self.criterion_G(ptcloud_pred_fine, ptcloud)
            
            if epoch == self.total_epochs + 1 and self.save_results:
                pc_orig, pc2 = \
                ptcloud.cpu().numpy(), ptcloud_pred_fine.cpu().numpy()
                
                code = codeword.cpu().numpy()
                code = np.squeeze(code)
                self.finalchamferloss.push(batch_idx, loss = loss_ptc_fine.item())
                np.save('%s/oriptcloud_%04d.npy' % (self.vis_dir, batch_idx),pc_orig)
                np.save('%s/fineptcloud_%04d.npy'%(self.vis_dir, batch_idx),pc2)
                np.save('%s/codeword_%04d.npy'%(self.vis_dir, batch_idx),code)

                img = image.cpu()
                what3d_dataset.data_visualizer(pc_orig, img, type, self.vis_dir, batch_idx)
#                img = img.numpy()
#                self.image_saver.save(img)

            batch_idx += 1
            chamfer_loss += loss_ptc_fine.item()     
            test_loss += self.lambda_loss_fine * loss_ptc_fine.item()
            counter += 1

        test_loss = test_loss / float(counter)
        chamfer_loss = chamfer_loss / float(counter)

        if type == 'test':
            self.stats_finecd_itertest.push(self.train_iter, loss=chamfer_loss)
            self.stats_finecd_epochtest.push(epoch, loss = chamfer_loss)
            self.writer.add_scalar('Loss/test', chamfer_loss, epoch)

        elif type == 'val':
            self.stats_finecd_epochval.push(epoch, loss = chamfer_loss)
            self.writer.add_scalar('Loss/val', chamfer_loss, epoch)
        self.logger.info('{} set (epoch={:<3d}): AverageLoss={:.4f}: ChamferLoss={:.4f}: '.format(type, epoch, test_loss, chamfer_loss))

        return chamfer_loss


    def save_stats(self):

        self.stats_finecd_epochval.save(os.path.join(self.log_dir, 'stats_finecd_epochval.npz'))
        self.stats_finecd_itertrain.save(os.path.join(self.log_dir, 'stats_finecd_itertrain.npz'))
        self.stats_finecd_epochtrain.save(os.path.join(self.log_dir, 'stats_finecd_epochtrain.npz'))
        self.stats_finecd_epochtest.save(os.path.join(self.log_dir, 'stats_finecd_epochtest.npz'))
        self.stats_lr_itertrain.save(os.path.join(self.log_dir, 'stats_lr_itertrain.npz'))


    def run(self, train_loader, test_loader, val_loader):

        self.netG.to(self.device)
        self.logger.info('Network Architecture:')
        print(str(self.netG))
        sys.stdout.flush()

        for epoch in range(1, self.total_epochs+1):
            if self.use_manifold:
                new_train_loss = self.manifold_train(
                    epoch=epoch,
                    loader=train_loader,
                )
        
                new_val_loss = self.manifold_test(
                    epoch=epoch,
                    loader=val_loader,
                    type = 'val'
                )
            else:
                new_train_loss = self.train(
                    epoch=epoch,
                    loader=train_loader,
                )
            
                new_val_loss = self.test(
                    epoch=epoch,
                    loader=val_loader,
                    type = 'val'
                )

            if new_val_loss < self.val_loss:
                #print(self.train_loss, ',', new_train_loss)
                self.logger.info('Epoch %d saving checkpoint .... Training epoch loss %f' % (epoch, new_val_loss))
                #self.minloss['trainloss'] = (epoch, new_train_loss)
                # torch.save(self.netI.state_dict(), os.path.join(self.snapshot_dir, "model_image_train_best.pth"))
                torch.save(self.netG.state_dict(), os.path.join(self.snapshot_dir, "model_train_best.pth"))
                torch.save(self.optimizer_G.state_dict(), os.path.join(self.snapshot_dir,"solver_train_best.pth"))
                self.val_loss = new_val_loss
            if epoch % self.test_per_n_epoch == 0:
                if self.use_manifold:
                    self.manifold_test(
                        epoch=epoch,
                        loader=test_loader,
                        type = 'test'
                    )
                else:
                    self.test(
                        epoch=epoch,
                        loader=test_loader,
                        type = 'test'
                    )
                self.invoke_epoch_callback()
                
            self.lr_scheduler_G.step()
            
        self.save_stats()

        self.done = True






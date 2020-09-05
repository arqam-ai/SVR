import os
import sys
import argparse
import errno
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, proj3d
import torch
from torch.nn import init
import trimesh

def trimesh_remove_texture(scene_or_mesh):
    """convert trimesh mesh or trimesh scene to trimesh mesh only with vertices and faces 
    
    Params:
        mesh_or_scene(Trimesh mesh or scene): mesh or scene 
    Return:
        mesh(Trimesh mesh)
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = trimesh.Trimesh(vertices=scene_or_mesh.vertices, faces=scene_or_mesh.faces)
    return mesh

def make_D_label(input, value, device, random=False):
	if random:
		if value == 0:
			lower, upper = 0, 0.205
		elif value ==1:
			lower, upper = 0.8, 1.05
		D_label = torch.FloatTensor(input.data.size()).uniform_(lower, upper).to(device)
	else:
		D_label = torch.FloatTensor(input.data.size()).fill_(value).to(device)

	return D_label


def init_weights(net, init_type="kaiming", init_gain=0.02):
	"""Initialize network weights.
	Parameters:
		net (network)   -- network to be initialized
		init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
		init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
	We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
	work better for some applications. Feel free to try yourself.
	"""
	def init_func(m):  # define the initialization function
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)  # apply the initialization function <init_funce

def check_exist_or_mkdirs(path):
    '''thread-safe mkdirs if not exist'''
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def vis_pts(pts, clr, cmap):
    fig = plt.figure()
    fig.set_rasterized(True)
    ax = axes3d.Axes3D(fig)

    ax.set_alpha(0)
    ax.set_aspect('equal')
    min_lim = pts.min()
    max_lim = pts.max()
    ax.set_xlim3d(min_lim,max_lim)
    ax.set_ylim3d(min_lim,max_lim)
    ax.set_zlim3d(min_lim,max_lim)

    if clr is None:
        M = ax.get_proj()
        _,_,clr = proj3d.proj_transform(pts[:,0], pts[:,1], pts[:,2], M)
        clr = (clr-clr.min())/(clr.max()-clr.min())

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    ax.scatter(
        pts[:,0],pts[:,1],pts[:,2],
        c=clr,
        zdir='x',
        s=20,
        cmap=cmap,
        edgecolors='k'
    )
    return fig


def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_data_nolabel(x, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]

    return mixed_x, x, x[index,:],lam

    
def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_codeword(y_a, y_b, lam):
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_y



def normalize(self, ptcloud, ptnum, D):
        assert (ptcloud.shape[0]==1024)
        points_mean = torch.mean(ptcloud,dim = 0, keepdim=True)
        points_shifted = ptcloud - points_mean
        max_norm = torch.max(torch.norm(points_shifted,dim = 1))
        points_normalized = points_shifted/max_norm
        return points_normalized


def maskinit_generator(self, mask, B, ptnum, D, z_dims):
    xy_dims = int(ptnum / z_dims)
    flip_angle = torch.cuda.FloatTensor([math.pi])
    rot_matrix = torch.cuda.FloatTensor([[torch.cos(flip_angle), -torch.sin(flip_angle),0], 
              [torch.sin(flip_angle), torch.cos(flip_angle),0],
               [0,0,1]])
    for i in range(B):
        m = mask[i].transpose(1,2)                  # B x C x H x W -> C x W x H 
        m = m[0,:,:].squeeze()              
        xy = torch.nonzero(m)       
        xy_sample = xy[torch.randperm(xy.shape[0])[:xy_dims],:]
        while xy_sample.shape[0] < xy_dims:
            res = xy_dims - xy_sample.shape[0]
            xy_sample = torch.cat((xy_sample, xy[torch.randperm(xy.shape[0])[:res],:]), 0)
        xy_sample = xy_sample.repeat(z_dims, 1).float()
        xy_norm = self.normalize(xy_sample, ptnum, 2)
        z = torch.cuda.FloatTensor(ptnum, 1).uniform_(-1, 1)
        #z = torch.zeros(ptnum, 1).to(self.device)
        xyz = torch.cat((xy_norm,z),1)
        xyz = torch.matmul(xyz,rot_matrix).unsqueeze(0)
        #print(xyz.shape)
        print("x_max", torch.max(xyz[:,:,0]),"x_min", torch.min(xyz[:,:,0]))
        print("y_max", torch.max(xyz[:,:,1]),"y_min", torch.min(xyz[:,:,1]))
        print("z_max", torch.max(xyz[:,:,2]),"z_min", torch.min(xyz[:,:,2]))
        if i == 0:
            maskinit_ptcloud = xyz
        else:
            maskinit_ptcloud = torch.cat((maskinit_ptcloud,xyz),0)

    return maskinit_ptcloud



def gaussian_generator(self, B, N, D):
    noise = torch.FloatTensor(B, N, D)
    for i in range(B):
        if D == 3:
        # set gaussian ceters and covariances in 3D
            means = np.array(
                    [[0.0, 0.0, 0.0]]
                    )
            covs = np.array([np.diag([0.01, 0.01, 0.03])
                 #np.diag([0.08, 0.01, 0.01]),
                 #np.diag([0.01, 0.05, 0.01]),
                 #np.diag([0.03, 0.07, 0.01])
                 ])
            n_gaussians = means.shape[0]
        points = []
        for i in range(len(means)):
            x = np.random.multivariate_normal(means[i], covs[i], N )
            points.append(x)
        points = np.concatenate(points)
        #fit the gaussian model
        gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
        gmm.fit(points)
        noise[i] = torch.tensor(points,dtype= torch.float)

    return noise

def cube_generator(self, B, N, D):
    cube = torch.FloatTensor(B,N,D)
    x_count = 8
    y_count = 8
    z_count = 16
    count = x_count * y_count * z_count
    for i in range(B):
        one = np.zeros((count, 3))
        # setting vertices
        for t in range(count):
            x = float(t % x_count) / (x_count - 1)
            y = float((t / x_count) % y_count) / (y_count - 1)
            z = float(t / (x_count * y_count) % z_count) / (z_count - 1)
            one[t] = [x - 0.5, y - 0.5, z -0.5]
        one *= 0.5/one.max()
        cube[i]= torch.tensor(one,dtype= torch.float)
    return cube


def count_parameter_num(params):
    cnt = 0
    for p in params:
        cnt += np.prod(p.size())
    return cnt


class TrainTestMonitor(object):

    def __init__(self, log_dir, plot_loss_max=4., plot_extra=False):
        assert(os.path.exists(log_dir))

        stats_test = np.load(os.path.join(log_dir, 'stats_test.npz'))
        stats_train_running = np.load(os.path.join(log_dir, 'stats_train_running.npz'))

        self.title = os.path.basename(log_dir)
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()
        plt.title(self.title)

        # Training loss
        iter_loss = stats_train_running['iter_loss']
        self.ax1.plot(iter_loss[:,0], iter_loss[:,1],'-',label='train loss',color='r',linewidth=2)
        self.ax1.set_ylim([0, plot_loss_max])
        self.ax1.set_xlabel('iteration')
        self.ax1.set_ylabel('loss')

        # Test accuracy
        iter_acc = stats_test['iter_acc']
        max_accu_pos = np.argmax(iter_acc[:,1])
        test_label = 'max test accuracy {:.3f} @ {}'.format(iter_acc[max_accu_pos,1],max_accu_pos+1)
        self.ax2.plot(iter_acc[:,0], iter_acc[:,1],'o--',label=test_label,color='b',linewidth=2)
        self.ax2.set_ylabel('accuracy')

        if plot_extra:
            # Training accuracy
            iter_acc = stats_train_running['iter_acc']
            self.ax2.plot(iter_acc[:,0], iter_acc[:,1],'--',label='train accuracy',color='b',linewidth=.8)
            # Test loss
            iter_loss = stats_test['iter_loss']
            self.ax1.plot(iter_loss[:,0], iter_loss[:,1],'--',label='test loss',color='r',linewidth=.8)

        self.ax1.legend(loc='upper left', framealpha=0.8)
        self.ax2.legend(loc='lower right', framealpha=0.8)
        self.fig.show()


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)

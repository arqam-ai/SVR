from plyfile import PlyData, PlyElement
import numpy as np
import optparse
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tqdm
import random
import cv2

def downsampling(ply):
	sampled_ply = np.random.choice(ply,1024)
	return sampled_ply

def draw_pts(pts, clr, cmap, ax=None,sz=20):
    if ax is None:
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)

        ax.view_init(-45,-64)
    else:
        ax.cla()
    pts -= np.mean(pts,axis=0) #demean

    ax.set_alpha(255)
    ax.set_aspect('equal')
    min_lim = pts.min()
    max_lim = pts.max()
    ax.set_xlim3d(min_lim,max_lim)
    ax.set_ylim3d(min_lim,max_lim)
    ax.set_zlim3d(min_lim,max_lim)

    if cmap is None and clr is not None:
        #print(clr.shape)
        #print(pts.shape)
        assert(np.all(clr.shape==pts.shape))
        sct=ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=clr,
            zdir='y',
            s=sz,
            edgecolors=(0.5, 0.5, 0.5)
        )

    else:
        if clr is None:
            M = ax.get_proj()
            _,clr,_ = proj3d.proj_transform(pts[:,0], pts[:,1], pts[:,2], M)
        clr = (clr-clr.min())/(clr.max()-clr.min()) #normalization
        sct=ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=clr,
            zdir='y',
            s=sz,
            cmap=cmap,
            # depthshade=False,
            edgecolors=(0.5, 0.5, 0.5)
        )

    #ax.set_axis_off()
    ax.set_facecolor("white")
    return ax,sct

def check_exist_or_mkdirs(path):
    '''thread-safe mkdirs if not exist'''
    
    if not os.path.exists(path):
        os.makedirs(path)

		
def create_ptcloud_folder():

	f_class = open("splits/classes.txt","r")
	check_exist_or_mkdirs(args.datapath)
	class_num = 0
	instance_num = 0
	ply_num = 0 
	for cl in f_class:
		index = cl.find(' ')
		class_num += 1
		clname = cl[:index]
		check_exist_or_mkdirs(os.path.join(args.datapath, clname))
		for split in ['train','val','test']:
			f = open("splits/lists/%s/%s.txt"%(clname,split),"r")
			for x in f:
				instance_id = x[:-1]
				check_exist_or_mkdirs(os.path.join(args.datapath, clname,instance_id))
				instance_num += 1
				for view in ['0','1','2','3','4']:
					plydata = PlyData.read('points/%s/%s/%s.ply'%(clname, instance_id, view))
					plydata['vertex'].data = downsampling(plydata['vertex'].data)
					filename = '%s/%s/%s/%s.ply'%(args.datapath, clname, instance_id, view)
					plydata.write(filename)
					ply_num += 1
					print(ply_num)
					print(filename)
	print(class_num)
	print(instance_num)
	print(ply_num)

def ply_to_numpy():
	pass

def class_counter(indexfile):
    f_class = open(indexfile,"r")
    class_num = 0
    class_dic = {}
    class_list = []
    instance_num = 0
    class_counter = {}
    for line in f_class:
        index = line.find(' ')
        clname = line[:index]
        class_dic[clname] = class_num
        class_list += [clname]
        class_num += 1
        class_counter[clname] = 0
    sorted_class = sorted(class_dic.items(),key=lambda item:item[0],reverse = False)

    return class_list, class_dic,class_counter

def split_counter(class_dic):
	print(len(class_dic))
	split_dic = {'train':0, 'val':0, 'test':0}
	for split in ['train', 'val' ,'test']:
		for clname in tqdm.tqdm(class_dic, total= len(class_dic), desc = split):
			f = open("splits/lists/%s/%s.txt"%(clname,split),"r")
			for x in f:
				split_dic[split] += 1

	return split_dic

'''
def image_compress():
	class_dic = class_counter("splits/classes.txt")
	print(class_dic)
	split_image = np.zeros((1,224,224,3))
	instance_num = 0
	for split in ['train', 'val' ,'test']:
		for clname in class_dic:
			f = open("splits/lists/%s/%s.txt"%(clname,split),"r")
				for x in f:
					instance_id = x[:-1]
					for view in ['0']:
						instance_num += 1
						img = cv2.imread(os.path.join('rendering',clname, instance_id, '%s.png'%view))
						split_image = np.concatenate((split_image, img), axis=0)
						print(split_image.shape,instance_num)	
		split_image = split_image[1:]
		print(split, split_image.shape)
		split_image = np.zeros((1,224,224,3))
'''
def ptcloud_compress():
	class_list, class_dic,_= class_counter("splits/classes.txt")
	print(class_list)
	#split_ptcloud = np.zeros((1,1024,3))
	
	split_dic = split_counter(class_dic)

	train_ptcloud = np.zeros((split_dic['train'], 1024, 3))
	print(train_ptcloud.shape)
	test_ptcloud = np.zeros((split_dic['test'], 1024, 3))
	print(test_ptcloud.shape)
	val_ptcloud = np.zeros((split_dic['val'], 1024, 3))
	print(val_ptcloud.shape)
	instance_num = 0
	for view in ['1','2','3','4']:
		for split in ['train']:
			for clname in tqdm.tqdm(class_list, total= len(class_list), desc = split):
				f = open("splits/lists/%s/%s.txt"%(clname,split),"r")
				for x in f:
					instance_id = x[:-1]
					plydata = PlyData.read('%s/%s/%s/%s.ply'%(args.ptcloudpath, clname, instance_id, view))
					ptcloud = np.concatenate((plydata['vertex'].data['x'][:,np.newaxis], plydata['vertex'].data['y'][:,np.newaxis],
															plydata['vertex'].data['z'][:,np.newaxis]), axis = 1)#[np.newaxis,:]
					#split_ptcloud = np.concatenate((split_ptcloud, ptcloud), axis=0)	
					if split == 'train':
						train_ptcloud[instance_num] = ptcloud
					instance_num += 1
		print(view, split, instance_num)
		instance_num = 0

		for split in ['test']:
			for clname in tqdm.tqdm(class_list, total= len(class_list), desc = split):
				f = open("splits/lists/%s/%s.txt"%(clname,split),"r")
				for x in f:
					instance_id = x[:-1]
					plydata = PlyData.read('%s/%s/%s/%s.ply'%(args.ptcloudpath, clname, instance_id, view))
					ptcloud = np.concatenate((plydata['vertex'].data['x'][:,np.newaxis], plydata['vertex'].data['y'][:,np.newaxis],
															plydata['vertex'].data['z'][:,np.newaxis]), axis = 1)#[np.newaxis,:]
					#split_ptcloud = np.concatenate((split_ptcloud, ptcloud), axis=0)	
					if split == 'test':
						test_ptcloud[instance_num] = ptcloud
					instance_num += 1
		print(view, split,instance_num)
		instance_num = 0

		for split in ['val']:
			for clname in tqdm.tqdm(class_list, total= len(class_list), desc = split):
				f = open("splits/lists/%s/%s.txt"%(clname,split),"r")
				for x in f:
					instance_id = x[:-1]
					plydata = PlyData.read('%s/%s/%s/%s.ply'%(args.ptcloudpath, clname, instance_id, view))
					ptcloud = np.concatenate((plydata['vertex'].data['x'][:,np.newaxis], plydata['vertex'].data['y'][:,np.newaxis],
															plydata['vertex'].data['z'][:,np.newaxis]), axis = 1)#[np.newaxis,:]
					#split_ptcloud = np.concatenate((split_ptcloud, ptcloud), axis=0)	
					if split == 'val':
						val_ptcloud[instance_num] = ptcloud
					instance_num += 1
		print(view, split, instance_num)
		instance_num = 0
		np.savez('ptcloud_%s.npz'%(view), train = train_ptcloud, val = val_ptcloud, test = test_ptcloud)
		train_ptcloud = np.zeros((split_dic['train'], 1024, 3))
		print(train_ptcloud.shape)
		test_ptcloud = np.zeros((split_dic['test'], 1024, 3))
		print(test_ptcloud.shape)
		val_ptcloud = np.zeros((split_dic['val'], 1024, 3))
		print(val_ptcloud.shape)

		
		

def label_compress():
	class_list, class_dic, class_count = class_counter("splits/classes.txt")
	
	class_num = len(class_list)
	label_matrix = np.eye(class_num)
#	check_exist_or_mkdirs(args.datapath)
	split_dic = split_counter(class_dic)
	train_label = np.zeros((split_dic['train'], 1),dtype=np.int32)
	test_label = np.zeros((split_dic['test'], 1),dtype=np.int32)
	val_label = np.zeros((split_dic['val'], 1),dtype=np.int32)
	print(class_list)
	instance_num = 0
	counter = {'train':0, 'test': 0, 'val':0}
	for split in ['train']:
		for clname in class_list:
			#check_exist_or_mkdirs(os.path.join(args.datapath, clname))
			f = open("splits/lists/%s/%s.txt"%(clname,split),"r")
			for x in f:
				class_count[clname] += 1
				train_label[instance_num] = class_list.index(clname)
				instance_num += 1
		instance_num = 0

	for split in ['test']:
		for clname in class_list:
			#check_exist_or_mkdirs(os.path.join(args.datapath, clname))
			f = open("splits/lists/%s/%s.txt"%(clname,split),"r")
			for x in f:
				class_count[clname] += 1
				test_label[instance_num] = class_list.index(clname)
				instance_num += 1
		instance_num = 0

	for split in ['val']:
		for clname in class_list:
			#check_exist_or_mkdirs(os.path.join(args.datapath, clname))
			f = open("splits/lists/%s/%s.txt"%(clname,split),"r")
			for x in f:
				class_count[clname] += 1
				val_label[instance_num] = class_list.index(clname)
				instance_num += 1
		instance_num = 0
		
	print(len(train_label), len(test_label), len(val_label))

	np.savez('debug_label.npz', train = train_label, val = val_label, test = test_label)



def ptcloud_show():
	ptcloud_0 = np.load('ptcloud_0.npz')
	trainset = ptcloud_0['val']
	for i in range(trainset.shape[0]):
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(trainset[i][:,0],trainset[i][:,1],trainset[i][:,2])
		plt.show()

def image_compress(imgpath, splitpath):
	class_dic = class_counter(os.path.join("%s/classes.txt"%splitpath))
	split_dic = split_counter(class_dic)
	train_img = np.zeros((split_dic['train'], 224, 224, 3))
	print(train_img.shape)
	test_img = np.zeros((split_dic['test'], 224, 224, 3))
	print(test_img.shape)
	val_img = np.zeros((split_dic['val'], 224, 224, 3))
	print(val_img.shape)
	instance_num = 0
	for split in ['val']: #, 'train' ,'test'
		for clname in tqdm.tqdm(class_dic,total= len(class_dic), desc = split):
			f = open("%s/lists/%s/%s.txt"%(splitpath, clname,split),"r")
			for x in f:
				instance_id = x[:-1]
				for view in ['0']:
					img = cv2.imread(os.path.join(imgpath, clname, instance_id, '%s.png'%view))
					
					if split == 'train':
						train_img[instance_num] = img
					elif split == 'test':
						test_img[instance_num] = img
					elif split == 'val':
						val_img[instance_num] = img
					instance_num += 1 
		print(split, instance_num)
		instance_num = 0    

	np.save('img_0.npy', img)

def dic_add(dica, dicb):
	dic = {}
	for key in dica:
	    if dicb.get(key):
	        dic[key]=dica[key]+dicb[key]
	    else:
	        dic[key]=dica[key]
	for key in dicb:
	    if dica.get(key):
	        pass
	    else:
	        dic[key]=dicb[key]
	return dic

def dataset_statistic():
	trainset = trainset[0]
	valset = valset[0]
	testset = testset[0]
	set1 = 	dic_add(testset, valset)
	set = dic_add(set1, trainset)
	print(sum(set.values()))

	fig = plt.figure(figsize=(40, 12))
	plt.barh(set.keys(),set.values(),0.4,color="green")
	plt.xlabel("instance num per class")
	plt.ylabel("class name")
	plt.title("What 3D Total instance num %d"%sum(set.values()))
	plt.show()

def main(args):
	ptcloud_compress()
	
	



if __name__ == '__main__':
	parser = optparse.OptionParser(sys.argv[0], description="Training Encoder_decoder")
	parser.add_option("--ptcloudpath",
					  dest="ptcloudpath", type=str,
					  default="points_1024",
#					  default="label",
					  help='path of the training json file')
	parser.add_option("--labelpath",
					  dest="labelpath", type=str,
					  default="label",
#					  default="label",
					  help='path of the training json file')
	parser.add_option("--imgpath",
					  dest="imgpath", type=str,
					  default="renderings",
					  help='path of the training json file')
	parser.add_option("--splitpath",
					  dest="splitpath", type=str,
					  default="splits",
					  help='path of the training json file')
	(args, opts) = parser.parse_args()
	print(str(args))
	sys.stdout.flush()
	main(args)
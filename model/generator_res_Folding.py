'''
Generator(pointnet) for point cloud

author: Yefan
created: 8/8/19 11:21 PM
'''
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import numpy as np
from model.pointnet import PointwiseMLP
from utils.utils import init_weights
import argparse
import sys
#from utils.visualize_pts import draw_pts,colormap2d
from matplotlib import pyplot as plt
#from encoders import Encoder
from model.resnet_pytorch import resnet18

class GeneratorSingle(nn.Module):
    def __init__(self, dims):
        
        super(GeneratorSingle, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, X):
        return self.mlp.forward(X)

class GeneratorRes6(nn.Module):
    def __init__(self, width, codelength, input_dim = 2):
        super(GeneratorRes6, self).__init__()
        self.input_layer = PointwiseMLP([input_dim+codelength, width], doLastRelu=True)
        self.layer1 = PointwiseMLP([width, width], doLastRelu=False)
        self.layer2 = PointwiseMLP([width, width], doLastRelu=False)
        self.layer3 = PointwiseMLP([width, width], doLastRelu=False)
        self.layer4 = PointwiseMLP([width, width], doLastRelu=False)
        self.layer5 = PointwiseMLP([width, width], doLastRelu=False)
        self.layer6 = PointwiseMLP([width, width], doLastRelu=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        self.bn3 = nn.BatchNorm1d(width)
        self.bn4 = nn.BatchNorm1d(width)
        self.bn5 = nn.BatchNorm1d(width)
        self.bn6 = nn.BatchNorm1d(width)
        self.layer7 = PointwiseMLP([width, 64], doLastRelu=True)
        self.layer8 = PointwiseMLP([64, 3], doLastRelu=False)

    def forward(self, X):
        X = self.input_layer(X)
        Y = self.layer1(X)
        Y = self.bn1(Y.permute(0, 2, 1)).permute(0, 2, 1)
        X = self.layer2(Y + X)
        X = self.bn2(X.permute(0, 2, 1)).permute(0, 2, 1)
        Y = self.layer3(X)
        Y = self.bn3(Y.permute(0, 2, 1)).permute(0, 2, 1)
        X = self.layer4(Y + X)
        X = self.bn4(X.permute(0, 2, 1)).permute(0, 2, 1)
        Y = self.layer5(X)
        Y = self.bn5(Y.permute(0, 2, 1)).permute(0, 2, 1)
        X = self.layer6(Y + X)
        X = self.bn6(X.permute(0, 2, 1)).permute(0, 2, 1)
        X = self.layer7(X)
        X = self.layer8(X)

        return X


class GeneratorRes18(nn.Module):
    def __init__(self, width, codelength, input_dim = 2):

        super(GeneratorRes18, self).__init__()
        self.input_layer = PointwiseMLP([input_dim+codelength, width], doLastRelu=True)
        self.layer1 = PointwiseMLP([width, width], doLastRelu=True)
        self.layer2 = PointwiseMLP([width, width], doLastRelu=True)
        self.layer3 = PointwiseMLP([width, width], doLastRelu=True)
        self.layer4 = PointwiseMLP([width, width], doLastRelu=True)
        self.layer5 = PointwiseMLP([width, width], doLastRelu=True)
        self.layer6 = PointwiseMLP([width, width], doLastRelu=True)
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        self.bn3 = nn.BatchNorm1d(width)
        self.bn4 = nn.BatchNorm1d(width)
        self.bn5 = nn.BatchNorm1d(width)
        self.bn6 = nn.BatchNorm1d(width)
        self.layer7 = PointwiseMLP([width, 64], doLastRelu=True)        
        self.layer8 = PointwiseMLP([64, 3], doLastRelu=False)
        
    def forward(self, X):
        X = self.input_layer(X)
        Y = self.layer1(X)
        Y = self.bn1(Y.permute(0, 2, 1)).permute(0, 2, 1)
        X = self.layer2(Y + X)
        X = self.bn2(X.permute(0, 2, 1)).permute(0, 2, 1)
        
        Y = self.layer3(X)
        Y = self.bn3(Y.permute(0, 2, 1)).permute(0, 2, 1)
        X = self.layer4(Y + X)
        X = self.bn4(X.permute(0, 2, 1)).permute(0, 2, 1)
        
        Y = self.layer5(X)
        Y = self.bn5(Y.permute(0, 2, 1)).permute(0, 2, 1)
        X = self.layer6(Y + X)
        X = self.bn6(X.permute(0, 2, 1)).permute(0, 2, 1)
        X = self.layer7(X)
        X = self.layer8(X)
        
        return X


    
class GeneratorVanilla(nn.Module):

    def __init__(self, grid_dims, resgen_width, resgen_depth, 
                 resgen_codelength, class_num, block,
                 read_view = False, folding_twice = False):
        
        super(GeneratorVanilla,self).__init__()
        N = grid_dims[0]*grid_dims[1]
        u = (torch.arange(0., grid_dims[0]) / grid_dims[0] - 0.5).repeat(grid_dims[1])
        v = (torch.arange(0., grid_dims[1]) / grid_dims[1] - 0.5).expand(grid_dims[0], -1).t().reshape(-1)
        t = torch.empty(grid_dims[0]*grid_dims[1], dtype = torch.float)
        t.fill_(0.)
       
        #self.read_view_branch = nn.Linear(2, resgen_codelength)
        #self.read_view = read_view
        self.folding_twice = folding_twice
        
        self.encoder = resnet18(pretrained=False)
        init_weights(self.encoder, init_type="kaiming")
        self.grid = torch.stack((u, v), 1)
        self.N = grid_dims[0] * grid_dims[1]
        if block == "foldingres6":
            GeneratorRes = GeneratorRes6
        elif block == "foldingres18":
            GeneratorRes = GeneratorRes18
        self.G1 = GeneratorRes(resgen_width, resgen_codelength, input_dim = 2)
        init_weights(self.G1, init_type="xavier")
        if self.folding_twice:
            self.G2 = GeneratorRes(resgen_width, resgen_codelength, input_dim = 3)
            init_weights(self.G2, init_type="xavier")
        self.classifier = nn.Linear(512, class_num)
          
    def forward(self, X, view=None):

        img_feat = self.encoder(X)                      # B* 512
        img_feat = img_feat.unsqueeze(1)                # B* 1 * 512                 
        codeword = img_feat.expand(-1, self.N, -1)      # B* self.N *512

        class_prediction = self.classifier(img_feat)
        B = codeword.shape[0]       					# extract batch size
        tmpGrid = self.grid
        tmpGrid = tmpGrid.unsqueeze(0)
        tmpGrid = tmpGrid.expand(B, -1, -1)     		# BxNx2
        tmpGrid = tmpGrid.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        #1st generation
        f = torch.cat((tmpGrid, codeword), 2)
        f1 = self.G1.forward(f)
        #2nd generation
        if self.folding_twice:
            f = torch.cat((f1, codeword), 2)
            f = self.G2.forward(f)
            return f1, f, img_feat , class_prediction

        return f1, f1, img_feat , class_prediction        

def main(args):
    kwargs = {'num_workers':4, 'pin_memory':True} if args.cuda else {}
    generator = GeneratorVanilla(
        FC_dims =(2048, 1024, 512),
        grid_dims=(32,32,1),
        Generate1_dims=(514,512,512,3),
        Generate2_dims=(515,512,512,3))
    generator = generator.cuda()
    y = generator(torch.randn(2,3,128,128).cuda())

    '''
    for x in dataloader:
        generator(x["image"])
        break 
    '''

if __name__ == '__main__':


    parser = argparse.ArgumentParser(sys.argv[0])
    '''
    parser.add_argument('-0','--jsonfile-pkl',type=str,
                        help='path of the jsonfile')
    parser.add_argument('-d','--parent-dir',type =str,
                        help ='path of data file')
    parser.add_argument('--image-size', type = int, default =128,
                        help ='image size ')
    parser.add_argument('--shuffle-point-order',type=str, default= 'no',
                         help ='whether/how to shuffle point order (no/offline/online)')
    parser.add_argument('--batch-size',type=int, default=16,
                        help='training batch size')
    parser.add_argument('--test-batch-size',type=int,default=32,
                        help='testing batch size')
    '''
    args = parser.parse_args(sys.argv[1:])
    args.cuda = torch.cuda.is_available()
    print(str(args))
    main(args)
    
    
    
    
  






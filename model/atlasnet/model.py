from model.atlasnet.atlasnet import Atlasnet
from model.atlasnet.model_blocks import PointNet
import torch.nn as nn
import model.resnet as resnet
#from torchvision import models


class EncoderDecoder(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, args):
        super(EncoderDecoder, self).__init__()
        if args.SVR:
            self.encoder = resnet.resnet18(pretrained=False, num_classes=args.bottleneck_size)
        else:
            self.encoder = PointNet(nlatent=args.bottleneck_size)

        self.decoder = Atlasnet(args)
        self.to(args.device)


        self.eval()

    def forward(self, x, train=True):
        return self.decoder(self.encoder(x), train=train)

    def generate_mesh(self, x):
        return self.decoder.generate_mesh(self.encoder(x))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

import os
from model.im2mesh.psgn import models
import model.resnet as resnet
from model.im2mesh.psgn.models.decoder import Decoder


def get_model(bottleneck_size, pts_num):
    encoder = resnet.resnet18(pretrained=False, num_classes=bottleneck_size)
    decoder = Decoder(dim=3, bottleneck_size=bottleneck_size, n_points=pts_num)
    model = models.PCGN(decoder, encoder)
    return model







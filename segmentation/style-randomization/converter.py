import os

import torch
import torch.nn as nn
from torch.utils.serialization import load_lua

from models import VGGEncoder, VGGDecoder

def weight_assign(lua, pth, maps):
    for k, v in maps.items():
        getattr(pth, k).weight = nn.Parameter(lua.get(v).weight.float())
        getattr(pth, k).bias = nn.Parameter(lua.get(v).bias.float())

if __name__ == '__main__':
    ## VGGEncoder4
    vgg4 = load_lua('pretrained/encoder.t7', long_size=8)
    e4 = VGGEncoder()
    weight_assign(vgg4, e4, {
        'conv0': 0,
        'conv1_1': 2,
        'conv1_2': 5,
        'conv2_1': 9,
        'conv2_2': 12,
        'conv3_1': 16,
        'conv3_2': 19,
        'conv3_3': 22,
        'conv3_4': 25,
        'conv4_1': 29,
    })
    torch.save(e4.state_dict(), 'pretrained/encoder_pretrained.pth')
    
    ## VGGDecoder4
    inv4 = load_lua('pretrained/recon.t7', long_size=8)
    d4 = VGGDecoder(num_class=3,use_softmax=False)
    weight_assign(inv4, d4, {
        'conv4_1': 1,
        'conv3_4': 5,
        'conv3_3': 8,
        'conv3_2': 11,
        'conv3_1': 14,
        'conv2_2': 18,
        'conv2_1': 21,
        'conv1_2': 25,
        'conv1_1': 28,
    })
    torch.save(d4.state_dict(), 'pretrained/recon_pretrained.pth')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import os
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from skimage import io, transform
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image
import time
import copy
import sys
import math
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import office_model

def my_lr_scheduler(optimizer, preval, curval, prelr, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 when current train_loss + val_loss < previous train_loss + val_loss"""
    if prelr > 1:
        lr = init_lr
        print('LR is set to {}'.format(lr))
    elif curval > preval:
        lr = (optimizer.param_groups[0])['lr'] * 0.1
        print('LR is set to {}'.format(lr))
    else:
        lr = prelr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr



if __name__ == '__main__':
    data_transform = {
        'Training': transforms.Compose([
            transforms.Scale([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
        ]),
        'Validate': transforms.Compose([
            transforms.Scale([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
        ]),
        'Testing': transforms.Compose([
            transforms.Scale([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
        ])
    }

    datasetRoot = 'D:/dataset/office/domain_adaptation_images/'
    datasetNames = ['amazon', 'dslr', 'webcam']

    phases = ['amazon', 'dslr', 'webcam']
    #phases = ['dslr', 'amazon', 'webcam']
    #phases = ['webcam', 'amazon', 'dslr']
    batch_size = 8
    init_lr = 0.001
    weight_decay = 0.001
    gpu_id = 0
    num_epochs = 40

    data_transform = {x: transforms.Compose([
            transforms.Scale([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
        ]) for x in datasetNames}

    dsets = {
        x: ImageFolder(os.path.join(os.path.join(datasetRoot, x), 'images'),
                                     data_transform[x])
        for x in datasetNames}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=4)
                    for x in datasetNames}
    dset_sizes = {x: len(dsets[x]) for x in datasetNames}

    model = office_model.office_model()

    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    model_best = model.train_model(model, dset_loaders, dset_sizes, optimizer, my_lr_scheduler, init_lr, phases, batch_size=batch_size,
                num_epochs=num_epochs, gpu_id=gpu_id, save_best='Training')


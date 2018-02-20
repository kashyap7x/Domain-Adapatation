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
import torch.nn.functional as F



class office_model(nn.Module):
    def __init__(self):
        super(office_model, self).__init__()
        self.resnet = models.resnet50(True)
        #self.alexnet = models.alexnet(True)
        #self.resnet = models.resnet18(False)
        self.fc = nn.Linear(2048, 31)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 31),
        )

    def alexnetOutput(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def resnetOutput(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.resnetOutput(x)
        x = self.fc(x)
        #x = self.alexnetOutput(x)
        return x

    def train_model(self, model, dset_loaders, dset_sizes, optimizer, lr_scheduler, init_lr, phases, batch_size=1,
                    num_epochs=20, gpu_id=-1, save_best = 'Training'):
        since = time.time()
        if gpu_id >= 0:
            # manually set to cuda
            model = model.cuda(gpu_id)
        else:
            model = model.cpu()
        # initialize
        best_model = model
        best_val_loss = 100000
        prelr = 10000
        preval = 10000
        curval = 1000
        for epoch in range(num_epochs):
            info = {}
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in phases:
                if phase == phases[0]:
                    optimizer, prelr = lr_scheduler(optimizer, preval, curval, prelr, init_lr=init_lr)
                    preval = curval
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                # tqdm is a progress bar wrapper
                for data in tqdm(dset_loaders[phase]):
                    # get the inputs
                    inputs, target = data
                    # wrap them in Variable
                    if gpu_id >= 0:
                        inputs, target = Variable(inputs.cuda(gpu_id)), Variable(target.cuda(gpu_id))
                    else:
                        inputs, target = Variable(inputs), Variable(target)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model.forward(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    crossentropy = nn.CrossEntropyLoss()
                    loss = crossentropy(outputs, target)

                    # backward + optimize only if in training phase
                    if phase == phases[0]:
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.data[0] * inputs.size(0)
                    running_corrects += torch.sum(preds == target.data)

                epoch_loss = running_loss / dset_sizes[phase]
                epoch_acc = running_corrects / dset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                if phase == phases[0]:
                    curval = epoch_loss
                if phase == phases[1] and save_best == 'Training':
                    best_model = copy.deepcopy(model)
                elif phase == phases[1] and save_best == 'Validate':
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model = copy.deepcopy(model)

        return best_model

    def train_model_SGD(self, model, dset_loaders, dset_sizes, optimizer, lr_scheduler, init_lr, phases, gamma, power, batch_size=1, gpu_id=-1, maxIter = 10000, testInterv = 500, save_best = 'Training'):
        since = time.time()
        if gpu_id >= 0:
            # manually set to cuda
            model = model.cuda(gpu_id)
        else:
            model = model.cpu()
        # initialize
        best_model = model
        best_val_loss = 100000
        prelr = 10000
        preval = 10000
        curval = 1000
        epoch = 0
        iter = 0
        while True:
            # Each epoch has a training and validation phase
            print('Epoch {}'.format(epoch))
            print('-' * 10)
            for phase in phases:
                if phase == phases[0]:
                    optimizer, prelr = lr_scheduler(optimizer, gamma, power, iter, init_lr=init_lr)
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                # tqdm is a progress bar wrapper
                for data in tqdm(dset_loaders[phase]):
                    # get the inputs
                    inputs, target = data
                    # wrap them in Variable
                    if gpu_id >= 0:
                        inputs, target = Variable(inputs.cuda(gpu_id)), Variable(target.cuda(gpu_id))
                    else:
                        inputs, target = Variable(inputs), Variable(target)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model.forward(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    crossentropy = nn.CrossEntropyLoss()
                    loss = crossentropy(outputs, target)

                    # backward + optimize only if in training phase
                    if phase == phases[0]:
                        loss.backward()
                        optimizer.step()
                        iter += 1
                        if iter >= maxIter:
                            break

                    # statistics
                    running_loss += loss.data[0] * inputs.size(0)
                    running_corrects += torch.sum(preds == target.data)

                epoch_loss = running_loss / dset_sizes[phase]
                epoch_acc = running_corrects / dset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                if phase == phases[0]:
                    curval = epoch_loss
                if phase == phases[1] and save_best == 'Training':
                    best_model = copy.deepcopy(model)
                elif phase == phases[1] and save_best == 'Validate':
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model = copy.deepcopy(model)
            epoch += 1
        return best_model
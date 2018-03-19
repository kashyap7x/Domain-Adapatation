import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import copy
from tqdm import tqdm


class office_model(nn.Module):
    def __init__(self):
        super(office_model, self).__init__()
        self.resnet = models.resnet50(True)
        # self.alexnet = models.alexnet(True)
        # self.resnet = models.resnet18(True)
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

    def train_model(self, model, dset_loaders, dset_sizes, optimizer, lr_scheduler, init_lr, phases, gamma, power, gpu_id=-1, maxIter = 10000, save_best = 'Training'):
        if gpu_id >= 0:
            # manually set to cuda
            model = model.cuda(gpu_id)
        else:
            model = model.cpu()
        # initialize
        best_model = model
        best_val_loss = 100000
        prelr = 10000
        epoch = 0
        iter = 0
        while True:
            # Each epoch has a training and 2 validation phases
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
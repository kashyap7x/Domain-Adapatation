import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm
seed = 1337
log_interval = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
CE = nn.CrossEntropyLoss()
MSE = nn.MSELoss()

class syn_net(nn.Module):
    '''
    Synthetic gradient module. Currently a single hidden layer network
    '''
    def __init__(self):
        super(syn_net, self).__init__()
        self.fc1 = nn.Linear(31, 10)
        self.fc2 = nn.Linear(10, 31)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class office_model(nn.Module):
    '''
    Main classification model. ResNet or Alexet
    '''
    def __init__(self):
        super(office_model, self).__init__()

        # For ResNet-50
        #self.resnet = models.resnet50(True)
        #self.fc = nn.Linear(2048, 31)

        # For ResNet-18
        self.resnet = models.resnet18(True)
        self.fc = nn.Linear(512, 31)

        '''
        # For AlexNet
        self.alexnet = models.alexnet(True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 31),
        )
        '''

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

    '''
    def alexnetOutput(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    '''

    def forward(self, x):
        # For ResNets
        x = self.resnetOutput(x)
        x = self.fc(x)

        # For AlexNet
        #x = self.alexnetOutput(x)
        return x


# Custom hook to store intermediate gradients of the loss function
grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


def train_model(model, synth, model_optimizer, synth_optimizer, dset_sizes, dset_loaders, lr_scheduler, init_lr, phases, gamma, power, maxIter=10000, maxEpoch=None, gpu_id=-1, save_best = 'Training'):
    if gpu_id >= 0:
        # Manually set to cuda
        model = model.cuda(gpu_id)
    else:
        model = model.cpu()

    # Initialize
    best_model = model
    best_val_loss = 1e6

    # For custom step scheduler
    prelr = 1e5
    preval = 1e5
    curval = 1e4

    if maxEpoch:
        maxIter = dset_sizes[0] * maxEpoch

    epoch = 0
    iter = 0
    while True:
        # Each epoch has a training and validation phase
        print('Epoch {}'.format(epoch))
        print('-' * 10)
        for phase in phases:
            if phase == phases[0]:
                # For caffe scheduler
                model_optimizer, prelr = lr_scheduler(model_optimizer, gamma, power, iter, init_lr=init_lr)

                # For custom step scheduler
                #model_optimizer, prelr = lr_scheduler(model_optimizer, preval, curval, prelr, init_lr=init_lr)
                #preval = curval

                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_syn_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # tqdm is a progress bar wrapper
            for data in tqdm(dset_loaders[phase]):
                # Get the inputs
                inputs, target = data
                # Wrap them in Variable
                if gpu_id >= 0:
                    inputs, target = Variable(inputs.cuda(gpu_id)), Variable(target.cuda(gpu_id))
                else:
                    inputs, target = Variable(inputs), Variable(target)

                # Zero the parameter gradients of both optimizers
                model_optimizer.zero_grad()
                synth_optimizer.zero_grad()

                # Forward classification model
                outputs = model.forward(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = CE(outputs, target)

                # Backward + optimize only if in training phase
                if phase == phases[0]:
                    # Store last layer gradients in dict
                    outputs.register_hook(save_grad('out'))
                    loss.backward(retain_graph=True)
                    model_optimizer.step()

                    # Forward + backward synthetic gradient module
                    syn_grads = synth.forward(outputs)
                    syn_loss = MSE(syn_grads, grads['out'].detach())
                    syn_loss.backward()
                    synth_optimizer.step()

                    iter += 1
                    if iter >= maxIter:
                        break

                # Statistics
                running_loss += loss.data[0] * inputs.size(0)
                if phase == phases[0]:
                    running_syn_loss += syn_loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == target.data)

            epoch_loss = running_loss / dset_sizes[phase]
            if phase == phases[0]:
                epoch_syn_loss = running_syn_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            if phase == phases[0]:
                print('{} Loss: {:.4f} Syn_Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_syn_loss, epoch_acc))
            else:
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
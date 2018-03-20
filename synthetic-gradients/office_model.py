import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class syn_net(nn.Module):
    """
    Synthetic gradient module. Currently a small network with 2 hidden layers of size hidden_size.
    """
    def __init__(self, input_dims=31, hidden_size=64):
        super(syn_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dims, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(input_dims, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        self.layer3 = nn.Linear(hidden_size*2, input_dims)

    def forward(self, x, w):
        x = self.layer1(x)
        w = w.unsqueeze(0).repeat(x.size(0), 1, 1)
        w = torch.mean(w, 2)
        w = self.layer2(w)
        x = self.layer3(torch.cat((x,w),1))
        return x


class office_model(nn.Module):
    """
    Main classification model. ResNet or Alexet
    """
    def __init__(self, init_lr, synth_lr, momentum, weight_decay):
        super(office_model, self).__init__()

        # For ResNet-50
        # self.resnet = models.resnet50(True)
        # self.fc = nn.Linear(2048, 31)

        # For ResNet-18
        self.resnet = models.resnet18(True)
        self.fc = nn.Linear(512, 31)

        # For AlexNet
        # self.alexnet = models.alexnet(True)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        # )
        # self.fc = nn.Linear(4096, 31)

        # Synthetic gradient generator for the final layer
        self._fc = syn_net()

        # Optimizers for main network and synthetic module
        self.model_optimizer = optim.SGD(self.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
        self.synth_optimizer = optim.Adam(self._fc.parameters(), lr=synth_lr)

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

    def alexnetOutput(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def forward(self, x):
        # For ResNets
        x = self.resnetOutput(x)

        # For AlexNet
        # x = self.alexnetOutput(x)

        # Predictions
        x = self.fc(x)
        w = self.fc.weight

        # Estimated gradients
        grad_fc = self._fc(x, w)
        return x, grad_fc



class synthetic_trainer():
    """
    Trainer class to optimize the model
    """
    def __init__(self, model, phases):
        self.model = model
        self.phases = phases

        # Loss functions for the two networks
        self.model_loss = nn.CrossEntropyLoss()
        self.synth_loss = nn.MSELoss()

    def save_grad(self, name):
        """
        Custom hook to store intermediate gradients of the loss function
        """
        def hook(grad):
            self.backprop_grads[name] = grad
            self.backprop_grads[name].volatile = False
        return hook

    def optimize_model(self, with_synthetic_grads, inputs, target, model_optimizer, forward):
        """
        Updates the parameters of the main model
        :param with_synthetic_grads: if True, uses synthetetic gradients for updates, else does standard backprop
        :param inputs: images
        :param target: labels
        :param model_optimizer: Pytorch optimizer for the main network
        :param forward: forward function of the main model
        """
        # Zero the parameter gradients
        model_optimizer.zero_grad()

        # Forward classification model
        out, grad = forward(inputs)
        if with_synthetic_grads:
            # Use synthetic gradients for backward pass
            out.backward(grad.detach().data)
        else:
            # Normal backprop
            loss = self.model_loss(out, target)
            loss.backward()

        # Step main optimizer
        model_optimizer.step()
        out.detach()

    def optimize_synth(self, phase, inputs, target, model_optimizer, synth_optimizer, forward):
        """
        If in train phase, updates the parameters of the synthetic model. Also computes loss/accuracy statistics (in all phases)
        :param phase: current phase
        :param inputs: images
        :param target: labels
        :param model_optimizer: Pytorch optimizer for the main network
        :param synth_optimizer: Pytorch optimizer for the synthetic module
        :param forward: forward function of the main model
        :return:
        """
        # Zero the parameter gradients of both optimizers
        model_optimizer.zero_grad()
        synth_optimizer.zero_grad()

        # Forward classification model
        outputs, grads = forward(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = self.model_loss(outputs, target)

        # Clear old cached copy of gradients
        self.backprop_grads = {}

        # Backward + optimize only if in training phase
        if phase == self.phases[0]:
            # Store last layer gradients in dict
            outputs.register_hook(self.save_grad('out'))
            # Do the back pass but not the update
            loss.backward(retain_graph=True)

            # Forward + backward synthetic gradient module
            syn_loss = self.synth_loss(grads, self.backprop_grads['out'].detach())
            syn_loss.backward()

            # Step synthetic optimizer
            synth_optimizer.step()
            return loss, syn_loss, preds
        else:
            # Return statistics if testing
            return loss, preds


    def train_model(self, ratios, batch_size, dset_sizes,dset_loaders, lr_scheduler, init_lr, gamma, power, maxIter=10000, maxEpoch=None, gpu_id=-1, save_best = 'Training'):
        """
        Main training loop
        :param ratios: sampling ratio for the source (real grads) and target (synthetic grads) domains
        :param batch_size: batch size for training
        :param dset_sizes: number of images in the datasets
        :param dset_loaders: mini-batch loaders
        :param lr_scheduler: learning rate decay function
        :param init_lr: initial learning rate
        :param gamma: used for exponential LR decay
        :param power: used for exponential LR decay
        :param maxIter: maximum iterations
        :param maxEpoch: maximum epochs, overwrites maximum iterations if both are provided
        :param gpu_id: GPU to use, -1 for CPU
        :param save_best: save best model based on 'Training' or 'Validation' accuracy
        :return: the best model after training for maxIter iterations
        """

        if gpu_id >= 0:
            # Manually set to cuda
            self.model.cuda(gpu_id)
        else:
            self.model.cpu()

        # Initialize
        best_model = self.model
        best_val_loss = 1e6

        # For custom step scheduler
        prelr = 1e5
        preval = 1e5
        curval = 1e4

        # Overwrite default iterations if epochs given as input
        if maxEpoch:
            maxIter = dset_sizes[0] * maxEpoch

        epoch = 0
        iteration = 0
        while True:
            # Each epoch has a training and one or more validation phases
            print('Epoch {}'.format(epoch))
            print('-' * 10)
            for phase in self.phases:
                if phase == self.phases[0]:
                    # For caffe scheduler
                    self.model.model_optimizer, prelr = lr_scheduler(self.model.model_optimizer, gamma, power, iteration, init_lr=init_lr)

                    # For custom step scheduler
                    #model_optimizer, prelr = lr_scheduler(model_optimizer, preval, curval, prelr, init_lr=init_lr)
                    #preval = curval

                    # Set model to training mode
                    self.model.train(True)
                else:
                    # Set model to evaluate mode
                    self.model.train(False)

                # Initialization for statistics
                running_loss = 0.0
                running_syn_loss = 0.0
                average_gradient = 0.0
                running_corrects = 0

                # Iterate over data.
                # tqdm is a progress bar wrapper
                for i in tqdm(range(dset_sizes[phase]//batch_size + 1), ncols=100):
                    # Get the inputs
                    if phase == self.phases[0]:
                        sample, key = utils.random_sampler(ratios, dset_loaders)
                        inputs, target = sample
                    else:
                        key = phase
                        inputs, target = iter(dset_loaders[key]).next()
                    # Wrap them in Variable
                    if gpu_id >= 0:
                        inputs, target = Variable(inputs.cuda(gpu_id)), Variable(target.cuda(gpu_id))
                    else:
                        inputs, target = Variable(inputs), Variable(target)

                    if phase == self.phases[0]:
                        # Check for batch size 1 (causes errors with batch normalization)
                        if(len(target.data) != 1):
                            # Optimize the main model
                            self.optimize_model(key!=phase, inputs, target, self.model.model_optimizer, self.model.forward)

                            # Optimize the synthetic module and collect statistics
                            loss, syn_loss, preds = self.optimize_synth(phase, inputs, target, self.model.model_optimizer,self.model.synth_optimizer, self.model)
                        iteration += 1
                    else:
                        # Collect statistics
                        loss, preds = self.optimize_synth(phase, inputs, target, self.model.model_optimizer,self.model.synth_optimizer, self.model)

                    # Statistics
                    # Do not take train statistics if batch size was 1
                    if (len(target.data) != 1) or phase != self.phases[0]:
                        running_loss += loss.data[0] * inputs.size(0)
                        running_corrects += torch.sum(preds == target.data)
                        if phase == self.phases[0]:
                            average_gradient += torch.sum((self.backprop_grads['out'].detach().data)**2)
                            running_syn_loss += syn_loss.data[0] * inputs.size(0)

                # Display statistics
                epoch_loss = running_loss / dset_sizes[phase]
                epoch_acc = running_corrects / dset_sizes[phase]
                if phase == self.phases[0]:
                    epoch_scaled_syn_loss = np.sqrt(running_syn_loss / average_gradient)
                    epoch_syn_loss = running_syn_loss / dset_sizes[phase]
                    print('{} Loss: {:.4f} Syn_Loss: {:.4f} Scaled_Syn_Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_syn_loss, epoch_scaled_syn_loss, epoch_acc))
                else:
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))

                # Save best model
                if phase == self.phases[0]:
                    curval = epoch_loss
                if phase == self.phases[1] and save_best == 'Training':
                    best_model = copy.deepcopy(self.model)
                elif phase == self.phases[1] and save_best == 'Validate':
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model = copy.deepcopy(self.model)
            epoch += 1
            
            if iteration >= maxIter:
                break

        return best_model
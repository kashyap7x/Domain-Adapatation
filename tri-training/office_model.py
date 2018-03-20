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


class branch_net(nn.Module):
    """
    Classifier branch module. Currently a small network with 1 hidden layer of size hidden_size.
    """
    def __init__(self, input_dims=2048, hidden_size=64, output_dims=31):
        super(branch_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dims, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        self.layer2 = nn.Linear(hidden_size, output_dims)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def get_weights(self):
        return self.layer2.weight


class syn_net(nn.Module):
    """
    Synthetic annotation module. Currently a network with 6 hidden layers (one from each input) of size hidden_size.
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

        self.layer3 = nn.Sequential(
            nn.Linear(input_dims, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(input_dims, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Linear(input_dims, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(input_dims, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        self.layer7 = nn.Linear(hidden_size*6, input_dims)

    def forward(self, x1, w1, x2, w2, xt, wt):
        x1 = self.layer1(x1)
        w1 = w1.unsqueeze(0).repeat(x1.size(0), 1, 1)
        w1 = torch.mean(w1, 2)
        w1 = self.layer2(w1)

        x2 = self.layer3(x2)
        w2 = w2.unsqueeze(0).repeat(x2.size(0), 1, 1)
        w2 = torch.mean(w2, 2)
        w2 = self.layer4(w2)

        xt = self.layer5(xt)
        wt = wt.unsqueeze(0).repeat(xt.size(0), 1, 1)
        wt = torch.mean(wt, 2)
        wt = self.layer6(wt)

        g = self.layer7(torch.cat((x1,w1,x2,w2,xt,wt),1))
        return g


class office_model(nn.Module):
    """
    Main classification model. ResNet backbone
    """
    def __init__(self, init_lr, synth_lr, momentum, weight_decay):
        super(office_model, self).__init__()

        # Shared layers
        self.F = models.resnet50(True)

        # Branches
        self.F1 = branch_net(2048, 64, 31)
        self.F2 = branch_net(2048, 64, 31)
        self.Ft = branch_net(2048, 64, 31)

        # Synthetic annotation generator
        self.S = syn_net()

        # Optimizers for all networks
        self.optimizer_F = optim.SGD(self.F.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
        self.optimizer_F1 = optim.SGD(self.F1.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
        self.optimizer_F2 = optim.SGD(self.F2.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
        self.optimizer_Ft = optim.SGD(self.Ft.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
        self.optimizer_S = optim.Adam(self.S.parameters(), lr=synth_lr)

    def forward_F(self, x):
        x = self.F.conv1(x)
        x = self.F.bn1(x)
        x = self.F.relu(x)
        x = self.F.maxpool(x)

        x = self.F.layer1(x)
        x = self.F.layer2(x)
        x = self.F.layer3(x)
        x = self.F.layer4(x)

        x = self.F.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        # Forward through shared layers
        x = self.forward_F(x)

        # Predictions and weights of branches
        y1 = self.F1(x)
        w1 = self.F1.get_weights()

        y2 = self.F2(x)
        w2 = self.F2.get_weights()

        yt = self.Ft(x)
        wt = self.Ft.get_weights()

        g = self.S(y1, w1, y2, w2, yt, wt)
        return (y1, y2, yt), g


class synthetic_trainer():
    """
    Trainer class to optimize the model
    """
    def __init__(self, model, phases, alpha, beta, tau):
        self.model = model
        self.phases = phases
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        # Loss functions for the two networks
        self.model_loss = nn.CrossEntropyLoss()
        self.synth_loss = nn.MSELoss()

    def optimize_model(self, adapt, inputs, target):
        """
        Updates the parameters of the main model
        :param adapt: if True, uses synthetetic gradients for updates, else does standard backprop
        :param inputs: images
        :param target: labels
        :param model_optimizer: Pytorch optimizer for the main network
        :param forward: forward function of the main model
        """
        # Zero the parameter gradients
        self.model.optimizer_F.zero_grad()
        self.model.optimizer_F1.zero_grad()
        self.model.optimizer_F2.zero_grad()
        self.model.optimizer_Ft.zero_grad()
        self.model.optimizer_S.zero_grad()

        # Forward classification model
        (y1, y2, yt), g = self.model.forward(inputs)
        _, preds_1 = torch.max(y1.data, 1)
        _, preds_2 = torch.max(y2.data, 1)
        _, preds_t = torch.max(yt.data, 1)
        value_g, preds_g = torch.max(g.data, 1)

        if adapt:
            equal_idx = (torch.eq(preds_1, preds_2))
            conf_idx = (value_g > self.tau)
            adapt_idx = torch.nonzero(equal_idx & conf_idx).squeeze()
            loss_F1 = utils.make_variable(torch.zeros(1))
            loss_F2 = utils.make_variable(torch.zeros(1))
            loss_Ft = utils.make_variable(torch.zeros(1))
            loss_syn = utils.make_variable(torch.zeros(1))
            if len(adapt_idx.size()) > 0:
                loss_F1 = self.model_loss(y1[adapt_idx, :], utils.make_variable(preds_g[adapt_idx]))
                loss_F2 = self.model_loss(y2[adapt_idx, :], utils.make_variable(preds_g[adapt_idx]))
                loss_Ft = self.model_loss(yt[adapt_idx, :], utils.make_variable(preds_g[adapt_idx]))
                loss_syn = self.model_loss(g[adapt_idx, :], utils.make_variable(preds_g[adapt_idx]))
        else:
            loss_F1 = self.model_loss(y1, target)
            loss_F2 = self.model_loss(y2, target)
            loss_Ft = utils.make_variable(torch.zeros(1))
            loss_syn = self.model_loss(g, target)

        loss_similiarity = utils.similiarity_penalty(self.model.F1.get_weights(), self.model.F2.get_weights())
        loss_F = loss_F1 + loss_F2 + loss_Ft + self.alpha * loss_similiarity + self.beta * loss_syn
        loss_F.backward()

        # Step main optimizers
        self.model.optimizer_F.step()
        self.model.optimizer_F1.step()
        self.model.optimizer_F2.step()
        self.model.optimizer_Ft.step()
        self.model.optimizer_S.step()
        return loss_F1, loss_F2, loss_Ft, loss_syn, loss_similiarity, preds_1, preds_2, preds_t, preds_g

    def collect_stats(self, inputs, target):
        """
        Computes loss/accuracy statistics
        :param inputs: images
        :param target: labels
        :return:
        """

        # Forward classification model
        (y1, y2, yt), g = self.model.forward(inputs)
        _, preds_1 = torch.max(y1.data, 1)
        _, preds_2 = torch.max(y2.data, 1)
        _, preds_t = torch.max(yt.data, 1)
        _, preds_g = torch.max(g.data, 1)

        loss_F1 = self.model_loss(y1, target)
        loss_F2 = self.model_loss(y2, target)
        loss_Ft = self.model_loss(yt, target)
        loss_syn = self.model_loss(g, target)
        loss_similiarity = utils.similiarity_penalty(self.model.F1.get_weights(), self.model.F2.get_weights())
        return loss_F1, loss_F2, loss_Ft, loss_syn, loss_similiarity, preds_1, preds_2, preds_t, preds_g

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
                    self.model.optimizer_F, prelr = lr_scheduler(self.model.optimizer_F, gamma, power, iteration, init_lr=init_lr)
                    self.model.optimizer_F1, prelr = lr_scheduler(self.model.optimizer_F1, gamma, power, iteration,
                                                                 init_lr=init_lr)
                    self.model.optimizer_F2, prelr = lr_scheduler(self.model.optimizer_F2, gamma, power, iteration,
                                                                 init_lr=init_lr)
                    self.model.optimizer_Ft, prelr = lr_scheduler(self.model.optimizer_Ft, gamma, power, iteration,
                                                                 init_lr=init_lr)

                    # Set model to training mode
                    self.model.train(True)
                else:
                    # Set model to evaluate mode
                    self.model.train(False)

                # Initialization for statistics
                running_loss_F1 = 0.0
                running_corrects_F1 = 0
                running_loss_F2 = 0.0
                running_corrects_F2 = 0
                running_loss_Ft = 0.0
                running_corrects_Ft = 0
                running_loss_syn = 0.0
                running_corrects_syn = 0
                running_loss_similarity = 0.

                # Iterate over data.
                # tqdm is a progress bar wrapper
                for i in tqdm(range(dset_sizes[phase]//batch_size), ncols=100):
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
                            loss_F1, loss_F2, loss_Ft, loss_syn, loss_similiarity, preds_1, preds_2, preds_t, preds_g = \
                                self.optimize_model(key!=phase, inputs, target)
                        iteration += 1
                    else:
                        # Collect statistics
                        loss_F1, loss_F2, loss_Ft, loss_syn, loss_similiarity, preds_1, preds_2, preds_t, preds_g = \
                            self.collect_stats(inputs, target)

                    # Statistics
                    # Do not take train statistics if batch size was 1
                    if (len(target.data) != 1) or phase != self.phases[0]:
                        running_loss_F1 += loss_F1.data[0] * inputs.size(0)
                        running_loss_F2 += loss_F2.data[0] * inputs.size(0)
                        running_loss_Ft += loss_Ft.data[0] * inputs.size(0)
                        running_loss_syn += loss_syn.data[0] * inputs.size(0)
                        running_loss_similarity += loss_similiarity.data[0] * inputs.size(0)
                        running_corrects_F1 += torch.sum(preds_1 == target.data)
                        running_corrects_F2 += torch.sum(preds_2 == target.data)
                        running_corrects_Ft += torch.sum(preds_t == target.data)
                        running_corrects_syn += torch.sum(preds_g == target.data)

                # Display statistics
                epoch_loss_F1 = running_loss_F1 / dset_sizes[phase]
                epoch_loss_F2 = running_loss_F2 / dset_sizes[phase]
                epoch_loss_Ft = running_loss_Ft / dset_sizes[phase]
                epoch_loss_syn = running_loss_syn / dset_sizes[phase]
                epoch_loss_similarity = running_loss_similarity / dset_sizes[phase]
                epoch_acc_F1 = running_corrects_F1 / dset_sizes[phase]
                epoch_acc_F2 = running_corrects_F2 / dset_sizes[phase]
                epoch_acc_Ft = running_corrects_Ft / dset_sizes[phase]
                epoch_acc_syn = running_corrects_syn / dset_sizes[phase]

                print('{} Loss 1: {:.4f} Acc 1: {:.4f} Loss 2: {:.4f} Acc 2: {:.4f} Loss Sim: {:.4f}\n'
                      'Loss T: {:.4f} Acc T: {:.4f} Loss Syn: {:.4f} Acc Syn: {:.4f}'.format(
                    phase, epoch_loss_F1, epoch_acc_F1, epoch_loss_F2, epoch_acc_F2, epoch_loss_similarity,
                    epoch_loss_Ft, epoch_acc_Ft, epoch_loss_syn, epoch_acc_syn))

                # Save best model
                if phase == self.phases[0]:
                    curval = epoch_loss_syn
                if phase == self.phases[1] and save_best == 'Training':
                    best_model = copy.deepcopy(self.model)
                elif phase == self.phases[1] and save_best == 'Validate':
                    if epoch_loss_syn < best_val_loss:
                        best_val_loss = epoch_loss_syn
                        best_model = copy.deepcopy(self.model)
            epoch += 1
            
            if iteration >= maxIter:
                break
                        
        return best_model

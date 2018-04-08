import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

class branch_net(nn.Module):
    """
    Classifier branch module. Currently a small network with 1 hidden layer of size hidden_size.
    """
    def __init__(self, input_dims=2048, hidden_size=64, output_dims=31, loss_weights=None):
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
        return nn.functional.log_softmax(x)

    def get_weights(self):
        return self.layer2.weight


class syn_net(nn.Module):
    """
    Synthetic annotation module. Currently a network with 6 layers (one from each input, and a classifier) of size hidden_size.
    """
    def __init__(self, input_dims=2048, hidden_size=64, output_dims=31):
        super(syn_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dims, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(output_dims, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(output_dims, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.layer4 = nn.Linear(hidden_size*3+2, output_dims)

    def forward(self, x, x1, w1, d):
        x = self.layer1(x)

        x1 = self.layer2(x1)
        w1 = w1.unsqueeze(0).repeat(x1.size(0), 1, 1)
        w1 = torch.mean(w1, 2)
        w1 = self.layer3(w1)

        g = self.layer4(torch.cat((x,x1,w1,d),1))
        return nn.functional.log_softmax(g)


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
        self.D = branch_net(2048, 64, 2)

        # Synthetic annotation generator
        self.S = syn_net(2048, 64, 31)

        # Optimizers for all networks
        self.optimizer_F = optim.SGD(self.F.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
        self.optimizer_F1 = optim.SGD(self.F1.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
        self.optimizer_D = optim.SGD(self.D.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
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

    def forward(self, x, alpha):
        # Forward through shared layers
        x = self.forward_F(x)

        # Predictions and weights of branches
        y1 = self.F1(x)
        w1 = self.F1.get_weights()

        xrev = utils.ReverseLayerF.apply(x, alpha)
        d = self.D(xrev)

        g = self.S(x, y1, w1, d)
        return (y1, d), g


class synthetic_trainer():
    """
    Trainer class to optimize the model
    """
    def __init__(self, model, phases, ratios, alpha, beta, tau):
        self.model = model
        self.phases = phases
        self.ratios = ratios
        self.alpha = alpha
        self.beta = beta
        self.tau = tau

        # Loss functions for the two networks
        self.model_loss = nn.NLLLoss()
        loss_weights = utils.make_variable(torch.FloatTensor([ratios[phases[1]], ratios[phases[0]]]))
        self.domain_loss = nn.NLLLoss(weight=loss_weights)

    def optimize_model(self, adapt, inputs, target, alpha):
        """
        Updates the parameters of the main model
        :param adapt: if True, uses synthetetic gradients for updates, else does standard backprop
        :param inputs: images
        :param target: labels
        :param model_optimizer: Pytorch optimizer for the main network
        :param forward: forward function of the main model
        """
        batch_size = len(target)

        # Zero the parameter gradients
        self.model.optimizer_F.zero_grad()
        self.model.optimizer_F1.zero_grad()
        self.model.optimizer_D.zero_grad()
        self.model.optimizer_S.zero_grad()

        # Forward classification model
        (y1, d), g = self.model.forward(inputs, alpha)
        _, preds_1 = torch.max(y1.data, 1)
        _, preds_d = torch.max(d.data, 1)
        value_g, preds_g = torch.max(g.data, 1)

        if adapt:
            domain_label = torch.ones(batch_size)
            equal_idx = (torch.eq(utils.make_variable(torch.zeros(batch_size)).long().data, preds_d))
            conf_idx = (value_g > np.log(self.tau))
            adapt_idx = torch.nonzero(equal_idx & conf_idx).squeeze()
            loss_F1 = utils.make_variable(torch.zeros(1))
            loss_syn = utils.make_variable(torch.zeros(1))
            if len(adapt_idx.size()) > 0:
                loss_F1 = self.model_loss(y1[adapt_idx, :], utils.make_variable(preds_g[adapt_idx]))
                loss_syn = self.model_loss(g[adapt_idx, :], utils.make_variable(preds_g[adapt_idx]))
        else:
            domain_label = torch.zeros(batch_size)
            loss_F1 = self.model_loss(y1, target)
            loss_syn = self.model_loss(g, target)

        loss_D = self.domain_loss(d, utils.make_variable(domain_label.long()))
        loss_F = loss_F1 + loss_D +  self.beta * loss_syn
        loss_F.backward()

        # Step main optimizers
        self.model.optimizer_F.step()
        self.model.optimizer_F1.step()
        self.model.optimizer_D.step()
        self.model.optimizer_S.step()
        return loss_F1, loss_D, loss_syn, preds_1, preds_d, preds_g

    def collect_stats(self, inputs, target):
        """
        Computes loss/accuracy statistics
        :param inputs: images
        :param target: labels
        :return:
        """
        batch_size = len(target)

        # Forward classification model
        (y1, d), g = self.model.forward(inputs, 1.)
        _, preds_1 = torch.max(y1.data, 1)
        _, preds_d = torch.max(d.data, 1)
        _, preds_g = torch.max(g.data, 1)

        loss_F1 = self.model_loss(y1, target)
        domain_label = torch.ones(batch_size)
        loss_D = self.domain_loss(d, utils.make_variable(domain_label.long()))
        loss_syn = self.model_loss(g, target)
        return loss_F1, loss_D, loss_syn,  preds_1, preds_d, preds_g

    def train_model(self, batch_size, dset_sizes,dset_loaders, lr_scheduler, init_lr, gamma, power, maxIter=10000, maxEpoch=None, gpu_id=-1, save_best = 'Training'):
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

        # Overwrite default iterations if epochs given as input
        if maxEpoch:
            maxIter = dset_sizes[0] * maxEpoch

        epoch = 0
        iteration = 0

        while True:
            # Each epoch has a training and one or more validation phases
            print('Epoch {}'.format(epoch))
            print('-' * 10)
            alpha = 2. / (1. + np.exp(-self.alpha * (iteration / maxIter))) - 1
            ratios = self.ratios

            for phase in self.phases:
                if phase == self.phases[0]:
                    # For caffe scheduler
                    self.model.optimizer_F = lr_scheduler(self.model.optimizer_F, gamma, power, iteration, init_lr=init_lr)
                    self.model.optimizer_F1 = lr_scheduler(self.model.optimizer_F1, gamma, power, iteration,
                                                                 init_lr=init_lr)
                    self.model.optimizer_D = lr_scheduler(self.model.optimizer_D, gamma, power, iteration,
                                                                 init_lr=init_lr)

                    # Set model to training mode
                    self.model.train(True)
                else:
                    # Set model to evaluate mode
                    self.model.train(False)

                # Initialization for statistics
                running_loss_F1 = 0.0
                running_corrects_F1 = 0
                running_loss_D = 0.0
                running_corrects_D = 0
                running_loss_syn = 0.0
                running_corrects_syn = 0

                if phase == self.phases[0]:
                    for i in tqdm(range(dset_sizes[phase]//batch_size), ncols=100):
                        # Get the inputs
                        sample, key = utils.random_sampler(ratios, dset_loaders)
                        inputs, target = sample

                        # Wrap them in Variable
                        if gpu_id >= 0:
                            inputs, target = Variable(inputs.cuda(gpu_id)), Variable(target.cuda(gpu_id))
                        else:
                            inputs, target = Variable(inputs), Variable(target)

                        # Check for batch size 1 (causes errors with batch normalization)
                        if(len(target.data) != 1):
                            # Optimize the main model
                            loss_F1, loss_D, loss_syn, preds_1, preds_d, preds_g = \
                                self.optimize_model(key!=phase, inputs, target, alpha)
                        iteration += 1

                        if key!=phase:
                            domain_label = torch.ones(len(target.data))
                        else:
                            domain_label = torch.zeros(len(target.data))

                        # Statistics
                        # Do not take train statistics if batch size was 1
                        if (len(target.data) != 1):
                            running_loss_F1 += loss_F1.data[0] * inputs.size(0)
                            running_loss_D += loss_D.data[0] * inputs.size(0)
                            running_loss_syn += loss_syn.data[0] * inputs.size(0)
                            running_corrects_F1 += torch.sum(preds_1 == target.data)
                            running_corrects_D += torch.sum(preds_d == utils.make_variable(domain_label.long()).data)
                            running_corrects_syn += torch.sum(preds_g == target.data)
                else:
                    for inputs, target in tqdm(dset_loaders[phase], ncols=100):
                        # Get the inputs
                        # Wrap them in Variable
                        if gpu_id >= 0:
                            inputs, target = Variable(inputs.cuda(gpu_id)), Variable(target.cuda(gpu_id))
                        else:
                            inputs, target = Variable(inputs), Variable(target)

                        # Collect statistics
                        loss_F1, loss_D, loss_syn, preds_1, preds_d, preds_g = \
                            self.collect_stats(inputs, target)

                        domain_label = torch.ones(len(target.data))

                        # Statistics
                        running_loss_F1 += loss_F1.data[0] * inputs.size(0)
                        running_loss_D += loss_D.data[0] * inputs.size(0)
                        running_loss_syn += loss_syn.data[0] * inputs.size(0)
                        running_corrects_F1 += torch.sum(preds_1 == target.data)
                        running_corrects_D += torch.sum(preds_d == utils.make_variable(domain_label.long()).data)
                        running_corrects_syn += torch.sum(preds_g == target.data)

                # Display statistics
                epoch_loss_F1 = running_loss_F1 / dset_sizes[phase]
                epoch_loss_D = running_loss_D / dset_sizes[phase]
                epoch_loss_syn = running_loss_syn / dset_sizes[phase]
                epoch_acc_F1 = running_corrects_F1 / dset_sizes[phase]
                epoch_acc_D = running_corrects_D / dset_sizes[phase]
                epoch_acc_syn = running_corrects_syn / dset_sizes[phase]

                print('{} Loss 1: {:.4f} Acc 1: {:.4f} Loss D: {:.4f} Acc D: {:.4f} Loss Syn: {:.4f} Acc Syn: {:.4f}'.format(
                    phase, epoch_loss_F1, epoch_acc_F1, epoch_loss_D, epoch_acc_D, epoch_loss_syn, epoch_acc_syn))

                # Save best model
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

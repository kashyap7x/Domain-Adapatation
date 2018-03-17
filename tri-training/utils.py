import numpy as np
import torch
from torch.autograd import Variable

def my_lr_scheduler(optimizer, preval, curval, prelr, init_lr=0.001):
    """
    Decay learning rate by a factor of 0.1 when current train_loss + val_loss < previous train_loss + val_loss
    """
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


def inv_lr_scheduler(optimizer, gamma, power, iter, init_lr=0.001):
    """
    Inverse exponential LR decay every iteration (based on caffe implementation)
    """
    lr = init_lr * (1 + gamma * iter) ** (- power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr


def random_sampler(ratios, dset_loaders):
    """
    Sampler that gets batches from different datasets based on a sampling ratio
    :param ratios: a dict with phase as key and ratio as value
    :param dset_loaders: a dict with phase as key and torch.utils.data.DataLoader as value
    :return: randomly return next batch according to ratios
    """
    ratio_sum = 0
    for key in ratios:
        ratio_sum += ratios[key]
    prob = np.random.rand(1)
    ratio_sum_tmp = 0
    for key in ratios:
        ratio_sum_tmp += ratios[key] / ratio_sum
        if ratio_sum_tmp >= prob:
            return iter(dset_loaders[key]).next(), key


def similiarity_penalty(F1, F2):
    """
    Calculate similiarity penalty |W_1^T W_2|.
    """
    return torch.sum(torch.abs(torch.mm(F1.transpose(0, 1), F2)))


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    if torch.cuda.is_available():
        y_one_hot = y_one_hot.cuda()
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot
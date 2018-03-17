import numpy as np
import pickle

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
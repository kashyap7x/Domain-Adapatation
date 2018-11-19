# System libs
import os
import time
# import math
import random
import argparse
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.io import loadmat
from scipy.misc import imresize, imsave
# Our libs
from dataset import GTA, CityScapes, BDD, trainID2Class
from models import ModelBuilder
from utils import AverageMeter, colorEncode, accuracy, make_variable, intersectionAndUnion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def forward_with_loss(nets, batch_data, is_train=True):
    (net_encoder, net_decoder_1, net_decoder_2, crit) = nets
    (imgs, segs, infos) = batch_data

    # feed input data
    if is_train:
        input_img = Variable(imgs)
        label_seg = Variable(segs)
    else:
        with torch.no_grad():
            input_img = Variable(imgs)
            label_seg = Variable(segs)

    input_img = input_img.cuda()
    label_seg = label_seg.cuda()

    # forward
    conv_feat = net_encoder(input_img[:,:3,:,:])
    pred_featuremap_1 = net_decoder_1(conv_feat)
    pred_featuremap_2 = net_decoder_2(pred_featuremap_1,input_img[:,3:,:,:])
    
    err = crit(pred_featuremap_2, label_seg)
    
    return pred_featuremap_2, err


# train one epoch
def train(nets, loader, optimizers, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    for net in nets:
        if not args.fix_bn:
            net.train()
        else:
            net.eval()

    # main loop
    tic = time.time()
    for i, batch_data in enumerate(loader):
        data_time.update(time.time() - tic)
        for net in nets:
            net.zero_grad()

        # forward pass
        pred, err = forward_with_loss(nets, batch_data, is_train=True)

        # Backward
        err.backward()

        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # calculate accuracy, and display
        if i % args.disp_iter == 0:
            acc, _ = accuracy(batch_data, pred)

            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {}, lr_decoder: {}, '
                  'Accuracy: {:4.2f}%, Loss: {}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_encoder, args.lr_decoder,
                          acc * 100, err.data.item()))

            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.data.item())
            history['train']['acc'].append(acc)


def evaluate(nets, loader, loader_2, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    loss_meter_2 = AverageMeter()
    acc_meter_2 = AverageMeter()
    intersection_meter_2 = AverageMeter()
    union_meter_2 = AverageMeter()

    # switch to eval mode
    for net in nets:
        net.eval()

    for i, batch_data in enumerate(loader):
        # forward pass
        pred, err = forward_with_loss(nets, batch_data, is_train=False)
        loss_meter.update(err.data.item())
        if i % args.disp_iter == 0:
            print('[Eval] iter {}, loss: {}'.format(i, err.data.item()))

        # calculate accuracy
        acc, pix = accuracy(batch_data, pred)
        acc_meter.update(acc, pix)

        intersection, union = intersectionAndUnion(batch_data, pred,
                                                   args.num_class)
        intersection_meter.update(intersection)
        union_meter.update(union)

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(trainID2Class[i], _iou))

    print('[Cityscapes Eval Summary]:')
    print('Epoch: {}, Loss: {}, Mean IoU: {:.4}, Accuracy: {:.2f}%'
          .format(epoch, loss_meter.average(), iou.mean(), acc_meter.average() * 100))

    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['acc'].append(acc_meter.average())
    history['val']['mIoU'].append(iou.mean())

    for i, batch_data in enumerate(loader_2):
        # forward pass
        pred, err = forward_with_loss(nets, batch_data, is_train=False)
        loss_meter_2.update(err.data.item())
        if i % args.disp_iter == 0:        
            print('[Eval] iter {}, loss: {}'.format(i, err.data.item()))

        # calculate accuracy
        acc, pix = accuracy(batch_data, pred)
        acc_meter_2.update(acc, pix)

        intersection, union = intersectionAndUnion(batch_data, pred,
                                                   args.num_class)
        intersection_meter_2.update(intersection)
        union_meter_2.update(union)

    iou = intersection_meter_2.sum / (union_meter_2.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(trainID2Class[i], _iou))

    print('[BDD Eval Summary]:')
    print('Epoch: {}, Loss: {}, Mean IoU: {:.4}, Accuracy: {:.2f}%'
          .format(epoch, loss_meter_2.average(), iou.mean(), acc_meter_2.average() * 100))

    history['val_2']['epoch'].append(epoch)
    history['val_2']['err'].append(loss_meter_2.average())
    history['val_2']['acc'].append(acc_meter_2.average())
    history['val_2']['mIoU'].append(iou.mean())

    # Plot figure
    if epoch > 0:
        fig = plt.figure()
        plt.plot(np.asarray(history['train']['epoch']),
                 np.log(np.asarray(history['train']['err'])),
                 color='b', label='gta')
        plt.plot(np.asarray(history['val']['epoch']),
                 np.log(np.asarray(history['val']['err'])),
                 color='c', label='cityscapes')
        plt.plot(np.asarray(history['val_2']['epoch']),
                 np.log(np.asarray(history['val_2']['err'])),
                 color='g', label='bdd')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Log(loss)')
        fig.savefig('{}/loss.png'.format(args.ckpt), dpi=200)
        plt.close('all')

        fig = plt.figure()
        plt.plot(history['train']['epoch'], history['train']['acc'],
                 color='b', label='gta')
        plt.plot(history['val']['epoch'], history['val']['acc'],
                 color='c', label='cityscapes')
        plt.plot(history['val_2']['epoch'], history['val_2']['acc'],
                 color='g', label='bdd')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        fig.savefig('{}/accuracy.png'.format(args.ckpt), dpi=200)
        plt.close('all')


def checkpoint(nets, history, args):
    print('Saving checkpoints...')
    (net_encoder, net_decoder_1, net_decoder_2, crit) = nets
    suffix_latest = 'latest.pth'
    suffix_best_acc = 'best_acc.pth'
    suffix_best_mIoU = 'best_mIoU.pth'
    suffix_best_err = 'best_err.pth'

    if args.num_gpus > 1:
        dict_encoder = net_encoder.module.state_dict()
        dict_decoder_1 = net_decoder_1.module.state_dict()
        dict_decoder_2 = net_decoder_2.module.state_dict()

    else:
        dict_encoder = net_encoder.state_dict()
        dict_decoder_1 = net_decoder_1.state_dict()
        dict_decoder_2 = net_decoder_2.state_dict()


    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_encoder,
               '{}/encoder_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_decoder_1,
               '{}/decoder_1_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_decoder_2,
               '{}/decoder_2_{}'.format(args.ckpt, suffix_latest))

    cur_err = history['val']['err'][-1]
    cur_acc = history['val']['acc'][-1]
    cur_mIoU = history['val']['mIoU'][-1]

    if cur_acc > args.best_acc:
        # save best accuracy instead
        # args.best_err = cur_err
        args.best_acc = cur_acc
        torch.save(history,
                   '{}/history_{}'.format(args.ckpt, suffix_best_acc))
        torch.save(dict_encoder,
                   '{}/encoder_{}'.format(args.ckpt, suffix_best_acc))
        torch.save(dict_decoder_1,
                   '{}/decoder_1_{}'.format(args.ckpt, suffix_best_acc))
        torch.save(dict_decoder_2,
                   '{}/decoder_2_{}'.format(args.ckpt, suffix_best_acc))

    if cur_mIoU > args.best_mIoU:
        # save best accuracy instead
        # args.best_err = cur_err
        args.best_mIoU = cur_mIoU
        torch.save(history,
                   '{}/history_{}'.format(args.ckpt, suffix_best_mIoU))
        torch.save(dict_encoder,
                   '{}/encoder_{}'.format(args.ckpt, suffix_best_mIoU))
        torch.save(dict_decoder_1,
                   '{}/decoder_1_{}'.format(args.ckpt, suffix_best_mIoU))
        torch.save(dict_decoder_2,
                   '{}/decoder_2_{}'.format(args.ckpt, suffix_best_mIoU))

    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(history,
                   '{}/history_{}'.format(args.ckpt, suffix_best_err))
        torch.save(dict_encoder,
                   '{}/encoder_{}'.format(args.ckpt, suffix_best_err))
        torch.save(dict_decoder_1,
                   '{}/decoder_1_{}'.format(args.ckpt, suffix_best_err))
        torch.save(dict_decoder_2,
                   '{}/decoder_2_{}'.format(args.ckpt, suffix_best_err))


def create_optimizers(nets, args):
    (net_encoder, net_decoder_1, net_decoder_2, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        net_encoder.parameters(),
        lr=args.lr_encoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    optimizer_decoder_1 = torch.optim.SGD(
        net_decoder_1.parameters(),
        lr=args.lr_decoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    optimizer_decoder_2 = torch.optim.SGD(
        net_decoder_2.parameters(),
        lr=args.lr_decoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)

    return (optimizer_encoder, optimizer_decoder_1, optimizer_decoder_2)


def adjust_learning_rate(optimizers, epoch, args):
    drop_ratio = (1. * (args.num_epoch - epoch) / (args.num_epoch - epoch + 1)) \
                 ** args.lr_pow
    args.lr_encoder *= drop_ratio
    args.lr_decoder *= drop_ratio
    (optimizer_encoder, optimizer_decoder_1, optimizer_decoder_2) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = args.lr_encoder
    for param_group in optimizer_decoder_1.param_groups:
        param_group['lr'] = args.lr_decoder
    for param_group in optimizer_decoder_2.param_groups:
        param_group['lr'] = args.lr_decoder


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(weights=args.weights_encoder)
    net_decoder_1 = builder.build_decoder(weights=args.weights_decoder_1)
    net_decoder_2 = builder.build_decoder(arch='c1',weights=args.weights_decoder_2)

    if args.weighted_class:
        crit = nn.NLLLoss(ignore_index=-1, weight=args.class_weight)
    else:
        crit = nn.NLLLoss(ignore_index=-1)
    
    # Dataset and Loader
    dataset_train = GTA(root=args.root_gta, cropSize=args.imgSize, is_train=1)
    dataset_val = CityScapes('val', root=args.root_cityscapes, cropSize=args.imgSize,
                             max_sample=args.num_val, is_train=1)
    dataset_val_2 = BDD('val', root=args.root_bdd, cropSize=args.imgSize,
                        max_sample=args.num_val, is_train=1)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val_2 = torch.utils.data.DataLoader(
        dataset_val_2,
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=True)
    args.epoch_iters = int(len(dataset_train) / args.batch_size)
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # load nets into gpu
    if args.num_gpus > 1:
        net_encoder = nn.DataParallel(net_encoder,
                                      device_ids=range(args.num_gpus))
        net_decoder_1 = nn.DataParallel(net_decoder_1,
                                        device_ids=range(args.num_gpus))
        net_decoder_2 = nn.DataParallel(net_decoder_2,
                                        device_ids=range(args.num_gpus))

    nets = (net_encoder, net_decoder_1, net_decoder_2, crit)
    for net in nets:
        net.cuda()

    # Set up optimizers
    optimizers = create_optimizers(nets, args)

    # Main loop
    history = {split: {'epoch': [], 'err': [], 'acc': [], 'mIoU': []}
               for split in ('train', 'val', 'val_2')}

    # optional initial eval
    evaluate(nets, loader_val, loader_val_2, history, 0, args)
    for epoch in range(1, args.num_epoch + 1):
        train(nets, loader_train, optimizers, history, epoch, args)

        # Evaluation
        if epoch % args.eval_epoch == 0:
            evaluate(nets, loader_val, loader_val_2, history, epoch, args)

        # checkpointing
        checkpoint(nets, history, args)

        # adjust learning rate
        adjust_learning_rate(optimizers, epoch, args)

    print('Training Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='spatialrefine_xy',
                        help="a name for identifying the experiment")
    parser.add_argument('--weights_encoder',
                        default='/home/selfdriving/kchitta/Style-Randomization/pretrained/encoder_GTA.pth',
                        help="weights to initialize encoder")
    parser.add_argument('--weights_decoder_1',
                        default='/home/selfdriving/kchitta/Style-Randomization/pretrained/decoder_1_GTA.pth',
                        help="weights to initialize segmentation branch")
    parser.add_argument('--weights_decoder_2',
                        default='',
                        help="weights to initialize spatial refinement branch")

    # Path related arguments
    parser.add_argument('--root_gta',
                        default='/home/selfdriving/datasets/GTA_full')
    parser.add_argument('--root_cityscapes',
                        default='/home/selfdriving/datasets/cityscapes_full')
    parser.add_argument('--root_bdd',
                        default='/home/selfdriving/datasets/bdd100k')

    # optimization related arguments
    parser.add_argument('--num_gpus', default=3, type=int,
                        help='number of gpus to use')
    parser.add_argument('--batch_size_per_gpu', default=6, type=int,
                        help='input batch size')
    parser.add_argument('--batch_size_per_gpu_eval', default=1, type=int,
                        help='eval batch size')
    parser.add_argument('--num_epoch', default=3, type=int,
                        help='epochs to train for')

    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_encoder', default=1e-4, type=float, help='LR')
    parser.add_argument('--lr_decoder', default=1e-3, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--fix_bn', default=0, type=int,
                        help='fix bn params')

    # Data related arguments
    parser.add_argument('--num_val', default=300, type=int,
                        help='number of images to evaluate')
    parser.add_argument('--num_class', default=19, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgSize', default=720, type=int,
                        help='input crop size for training')

    # Misc arguments
    parser.add_argument('--seed', default=1337, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=100,
                        help='frequency to display')
    parser.add_argument('--eval_epoch', type=int, default=1,
                        help='frequency to evaluate')

    # Mode select
    parser.add_argument('--weighted_class', default=True, type=bool, help='set True to use weighted loss')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.batch_size_eval = args.num_gpus * args.batch_size_per_gpu_eval

    # Specify certain arguments
    if args.weighted_class:
        args.enhanced_weight = 2.0
        args.class_weight = np.ones([19], dtype=np.float32)
        enhance_class = [1, 3, 4, 5, 6, 7, 9, 12, 14, 15, 16, 17, 18]
        args.class_weight[enhance_class] = args.enhanced_weight
        args.class_weight = torch.from_numpy(args.class_weight.astype(np.float32))

    args.id += '-ngpus' + str(args.num_gpus)
    args.id += '-batchSize' + str(args.batch_size)
    args.id += '-imgSize' + str(args.imgSize)
    args.id += '-lr_encoder' + str(args.lr_encoder)
    args.id += '-lr_decoder' + str(args.lr_decoder)
    args.id += '-epoch' + str(args.num_epoch)

    print('Model ID: {}'.format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    args.best_err = 2.e10  # initialize with a big number
    args.best_acc = 0
    args.best_mIoU = 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)

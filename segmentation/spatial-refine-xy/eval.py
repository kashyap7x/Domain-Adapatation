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


def visualize(batch_data, pred, args):
    colors = loadmat('../colormap.mat')['colors']
    (imgs, segs, infos) = batch_data
    for j in range(len(infos)):
        # get/recover image
        # img = imread(os.path.join(args.root_img, infos[j]))
        img = imgs[j,:3,:,:].clone()
        for t, m, s in zip(img,
                           [0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)

        # segmentation
        lab = segs[j].numpy()
        lab_color = colorEncode(lab, colors)

        # prediction
        pred_ = np.argmax(pred.data.cpu()[j].numpy(), axis=0)
        pred_color = colorEncode(pred_, colors)

        # aggregate images and save
        im_vis = np.concatenate((img, lab_color, pred_color),
                                axis=1).astype(np.uint8)
        imsave(os.path.join(args.vis,
                            infos[j].replace('/', '_')), im_vis)


def evaluate(nets, loader, loader_2, history, epoch, args, isVis=True):
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
        print('[Eval] iter {}, loss: {}'.format(i, err.data.item()))

        # calculate accuracy
        acc, pix = accuracy(batch_data, pred)
        acc_meter.update(acc, pix)

        intersection, union = intersectionAndUnion(batch_data, pred,
                                                   args.num_class)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # visualization
        if isVis:
            visualize(batch_data, pred, args)

    for i, batch_data in enumerate(loader_2):
        # forward pass
        pred, err = forward_with_loss(nets, batch_data, is_train=False)
        loss_meter_2.update(err.data.item())
        print('[Eval] iter {}, loss: {}'.format(i, err.data.item()))

        # calculate accuracy
        acc, pix = accuracy(batch_data, pred)
        acc_meter_2.update(acc, pix)

        intersection, union = intersectionAndUnion(batch_data, pred,
                                                   args.num_class)
        intersection_meter_2.update(intersection)
        union_meter_2.update(union)

        # visualization
        if isVis:
            visualize(batch_data, pred, args)

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
    dataset_val = CityScapes('val', root=args.root_cityscapes, max_sample=args.num_val, is_train=0)
    dataset_val_2 = BDD('val', root=args.root_bdd, max_sample=args.num_val, is_train=0)

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

    # Main loop
    history = {split: {'epoch': [], 'err': [], 'acc': [], 'mIoU': []}
               for split in ('train', 'val', 'val_2')}

    # eval
    evaluate(nets, loader_val, loader_val_2, history, 0, args)
    print('Evaluation Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the experiment")
    parser.add_argument('--suffix', default='_latest.pth',
                        help="which snapshot to load")

    # Path related arguments
    parser.add_argument('--root_cityscapes',
                        default='/home/selfdriving/datasets/cityscapes_full')
    parser.add_argument('--root_bdd',
                        default='/home/selfdriving/datasets/bdd100k')

    # optimization related arguments
    parser.add_argument('--num_gpus', default=3, type=int,
                        help='number of gpus to use')
    parser.add_argument('--batch_size_per_gpu_eval', default=1, type=int,
                        help='eval batch size')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evaluate')
    parser.add_argument('--num_class', default=19, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers')

    # Misc arguments
    parser.add_argument('--seed', default=1337, type=int, help='manual seed')
    # Specify visualization directory
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--vis', default='./vis',
                        help='folder to output visualization during training')
    
    parser.add_argument('--weighted_class', default=True, type=bool, help='set True to use weighted loss')
        
    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    if args.weighted_class:
        args.enhanced_weight = 2.0
        args.class_weight = np.ones([19], dtype=np.float32)
        enhance_class = [1, 3, 4, 5, 6, 7, 9, 12, 14, 15, 16, 17, 18]
        args.class_weight[enhance_class] = args.enhanced_weight
        args.class_weight = torch.from_numpy(args.class_weight.astype(np.float32))
        
    args.batch_size_eval = args.num_gpus * args.batch_size_per_gpu_eval

    model_id = 'spatialrefine_xy-ngpus3-batchSize18-imgSize720-lr_encoder0.0001-lr_decoder0.001-epoch3'

    print(args)

    args.weights_encoder = os.path.join(args.ckpt, model_id,
                                        'encoder' + args.suffix)
    args.weights_decoder_1 = os.path.join(args.ckpt, model_id,
                                          'decoder_1' + args.suffix)
    args.weights_decoder_2 = os.path.join(args.ckpt, model_id,
                                          'decoder_2' + args.suffix)
                                          
    args.vis = os.path.join(args.vis, args.id, model_id)
    
    if not os.path.exists(args.vis):
        os.makedirs(args.vis)
        
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)

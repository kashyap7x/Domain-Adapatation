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
from dataset import GTA, CityScapes, BDD
from models import ModelBuilder, Whitening, AdditiveNoise, WhitenedNoise
from utils import AverageMeter, colorEncode, accuracy, make_variable, intersectionAndUnion

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

trainID2Class = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle'
}


def forward_with_loss(nets, batch_data, is_train=True):
    (net_encoder, net_decoder_1, net_decoder_2, style, crit1, crit2) = nets
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
    out = style(net_encoder(input_img))
    pred_featuremap_1 = net_decoder_1(out)
    pred_featuremap_2 = net_decoder_2(out)

    err = crit1(pred_featuremap_1, label_seg) + args.beta * crit2(pred_featuremap_2, input_img)

    return pred_featuremap_1, pred_featuremap_2, err


def visualize_recon(batch_data, recons, args):
    (imgs, segs, infos) = batch_data

    for j in range(len(infos)):
        img = imgs[j].clone()
        for t, m, s in zip(img,
                           [0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)

        recon = recons[j].clone()
        for t, m, s in zip(recon,
                           [0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        recon = (recon.cpu().detach().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)

        im_vis = np.concatenate((img, recon),
                                axis=1).astype(np.uint8)
        imsave(os.path.join(args.vis_recon,
                            infos[j].replace('/', '_')), im_vis)


def visualize(batch_data, pred, args):
    colors = loadmat('../colormap.mat')['colors']
    (imgs, segs, infos) = batch_data
    for j in range(len(infos)):
        # get/recover image
        # img = imread(os.path.join(args.root_img, infos[j]))
        img = imgs[j].clone()
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
        torch.cuda.empty_cache()
        pred, recon, err = forward_with_loss(nets, batch_data, is_train=False)
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
            visualize_recon(batch_data, recon, args)


    for i, batch_data in enumerate(loader_2):
        # forward pass
        torch.cuda.empty_cache()
        pred, recon, err = forward_with_loss(nets, batch_data, is_train=False)
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
            visualize_recon(batch_data, recon, args)
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
    net_decoder_2 = builder.build_decoder(arch='c1', num_class=3, use_softmax=False,
                                          weights=args.weights_decoder_2)

    if args.weighted_class:
        crit1 = nn.NLLLoss(ignore_index=-1, weight=args.class_weight)
    else:
        crit1 = nn.NLLLoss(ignore_index=-1)
    crit2 = nn.MSELoss()
    
    # Style application module
    style = WhitenedNoise(10)
    
    # Dataset and Loader
    dataset_train = GTA(root=args.root_gta, cropSize=args.imgSize, is_train=1)
    dataset_val = CityScapes('val', root=args.root_cityscapes, cropSize=args.imgSize,
                             max_sample=args.num_val, is_train=0)
    dataset_val_2 = BDD('val', root=args.root_bdd, cropSize=args.imgSize,
                        max_sample=args.num_val, is_train=0)

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

    nets = (net_encoder, net_decoder_1, net_decoder_2, style, crit1, crit2)
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
    parser.add_argument('--id', default='WhitenedNoise10',
                        help="a name for identifying the experiment")
    parser.add_argument('--suffix', default='_best_mIoU.pth',
                        help="which snapshot to load")

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
    parser.add_argument('--ckpt', default='./ckpt/ResNet',
                        help='folder to output checkpoints')
    parser.add_argument('--vis', default='./vis',
                        help='folder to output visualization during training')
    parser.add_argument('--vis_recon', default='./vis_recon',
                        help='folder to output visualization of reconstruction during training')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    args.batch_size_eval = args.batch_size_per_gpu_eval

    model_id = 'baseline-ngpus3-batchSize18-imgSize720-lr_encoder0.001-lr_decoder0.01-epoch20-decay0.0001-beta0.1-weighted2.0[1, 3, 4, 5, 6, 7, 9, 12, 14, 15, 16, 17, 18]'

    print(args)

    args.weights_encoder = os.path.join(args.ckpt, model_id,
                                        'encoder' + args.suffix)
    args.weights_decoder_1 = os.path.join(args.ckpt, model_id,
                                          'decoder_1' + args.suffix)
    args.weights_decoder_2 = os.path.join(args.ckpt, model_id,
                                          'decoder_2' + args.suffix)
    
    args.vis = os.path.join(args.vis, args.id, model_id)
    args.vis_recon = os.path.join(args.vis_recon, args.id, model_id)
    
    if not os.path.exists(args.vis):
        os.makedirs(args.vis)
    if not os.path.exists(args.vis_recon):
        os.makedirs(args.vis_recon)
        
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)

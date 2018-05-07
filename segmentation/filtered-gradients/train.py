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
from dataset import GTA, CityScapes
from models import ModelBuilder
from utils import AverageMeter, colorEncode, accuracy, randomSampler, similiarityPenalty, make_variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def forward_with_loss(nets, batch_data, args, is_train=True, is_adapt=False):
    (net_encoder, net_decoder_1, net_decoder_2, net_syn, crit) = nets
    (imgs, segs, infos) = batch_data

    # feed input data
    input_img = Variable(imgs, volatile=not is_train)
    label_seg = Variable(segs, volatile=not is_train)
    input_img = input_img.cuda()
    label_seg = label_seg.cuda()

    # forward
    pred_featuremap_1 = net_decoder_1(net_encoder(input_img))
    pred_featuremap_2 = net_decoder_2(net_encoder(input_img))
    pred_featuremap_syn = net_syn(pred_featuremap_1, pred_featuremap_2)

    weights1 = net_decoder_1.module.get_weights()
    weights2 = net_decoder_2.module.get_weights()

    if is_adapt:
        _, pred_1 = torch.max(pred_featuremap_1, 1)
        _, pred_2 = torch.max(pred_featuremap_2, 1)
        _, pred_syn = torch.max(pred_featuremap_syn, 1)

        # reshape the feature map as class_num * (batch_size * h * w)
        pred_1 = pred_1.view(1, -1)
        pred_2 = pred_2.view(1, -1)
        pred_syn = pred_syn.view(1, -1)

        adapt_idx = (torch.eq(pred_1, pred_2)).squeeze()

        # all the rest are ignored indexes
        ignored_idx = (adapt_idx == 0).nonzero().squeeze()
        if len(ignored_idx.size()) > 0:
            pred_syn[..., ignored_idx] = -1

        # reshape back to use NLLLoss2d
        # TODO: not sure whether this view can reproduce the same one
        pred_syn = pred_syn.view(pred_featuremap_syn.size(0), pred_featuremap_syn.size(2), pred_featuremap_syn.size(3))

        if len(adapt_idx.size()) > 0:
            err_1 = crit(pred_featuremap_1, pred_syn)
            err_2 = crit(pred_featuremap_2, pred_syn)
            # err_syn = crit(pred_featuremap_syn, pred_syn)
            err_syn = 0
        else:
            err_1 = 0
            err_2 = 0
            err_syn = 0
    else:
        err_1 = crit(pred_featuremap_1, label_seg)
        err_2 = crit(pred_featuremap_2, label_seg)
        err_syn = crit(pred_featuremap_syn, label_seg)

    err_sim = similiarityPenalty(weights1.squeeze(), weights2.squeeze())

    err = err_1 + err_2 + args.alpha * err_sim + args.beta * err_syn

    return pred_featuremap_syn, err


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


# train one epoch
def train(nets, loader, loader_adapt, optimizers, history, epoch, args):
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
    #for i, batch_data in enumerate(loader):
    for i in range(args.epoch_iters):
        batch_data, is_adapt = randomSampler(args.ratio_source_init, args.ratio_source_final, \
                                             args.ratio_source_final_epoch, epoch, loader, loader_adapt)
        data_time.update(time.time() - tic)
        for net in nets:
            net.zero_grad()

        # forward pass
        pred, err = forward_with_loss(nets, batch_data, args, is_train=True, is_adapt=is_adapt)

        # Backward
        err.backward()

        for net in nets:
            nn.utils.clip_grad_norm(net.parameters(),1)
            #for param in net.parameters():
            #    print(param.grad.data.shape, param.grad.data.sum())

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
                          acc*100, err.data[0]))

            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.data[0])
            history['train']['acc'].append(acc)


def evaluate(nets, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # switch to eval mode
    for net in nets:
        net.eval()

    for i, batch_data in enumerate(loader):
        # forward pass
        torch.cuda.empty_cache()
        pred, err = forward_with_loss(nets, batch_data, args, is_train=False)
        loss_meter.update(err.data[0])
        print('[Eval] iter {}, loss: {}'.format(i, err.data[0]))

        # calculate accuracy
        acc, pix = accuracy(batch_data, pred)
        acc_meter.update(acc, pix)

        # visualization
        visualize(batch_data, pred, args)

    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['acc'].append(acc_meter.average())
    print('[Eval Summary] Epoch: {}, Loss: {}, Accurarcy: {:4.2f}%'
          .format(epoch, loss_meter.average(), acc_meter.average()*100))

    # Plot figure
    if epoch > 0:
        print('Plotting loss figure...')
        fig = plt.figure()
        plt.plot(np.asarray(history['train']['epoch']),
                 np.log(np.asarray(history['train']['err'])),
                 color='b', label='training')
        plt.plot(np.asarray(history['val']['epoch']),
                 np.log(np.asarray(history['val']['err'])),
                 color='c', label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Log(loss)')
        fig.savefig('{}/loss.png'.format(args.ckpt), dpi=200)
        plt.close('all')

        fig = plt.figure()
        plt.plot(history['train']['epoch'], history['train']['acc'],
                 color='b', label='training')
        plt.plot(history['val']['epoch'], history['val']['acc'],
                 color='c', label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        fig.savefig('{}/accuracy.png'.format(args.ckpt), dpi=200)
        plt.close('all')


def checkpoint(nets, history, args):
    print('Saving checkpoints...')
    (net_encoder, net_decoder_1, net_decoder_2, net_syn, crit) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    if args.num_gpus > 1:
        dict_encoder = net_encoder.module.state_dict()
        dict_decoder_1 = net_decoder_1.module.state_dict()
        dict_decoder_2 = net_decoder_2.module.state_dict()
        dict_syn = net_syn.module.state_dict()
    else:
        dict_encoder = net_encoder.state_dict()
        dict_decoder_1 = net_decoder_1.state_dict()
        dict_decoder_2 = net_decoder_2.state_dict()
        dict_syn = net_syn.state_dict()

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_encoder,
               '{}/encoder_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_decoder_1,
               '{}/decoder_1_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_decoder_2,
               '{}/decoder_2_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_syn,
               '{}/syn_{}'.format(args.ckpt, suffix_latest))

    cur_err = history['val']['err'][-1]
    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(history,
                   '{}/history_{}'.format(args.ckpt, suffix_best))
        torch.save(dict_encoder,
                   '{}/encoder_{}'.format(args.ckpt, suffix_best))
        torch.save(dict_decoder_1,
                   '{}/decoder_1_{}'.format(args.ckpt, suffix_best))
        torch.save(dict_decoder_2,
                   '{}/decoder_2_{}'.format(args.ckpt, suffix_best))
        torch.save(dict_syn,
                   '{}/syn_{}'.format(args.ckpt, suffix_best))


def create_optimizers(nets, args):
    (net_encoder, net_decoder_1, net_decoder_2, net_syn, crit) = nets
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
    optimizer_syn = torch.optim.SGD(
        net_syn.parameters(),
        lr=args.lr_decoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    return (optimizer_encoder, optimizer_decoder_1, optimizer_decoder_2, optimizer_syn)


def adjust_learning_rate(optimizers, epoch, args):
    drop_ratio = (1. * (args.num_epoch-epoch) / (args.num_epoch-epoch+1)) \
                 ** args.lr_pow
    args.lr_encoder *= drop_ratio
    args.lr_decoder *= drop_ratio
    (optimizer_encoder, optimizer_decoder_1, optimizer_decoder_2, optimizer_syn) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = args.lr_encoder
    for param_group in optimizer_decoder_1.param_groups:
        param_group['lr'] = args.lr_decoder
    for param_group in optimizer_decoder_2.param_groups:
        param_group['lr'] = args.lr_decoder
    for param_group in optimizer_syn.param_groups:
        param_group['lr'] = args.lr_decoder


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(arch=args.arch_encoder,
                                        fc_dim=args.fc_dim,
                                        weights=args.weights_encoder)
    net_decoder_1 = builder.build_decoder(arch=args.arch_decoder,
                                        fc_dim=args.fc_dim,
                                        num_class=args.num_class,
                                        weights=args.weights_decoder)
    net_decoder_2 = builder.build_decoder(arch=args.arch_decoder,
                                          fc_dim=args.fc_dim,
                                          num_class=args.num_class,
                                          weights=args.weights_decoder)
    net_syn = builder.build_syn()

    crit = nn.NLLLoss2d(ignore_index=-1)

    # Dataset and Loader
    dataset_train = GTA(cropSize=args.imgSize, root=args.root_playing)
    dataset_adapt =  CityScapes('train', root=args.root_cityscapes, cropSize=args.imgSize, is_train=1)
    dataset_val = CityScapes('val', root=args.root_cityscapes, cropSize=args.imgSize, max_sample=args.num_val, is_train=0)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_adapt = torch.utils.data.DataLoader(
        dataset_adapt,
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
        net_syn = nn.DataParallel(net_syn,
                                        device_ids=range(args.num_gpus))

    nets = (net_encoder, net_decoder_1, net_decoder_2, net_syn, crit)
    for net in nets:
        net.cuda()

    # Set up optimizers
    optimizers = create_optimizers(nets, args)

    # Main loop
    history = {split: {'epoch': [], 'err': [], 'acc': []}
               for split in ('train', 'val')}

    # optional initial eval
    # evaluate(nets, loader_val, history, 0, args)
    for epoch in range(1, args.num_epoch + 1):
        train(nets, loader_train, loader_adapt, optimizers, history, epoch, args)
        
        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(nets, loader_val, history, epoch, args)

        # checkpointing
        checkpoint(nets, history, args)

        # adjust learning rate
        adjust_learning_rate(optimizers, epoch, args)
        
    print('Training Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='adapt',
                        help="a name for identifying the model")
    parser.add_argument('--arch_encoder', default='resnet34_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='psp_bilinear',
                        help="architecture of net_decoder")
    parser.add_argument('--weights_encoder', default='/home/selfdriving/kchitta/Domain-Adapatation/segmentation/pretrained/encoder_best.pth',
                        help="weights to finetune net_encoder")
    parser.add_argument('--weights_decoder', default='/home/selfdriving/kchitta/Domain-Adapatation/segmentation/pretrained/decoder_best.pth',
                        help="weights to finetune net_decoder")
    parser.add_argument('--fc_dim', default=512, type=int,
                        help='number of features between encoder and decoder')

    # Path related arguments
    parser.add_argument('--root_cityscapes',
                        default='/home/selfdriving/datasets/cityscapes_full')
    parser.add_argument('--root_playing',
                        default='/home/selfdriving/datasets/GTA_full')

    # optimization related arguments
    parser.add_argument('--num_gpus', default=3, type=int,
                        help='number of gpus to use')
    parser.add_argument('--batch_size_per_gpu', default=4, type=int,
                        help='input batch size')
    parser.add_argument('--batch_size_per_gpu_eval', default=3, type=int,
                        help='eval batch size')
    parser.add_argument('--num_epoch', default=20, type=int,
                        help='epochs to train for')
    parser.add_argument('--ratio_source_init', default=0.9, type=float,
                        help='initial sampling ratio for source domain')
    parser.add_argument('--ratio_source_final', default=0.1, type=float,
                        help='final sampling ratio for source domain')
    parser.add_argument('--ratio_source_final_epoch', default=10, type=int,
                        help='epoch beyond which to maintain final ratio')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_encoder', default=1e-3, type=float, help='LR')
    parser.add_argument('--lr_decoder', default=1e-2, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--alpha', default=0.01, type=float,
                        help='weight of similarity loss')
    parser.add_argument('--beta', default=1, type=float,
                        help='weight of synthetic loss')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--fix_bn', default=0, type=int,
                        help='fix bn params')

    # Data related arguments
    parser.add_argument('--num_val', default=90, type=int,
                        help='number of images to evaluate')
    parser.add_argument('--num_class', default=19, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=1, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgSize', default=600, type=int,
                        help='input image size')
    parser.add_argument('--segSize', default=600, type=int,
                        help='output image size')

    # Misc arguments
    parser.add_argument('--seed', default=1337, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--vis', default='./vis',
                        help='folder to output visualization during training')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')
    parser.add_argument('--eval_epoch', type=int, default=1,
                        help='frequency to evaluate')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.batch_size_eval = args.num_gpus * args.batch_size_per_gpu_eval

    if args.num_val < args.batch_size:
        args.num_val = args.batch_size

    args.id += '-' + str(args.arch_encoder)
    args.id += '-' + str(args.arch_decoder)
    args.id += '-ngpus' + str(args.num_gpus)
    args.id += '-batchSize' + str(args.batch_size)
    args.id += '-imgSize' + str(args.imgSize)
    args.id += '-lr_encoder' + str(args.lr_encoder)
    args.id += '-lr_decoder' + str(args.lr_decoder)
    args.id += '-epoch' + str(args.num_epoch)
    args.id += '-ratio' + str(args.ratio_source_init) + '-' + str(args.ratio_source_final) + '-' +str(args.ratio_source_final_epoch)
    args.id += '-alpha' + str(args.alpha)
    args.id += '-beta' + str(args.beta)
    args.id += '-decay' + str(args.weight_decay)
    print('Model ID: {}'.format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.vis, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)
    if not os.path.exists(args.vis):
        os.makedirs(args.vis)

    args.best_err = 2.e10   # initialize with a big number

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)

# System libs
import os
import datetime
import argparse
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.io import loadmat
from scipy.misc import imsave
from scipy.ndimage import zoom
# Our libs
from dataset import CityScapes
from models import ModelBuilder
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion

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


# forward func for evaluation
def forward_multiscale(nets, batch_data, args):
    (net_encoder, net_decoder_1, net_decoder_2, net_syn, crit) = nets
    (imgs, segs, infos) = batch_data

    segSize = (segs.size(1), segs.size(2))
    pred = torch.zeros(imgs.size(0), args.num_class, segs.size(1), segs.size(2))
    pred = Variable(pred, volatile=True).cuda()

    for scale in args.scales:
        imgs_scale = zoom(imgs.numpy(),
                          (1., 1., scale, scale),
                          order=1,
                          prefilter=False,
                          mode='nearest')

        # feed input data
        input_img = Variable(torch.from_numpy(imgs_scale),
                             volatile=True).cuda()

        # forward
        pred_featuremap_1 = net_decoder_1(net_encoder(input_img))
        pred_featuremap_2 = net_decoder_2(net_encoder(input_img))
        pred_scale = net_syn(pred_featuremap_1, pred_featuremap_2)


        # average the probability
        pred = pred + pred_scale / len(args.scales)


    # pred = torch.log(pred)

    label_seg = Variable(segs, volatile=True).cuda()
    err = crit(pred, label_seg)
    return pred, err


def visualize_result(batch_data, pred, args):
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
        imsave(os.path.join(args.result,
                            infos[j].replace('/', '_')), im_vis)


def evaluate(nets, loader, args):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    # switch to eval mode
    for net in nets:
        net.eval()

    for i, batch_data in enumerate(loader):
        # forward pass
        pred, err = forward_multiscale(nets, batch_data, args)
        loss_meter.update(err.data[0])
        # calculate accuracy
        acc, pix = accuracy(batch_data, pred)
        intersection, union = intersectionAndUnion(batch_data, pred,
                                                   args.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        print('[{}] iter {}, loss: {}, accuracy: {}'
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      i, err.data[0], acc))

        # visualization
        if args.visualize:
            visualize_result(batch_data, pred, args)

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(trainID2Class[i], _iou))

    print('[Eval Summary]:')
    print('Loss: {}, Mean IoU: {:.4}, Accurarcy: {:.2f}%'
          .format(loss_meter.average(), iou.mean(), acc_meter.average()*100))


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(arch=args.arch_encoder,
                                        fc_dim=args.fc_dim,
                                        weights=args.weights_encoder)
    net_decoder_1 = builder.build_decoder(arch=args.arch_decoder,
                                          fc_dim=args.fc_dim,
                                          num_class=args.num_class,
                                          weights=args.weights_decoder_1)
    net_decoder_2 = builder.build_decoder(arch=args.arch_decoder,
                                          fc_dim=args.fc_dim,
                                          num_class=args.num_class,
                                          weights=args.weights_decoder_2)
    net_syn = builder.build_syn(weights=args.weights_syn)

    crit = nn.NLLLoss2d(ignore_index=-1)

    # Dataset and Loader
    dataset_val = CityScapes('val', root=args.root_cityscapes, max_sample=args.num_val, is_train=0)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    nets = (net_encoder, net_decoder_1, net_decoder_2, net_syn, crit)
    for net in nets:
        net.cuda()

    # Main loop
    evaluate(nets, loader_val, args)

    print('Evaluation Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    # parser.add_argument('--id', required=True,
    #                     help="a name for identifying the model to load")
    parser.add_argument('--id', help="a name for identifying the model to load")
    parser.add_argument('--suffix', default='_best.pth',
                        help="which snapshot to load")
    parser.add_argument('--arch_encoder', default='resnet34_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='psp_bilinear',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=512, type=int,
                        help='number of features between encoder and decoder')

    # Path related arguments
    parser.add_argument('--list_val',
                        default='./data/ADE20K_object150_val.txt')
    parser.add_argument('--root_img',
                        default='./data/ADEChallengeData2016/images')
    parser.add_argument('--root_seg',
                        default='./data/ADEChallengeData2016/annotations')
    parser.add_argument('--root_cityscapes',
                        default='/home/selfdriving/datasets/cityscapes_full')
    parser.add_argument('--root_playing',
                        default='/home/selfdriving/datasets/GTA_full')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=19, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize')
    parser.add_argument('--imgSize', default=-1, type=int,
                        help='input image size, -1 = keep original')
    parser.add_argument('--segSize', default=-1, type=int,
                        help='output image size, -1 = keep original')

    # Misc arguments
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--visualize', default=0,
                        help='output visualization?')
    parser.add_argument('--result', default='./result',
                        help='folder to output visualization results')

    args = parser.parse_args()

    args.id = 'adapt-resnet34_dilated8-psp_bilinear-ngpus3-batchSize12-imgSize600-lr_encoder0.001-lr_decoder0.01-epoch10-ratio0.8-0.4-5-alpha0.01-beta1-decay0.0001'

    print(args)

    # scales for evaluation
    args.scales = (1, )
    # args.scales = (0.5, 0.75, 1, 1.25, 1.5)

    # absolute paths of model weights
    args.weights_encoder = os.path.join(args.ckpt, args.id,
                                        'encoder' + args.suffix)
    args.weights_decoder_1 = os.path.join(args.ckpt, args.id,
                                        'decoder_1' + args.suffix)
    args.weights_decoder_2 = os.path.join(args.ckpt, args.id,
                                          'decoder_2' + args.suffix)
    args.weights_syn = os.path.join(args.ckpt, args.id,
                                          'syn' + args.suffix)

    args.result = os.path.join(args.result, args.id)
    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)

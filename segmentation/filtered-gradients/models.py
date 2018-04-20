import torch
import torch.nn as nn
import torchvision
import resnet
# from utils import gather

class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, arch='resnet50', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        if arch == 'resnet34':
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34_dilated8':
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet34_dilated16':
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet34_quad':
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetQuad(orig_resnet)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50_dilated8':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet50_dilated16':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage))
        return net_encoder

    def build_decoder(self, arch='c1_bilinear', fc_dim=512, num_class=19,
                      upSampleScale=8, weights='', use_softmax=False):
        if arch == 'c1_bilinear':
            net_decoder = C1Bilinear(num_class=num_class,
                                     fc_dim=fc_dim,
                                     segSize=segSize,
                                     use_softmax=use_softmax)
        elif arch == 'psp_bilinear':
            net_decoder = PSPBilinear(num_class=num_class,
                                      fc_dim=fc_dim,
                                      upSampleScale=upSampleScale,
                                      use_softmax=use_softmax)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            pretrained_dict = torch.load(weights, map_location=lambda storage, loc: storage)
            net_decoder.load_state_dict(pretrained_dict, strict=False)
        return net_decoder

    def build_syn(self, num_class=19, use_softmax=False):
        net_syn = SynModel(num_class=num_class, use_softmax=use_softmax)
        return net_syn


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, dropout2d=False):
        super(ResnetDilated, self).__init__()
        self.dropout2d = dropout2d
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        if self.dropout2d:
            self.dropout = nn.Dropout2d(0.5)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.dropout2d:
            x = self.dropout(x)
        return x


# last conv, bilinear upsample
class C1Bilinear(nn.Module):
    def __init__(self, num_class=19, fc_dim=512, segSize=512,
                 use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.segSize = segSize
        self.use_softmax = use_softmax

        # last conv
        self.conv_last = nn.Conv2d(fc_dim, num_class, 1, 1, 0, bias=False)

    def forward(self, x, segSize=None):
        if segSize is None:
            segSize = (self.segSize, self.segSize)
        elif isinstance(segSize, int):
            segSize = (segSize, segSize)

        x = self.conv_last(x)

        if not (x.size(2) == segSize[0] and x.size(3) == segSize[1]):
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')

        if self.use_softmax:
            x = nn.functional.softmax(x)
        else:
            x = nn.functional.log_softmax(x)
        return x

    def get_weights(self):
        return self.conv_last.weight


# pyramid pooling, bilinear upsample
class PSPBilinear(nn.Module):
    def __init__(self, num_class=19, fc_dim=512, upSampleScale=8,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PSPBilinear, self).__init__()
        self.upSampleScale = upSampleScale
        self.use_softmax = use_softmax

        self.psp = []
        for scale in pool_scales:
            self.psp.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.psp = nn.ModuleList(self.psp)

        self.conv_final = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, x, upSampleScale=None):
        if upSampleScale is None:
            upSampleScale = self.upSampleScale
        input_size = x.size()
        psp_out = [x]
        for pool_scale in self.psp:
            psp_out.append(nn.functional.upsample(
                pool_scale(x),
                (input_size[2], input_size[3]),
                mode='bilinear'))
        psp_out = torch.cat(psp_out, 1)

        x = self.conv_final(psp_out)
        x = nn.functional.upsample(x, size=(input_size[2]*upSampleScale, input_size[3]*upSampleScale), mode='bilinear')

        if self.use_softmax:
            x = nn.functional.softmax(x)
        else:
            x = nn.functional.log_softmax(x)
        return x

    def get_weights(self):
        return self.conv_final[-1].weight

class SynModel(nn.Module):
    def __init__(self, num_class=19, use_softmax=False):
        super(SynModel, self).__init__()
        self.num_class = num_class
        self.use_softmax = use_softmax
        self.conv = nn.Conv2d(in_channels=2 * num_class, out_channels=num_class, kernel_size=1)

    def forward(self, pred_1, pred_2):
        pred = torch.cat([pred_1, pred_2], dim=1)
        pred = self.conv(pred)
        if self.use_softmax:
            pred = nn.functional.softmax(pred)
        else:
            pred = nn.functional.log_softmax(pred)
        return pred
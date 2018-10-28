import torch
import torch.nn as nn
import resnet
from lib.nn import SynchronizedBatchNorm2d


class ModelBuilder():
    def build_encoder(self, arch='resnet', weights=''):
        if arch == 'resnet':
            net_encoder = ResnetEncoder(resnet.resnet18())
        elif arch == 'vgg':
            net_encoder = VGGEncoder()
        if len(weights) > 0:
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage))
        return net_encoder

    def build_decoder(self, arch='ppm', num_class=19, use_softmax=True, weights=''):
        if arch == 'c1':
            net_decoder = C1Decoder(num_class, use_softmax)
        elif arch == 'ppm':
            net_decoder = PPMDecoder(num_class, use_softmax)
        elif arch == 'vgg':
            net_decoder = VGGDecoder(num_class, use_softmax)
        if len(weights) > 0:
            pretrained_dict = torch.load(weights, map_location=lambda storage, loc: storage)
            net_decoder.load_state_dict(pretrained_dict, strict=False)
        return net_decoder


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()

        # 224 x 224
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)

        self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu1_1 = nn.ReLU(inplace=True)
        # 224 x 224

        self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu1_2 = nn.ReLU(inplace=True)
        # 224 x 224
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 112 x 112

        self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu2_1 = nn.ReLU(inplace=True)
        # 112 x 112

        self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu2_2 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 56 x 56

        self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu3_1 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_2 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_3 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_4 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 28 x 28

        self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu4_1 = nn.ReLU(inplace=True)
        # 28 x 28

    def forward(self, x):
        out = self.conv0(x)

        out = self.pad1_1(out)
        out = self.conv1_1(out)
        out = self.relu1_1(out)

        out = self.pad1_2(out)
        out = self.conv1_2(out)
        pool1 = self.relu1_2(out)

        out, pool1_idx = self.maxpool1(pool1)

        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)

        out = self.pad2_2(out)
        out = self.conv2_2(out)
        pool2 = self.relu2_2(out)

        out, pool2_idx = self.maxpool2(pool2)

        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)

        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)

        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)

        out = self.pad3_4(out)
        out = self.conv3_4(out)
        pool3 = self.relu3_4(out)
        out, pool3_idx = self.maxpool3(pool3)

        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)

        return out, pool1_idx, pool2_idx, pool3_idx


class VGGDecoder(nn.Module):
    def __init__(self, num_class, use_softmax):
        super(VGGDecoder, self).__init__()
        self.num_class = num_class
        self.use_softmax = use_softmax

        self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu4_1 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 56 x 56

        self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_4 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_3 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_2 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu3_1 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 112 x 112

        self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu2_2 = nn.ReLU(inplace=True)
        # 112 x 112

        self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu2_1 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 224 x 224

        self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu1_2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_out = nn.Conv2d(64, self.num_class, 3, 1, 0)

    def forward(self, x, pool1_idx=None, pool2_idx=None, pool3_idx=None):
        out = x

        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)
        out = self.unpool3(out, pool3_idx)

        out = self.pad3_4(out)
        out = self.conv3_4(out)
        out = self.relu3_4(out)

        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)

        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)

        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)
        out = self.unpool2(out, pool2_idx)

        out = self.pad2_2(out)
        out = self.conv2_2(out)
        out = self.relu2_2(out)

        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)
        out = self.unpool1(out, pool1_idx)

        out = self.pad1_2(out)
        out = self.conv1_2(out)
        out = self.relu1_2(out)

        out = self.pad1_1(out)
        out = self.conv_out(out)

        if self.use_softmax:
            out = nn.functional.log_softmax(out, dim=1)

        return out


class ResnetEncoder(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, dropout2d=False):
        super(ResnetEncoder, self).__init__()
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


class C1Decoder(nn.Module):
    def __init__(self, num_class=19, use_softmax=True):
        super(C1Decoder, self).__init__()
        self.use_softmax = use_softmax

        # last conv
        self.conv_last = nn.Conv2d(512, num_class, 1, 1, 0)

    def forward(self, x):
        input_size = x.size()
        x = self.conv_last(x)
        x = nn.functional.upsample(x, size=(input_size[2]*8, input_size[3]*8), mode='bilinear')

        if self.use_softmax:
            x = nn.functional.log_softmax(x, dim=1)

        return x


class PPMDecoder(nn.Module):
    def __init__(self, num_class=19, use_softmax=True, pool_scales=(1, 2, 3, 6)):
        super(PPMDecoder, self).__init__()
        self.use_softmax = use_softmax

        self.psp = []
        for scale in pool_scales:
            self.psp.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(512, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.psp = nn.ModuleList(self.psp)

        self.conv_final = nn.Sequential(
            nn.Conv2d(512+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()
        psp_out = [x]
        for pool_scale in self.psp:
            psp_out.append(nn.functional.upsample(
                pool_scale(x),
                (input_size[2], input_size[3]),
                mode='bilinear'))
        psp_out = torch.cat(psp_out, 1)

        x = self.conv_final(psp_out)
        x = nn.functional.upsample(x, size=(input_size[2]*8, input_size[3]*8), mode='bilinear')

        if self.use_softmax:
            x = nn.functional.log_softmax(x, dim=1)

        return x


class Whitening(nn.Module):
    def __init__(self):
        super(Whitening, self).__init__()
        
    def forward(self, cont):
        cont_c, cont_h, cont_w = cont.size(0), cont.size(1), cont.size(2)
        cont_feat = cont.view(cont_c, -1).clone()
        
        cFSize = cont_feat.size()
        c_mean = torch.mean(cont_feat, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cont_feat)
        cont_feat = cont_feat - c_mean
        
        iden = torch.eye(cFSize[0])  # .double()
        iden = iden.cuda()
        
        contentConv = torch.mm(cont_feat, cont_feat.t()).div(cFSize[1] - 1) + iden
        c_u, c_e, c_v = torch.svd(contentConv, some=False)
        
        k_c = cFSize[0]
        for i in range(cFSize[0] - 1, -1, -1):
            if c_e[i] >= 0.00001:
                k_c = i + 1
                break
        
        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cont_feat)
        
        whiten_cF = whiten_cF.view_as(cont)
        
        return whiten_cF
    

class AdditiveNoise(nn.Module):
    def __init__(self, var):
        super(AdditiveNoise, self).__init__()
        self.var = var
        
    def forward(self, cont):
        noise = torch.randn(cont.size())*self.var
        noise = noise.cuda()
        cont_feat = cont + noise
        
        return cont_feat
    

class WhitenedNoise(nn.Module):
    def __init__(self, var):
        super(WhitenedNoise, self).__init__()
        self.var = var
        
    def forward(self, cont):
        cont_c, cont_h, cont_w = cont.size(0), cont.size(1), cont.size(2)
        cont_feat = cont.view(cont_c, -1).clone()
        
        cFSize = cont_feat.size()
        c_mean = torch.mean(cont_feat, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cont_feat)
        cont_feat = cont_feat - c_mean
        
        iden = torch.eye(cFSize[0])  # .double()
        iden = iden.cuda()
        
        contentConv = torch.mm(cont_feat, cont_feat.t()).div(cFSize[1] - 1) + iden
        c_u, c_e, c_v = torch.svd(contentConv, some=False)
        
        k_c = cFSize[0]
        for i in range(cFSize[0] - 1, -1, -1):
            if c_e[i] >= 0.00001:
                k_c = i + 1
                break
        
        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cont_feat)
        
        noise = torch.randn(cont_feat.size())*self.var
        styl_feat = noise.cuda()
        sFSize = styl_feat.size()
        s_mean = torch.mean(styl_feat, 1)
        styl_feat = styl_feat - s_mean.unsqueeze(1).expand_as(styl_feat)
        styleConv = torch.mm(styl_feat, styl_feat.t()).div(sFSize[1] - 1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)
        
        k_s = sFSize[0]
        for i in range(sFSize[0] - 1, -1, -1):
            if s_e[i] >= 0.00001:
                k_s = i + 1
                break
        
        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        
        targetFeature = targetFeature.view_as(cont)
        
        return targetFeature
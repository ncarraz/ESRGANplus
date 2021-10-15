import torch
import torch.nn as nn
import torchvision
from . import block as B

# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=act_type, \
                             mode=mode)
        conv1 = B.conv_block(nf, nf, kernel_size=4, stride=2, norm_type=norm_type, \
                             act_type=act_type, mode=mode)
        # 64, 64
        conv2 = B.conv_block(nf, nf * 2, kernel_size=3, stride=1, norm_type=norm_type, \
                             act_type=act_type, mode=mode)
        conv3 = B.conv_block(nf * 2, nf * 2, kernel_size=4, stride=2, norm_type=norm_type, \
                             act_type=act_type, mode=mode)
        # 32, 128
        conv4 = B.conv_block(nf * 2, nf * 4, kernel_size=3, stride=1, norm_type=norm_type, \
                             act_type=act_type, mode=mode)
        conv5 = B.conv_block(nf * 4, nf * 4, kernel_size=4, stride=2, norm_type=norm_type, \
                             act_type=act_type, mode=mode)
        # 16, 256
        conv6 = B.conv_block(nf * 4, nf * 8, kernel_size=3, stride=1, norm_type=norm_type, \
                             act_type=act_type, mode=mode)
        conv7 = B.conv_block(nf * 8, nf * 8, kernel_size=4, stride=2, norm_type=norm_type, \
                             act_type=act_type, mode=mode)
        # 8, 512
        conv8 = B.conv_block(nf * 8, nf * 8, kernel_size=3, stride=1, norm_type=norm_type, \
                             act_type=act_type, mode=mode)
        conv9 = B.conv_block(nf * 8, nf * 8, kernel_size=4, stride=2, norm_type=norm_type, \
                             act_type=act_type, mode=mode)
        # 4, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# class Discriminator_VGG_128(nn.Module):
#     def __init__(self, in_nc, nf):
#         super(Discriminator_VGG_128, self).__init__()
#         # [64, 128, 128]
#         self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
#         self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
#         self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
#         # [64, 64, 64]
#         self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
#         self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
#         self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
#         self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
#         # [128, 32, 32]
#         self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
#         self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
#         self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
#         self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
#         # [256, 16, 16]
#         self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
#         self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
#         self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
#         self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
#         # [512, 8, 8]
#         self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
#         self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
#         self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
#         self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
#
#         self.linear1 = nn.Linear(512 * 4 * 4, 100)
#         self.linear2 = nn.Linear(100, 1)
#
#         # activation function
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#     def forward(self, x):
#         fea = self.lrelu(self.conv0_0(x))
#         fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))
#
#         fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
#         fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))
#
#         fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
#         fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))
#
#         fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
#         fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))
#
#         fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
#         fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
#
#         fea = fea.view(fea.size(0), -1)
#         fea = self.lrelu(self.linear1(fea))
#         out = self.linear2(fea)
#         return out


# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


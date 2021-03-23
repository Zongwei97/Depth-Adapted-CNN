import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from aCNN import computeOffset
#from dcn.modules.deform_conv import DeformConv
from torchvision.ops.deform_conv import DeformConv2d
import torchvision
import math


class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
        return input
    
class PoolingModule(nn.Module):
    def __init__(self, inplanes, kernel_size=3, stride=1, padding=0, dilation=1):
        super(PoolingModule, self).__init__()
        self.pool = DeformConv2d(inplanes, inplanes, kernel_size=kernel_size, stride=stride,padding=padding,dilation=dilation, groups = inplanes)
        self.pool.weight.data.fill_(1/kernel_size**2)
        self.pool.bias.data.zero_()
        for param in self.pool.parameters():
            param.requires_grad = False 
    def forward(self, x, offset):
        x = self.pool(x, offset)
        return x
    
    
class DCNN(nn.Module):
    def __init__(self, n_classes=21, learned_billinear=False):
        super(DCNN, self).__init__()
        self.n_classes = n_classes
        
        
        self.downsample_depth2 = nn.AvgPool2d(3, padding = 1, stride = 2)       

        self.conv_block1 = mySequential(
            DeformConv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )


        self.conv_block2 = mySequential(
            DeformConv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        

        self.conv_block3 = mySequential(
            DeformConv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        
        self.conv_block4 = mySequential(
            DeformConv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1, padding=1),
        )
        
        
        self.conv_block5 = mySequential(
            DeformConv2d(512, 512, 3, padding=2, dilation = 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3,  padding=2, dilation = 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3,  padding=2, dilation = 2),
            nn.ReLU(inplace=True),
        )
        self.pool = PoolingModule(512, 3, padding=1, dilation = 1)
        #self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.classifier = mySequential(
            nn.Conv2d(512, 1024, 3, padding = 12, dilation = 12),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1, padding = 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(1024, self.n_classes, 1),
        )
        #self._initialize_weights()
        self.init_vgg16_params()
        print('init')

    def forward(self, x, depth):
        depth = depth.unsqueeze(0)
        offset = computeOffset(depth[0], 3, 1)
        offset = F.pad(offset, (1,1,1,1), "constant", 0)
        
        conv1 = self.conv_block1(x, offset)
        
        depth = F.interpolate(depth, conv1.size()[2:], mode='bilinear', align_corners=True)
        offset = computeOffset(depth[0], 3, 2)
        offset = F.pad(offset, (1,1,1,1), "constant", 0)
        conv2 = self.conv_block2(conv1, offset)
        
        depth = F.interpolate(depth, conv2.size()[2:], mode='bilinear', align_corners=True)
        offset = computeOffset(depth[0], 3, 2)
        offset = F.pad(offset, (1,1,1,1), "constant", 0)
        conv3 = self.conv_block3(conv2, offset)
        
        
        depth = F.interpolate(depth, conv3.size()[2:], mode='bilinear', align_corners=True)
        offset = computeOffset(depth[0], 3, 2)
        offset = F.pad(offset, (1,1,1,1), "constant", 0)
        conv4 = self.conv_block4(conv3, offset)
        
                                
        depth = F.interpolate(depth, conv4.size()[2:], mode='bilinear', align_corners=True)
        offset = computeOffset(depth[0], 3, 2)
        offset = F.pad(offset, (1,1,1,1), "constant", 0)
        conv5 = self.conv_block5(conv4, offset)

        encoder = self.pool(conv5, offset)
        
        #encoder = self.pool(conv5)

        score = self.classifier(encoder)

        out = F.interpolate(score, x.size()[2:], mode='bilinear', align_corners=True)
        return out
    
    def _initialize_weights(self):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
            self.classifier,
        ]
        for idx, conv_block in enumerate(blocks):
            for m in conv_block:
                if isinstance(m, nn.Conv2d) or isinstance(m, DeformConv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
                
                
    def init_vgg16_params(self):
        vgg16 = torchvision.models.vgg16(pretrained=True)
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
            
        for idx, conv_block in enumerate(blocks):

            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if (isinstance(l1, nn.Conv2d) or isinstance(l1, DeformConv2d)) and (isinstance(l2, nn.Conv2d) or isinstance(l2, DeformConv2d)):
                    assert l1.weight.size() == l2.weight.size()
                    l2.weight.data = l1.weight.data
                    assert l1.bias.size() == l2.bias.size()
                    l2.bias.data = l1.bias.data
        for ly in self.classifier.children():
            if isinstance(ly, nn.Conv2d) or isinstance(ly, DeformConv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                nn.init.constant_(ly.bias, 0)

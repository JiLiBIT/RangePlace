'''FPN in PyTorch.

See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=[3,3], stride=stride, padding=[1,1], bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 256

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        # self.layer1 = self._make_layer(block, 128, num_blocks, stride=[2,1])

        # Top layer
        # self.toplayer = nn.Conv2d(512, 256, kernel_size=[1,1], stride=[1,1], padding=0)  # Reduce channels
        # self.toplayer = nn.Conv2d(768, 1536, kernel_size=[1,1], stride=[1,1], padding=0)  # Reduce channels

        # self.toplayer = nn.Conv2d(768, 1536, kernel_size=[3,2], stride=[2,2], padding=[0,0])  # Reduce channels
        # # self.toplayer = nn.Sequential()
        # # Smooth layers
        # self.smooth1 = nn.Conv2d(768, 768, kernel_size=[3,3], stride=1, padding=[1,1])
        # self.smooth2 = nn.Conv2d(384, 384, kernel_size=[3,3], stride=1, padding=[1,1])
        # self.smooth3 = nn.Conv2d(192, 192, kernel_size=[3,3], stride=1, padding=[1,1])
        # # Lateral layers
        # self.latlayer1 = nn.Conv2d(768, 768, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0)
        # self.latlayer3 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        # self.upsample1 =  nn.Sequential(
        #     nn.ConvTranspose2d(1536, 768, kernel_size=[3,2], stride=[2,2],padding=[0,0]),
        #     # nn.ConvTranspose2d(1536, 768, kernel_size=[3,2], stride=[2,2],padding=[0,1]),
        # )
        # self.upsample2 =  nn.Sequential(
        #     nn.ConvTranspose2d(768, 384, kernel_size=[2,2], stride=[2,2]),    
        # )
        # self.upsample3 =  nn.Sequential(
        #     nn.ConvTranspose2d(384, 192, kernel_size=[2,2], stride=[2,2]),
        # )
        self.toplayer = nn.Sequential(
                                    nn.Conv2d(128, 128, kernel_size=[1,1], stride=[2,1]),
                                    nn.GELU(),
                                      ) # Reduce channels
        # self.toplayer = nn.Sequential()
        # Smooth layers
        self.smooth1 = nn.Conv2d(128, 128, kernel_size=[3,1], stride=1, padding=[1,0])
        self.smooth2 = nn.Conv2d(128, 128, kernel_size=[3,1], stride=1, padding=[1,0])
        self.smooth3 = nn.Conv2d(128, 128, kernel_size=[3,1], stride=1, padding=[1,0])

        # Lateral layers
        self.latlayer1 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

        self.upsample1 =  nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=[1,1], stride=[2,1],padding=[0,0]),
            # nn.ConvTranspose2d(1536, 768, kernel_size=[3,2], stride=[2,2],padding=[0,1]),
        )
        self.upsample2 =  nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=[1,1], stride=[2,1]),    
        )
        self.upsample3 =  nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=[2,1], stride=[2,1]),
        )


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,C,H,W = y.size()
        
        return F.upsample(x, size=(H,W), mode='bilinear') + y
        return x + y

    def forward(self, features):
        # Bottom-up
        # c1 = F.relu(self.bn1(self.conv1(x)))
        # c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)

        c1,c2,c3,c4 = features
        # c5 = self.layer1(c4)

        # Top-down
        p5 = self.toplayer(c4)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # print(p5.shape)
        # print(self.upsample1(p5).shape)
        # p4 = self.upsample1(p5) + self.latlayer1(c4)
        # p3 = self.upsample2(p4) + self.latlayer2(c3)
        # p2 = self.upsample3(p3) + self.latlayer3(c2)
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5


def FPN101():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN(Bottleneck, 2)

if __name__ == "__main__":

    c1 = Variable(torch.randn(1,64,6,896))
    c2 = Variable(torch.randn(1,128,3,896))
    c3 = Variable(torch.randn(1,128,2,896))
    c4 = Variable(torch.randn(1,128,1,896))
    x =[c1,c2,c3,c4]
    net = FPN(Bottleneck, 2)
    
    fms = net(x)
    for fm in fms:
        print(fm.size())

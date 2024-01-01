import math
import torch.nn as nn
import torch.nn.functional as F
from Triplet import TripletAttention


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()
        
        #groups=in_channels=out_channels
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        #1x1
        self.pointwise = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
    
class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
       
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,kernel_size=1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.attention = TripletAttention()
        
        self.relu = nn.LeakyReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            
            rep.append(SeparableConv2d(in_filters,out_filters,kernel_size=3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,kernel_size=3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            #这里的卷积不改变特征图尺寸
            rep.append(SeparableConv2d(in_filters,out_filters,kernel_size=3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.LeakyReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(kernel_size=3,stride=strides,padding=1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        #x = self.attention(x)
        return x

class Xception(nn.Module):
    def __init__(self, num_classes=12):
        super(Xception, self).__init__()
        self.num_classes = num_classes

        ### define Entry flow ###
        self.conv1 = nn.Conv2d(in_channels=2050, out_channels=32, kernel_size=3,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3,stride=1,padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        # Block：in_filters,out_filters,reps,stride,start_with_relu,grow_first
        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)
         
        
        ###  Middle flow ###
        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        
        ### Exit flow ###
        self.block12=Block(728,1025,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1025,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2050,3,1,1)
        self.bn4 = nn.BatchNorm2d(2050)

        self.fc = nn.Linear(2050, num_classes)
        ############

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def forward(self, x):
        ### Entry flow ###
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        ### 定义 Middle flow ###
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        
        ### Exit flow ###
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
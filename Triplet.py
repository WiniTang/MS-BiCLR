import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            relu=True,
            bn=True,
            bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )

class ChannelShuffle(nn.Module):
    def __init__(self,groups):
        super(ChannelShuffle,self).__init__()
        self.groups=groups

    def forward(self,x):
        g=self.groups
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // g
        print(channels_per_group)
        # reshape
        # b, c, h, w =======>  b, g, c_per, h, w
        x = x.view(batch_size, g, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)
        return x
    
class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        
        # groups=in_channels=out_channels
        self.depthwise = BasicConv(2, 2, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=2, relu=True)
        
        self.pointwise = BasicConv(2, 1, kernel_size=1, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.depthwise(x_compress)
        x_out = self.depthwise(x_out)#？
        x_out = self.pointwise(x_out)
        scale = torch.sigmoid_(x_out)
        return x * scale

class AttentionGate1(nn.Module):
    def __init__(self):
        super(AttentionGate1, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        '''
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )
        '''
        # groups=in_channels=out_channels
        self.depthwise = BasicConv(2, 2, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=2, relu=True)

        self.pointwise = BasicConv(2, 1, kernel_size=1, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.depthwise(x_compress)
        x_out = self.depthwise(x_out)
        x_out = self.pointwise(x_out)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

        # multi-scale
            # groups=in_channels=out_channels
            self.conv1dw = BasicConv(2050, 1025, 5, stride=1, padding=(5 - 1) // 2, groups=1025, relu=True)
            #channel shuffle
            #self.shuffle=ChannelShuffle(512)
            
            self.conv1pw = BasicConv(1025, 1025, kernel_size=1, relu=False)
            self.conv2dw = BasicConv(2050, 1025, 3, stride=1, padding=(3 - 1) // 2, groups=1025, relu=True)
            self.conv2pw = BasicConv(1025, 1025, kernel_size=1, relu=False)
        
    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            xconv1 = self.conv1dw(x)
            
            xconv2 = self.conv2dw(x)
            xconv = torch.cat((xconv1,xconv2),1)
            x_out = self.hw(xconv)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)

        return x_out

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet

from torchvision.models.resnet import resnet50

from Xception import Xception

# SE-Block
class SE_Block(nn.Module):
    def __init__(self, input_channel, reduction=16):
        super(SE_Block, self).__init__()

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(input_channel, input_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(input_channel // reduction, input_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.adaptive_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)  
        y = x * y.expand_as(x) 
        return  x+y

# stage one ,unsupervised learning
class SimCLRStage1(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLRStage1, self).__init__()
        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d) and not isinstance(module, nn.AdaptiveAvgPool2d):
                self.f.append(module)
        # encoder
        self.f.append(SE_Block(2048))
        self.f = nn.Sequential(*self.f)
        
       
        # 3 linear layers
        self.g = nn.Sequential(nn.Linear(32768, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, 512, bias=True),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))
        

    def forward(self, x):
        feature = self.f(x)
        feature = torch.flatten(feature, start_dim=1)  # 从batch_size下一维到最后一个维度进行展平
        out = self.g(feature)
        out = torch.flatten(out, start_dim=1)  # 从batch_size下一维到最后一个维度进行展平
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


# stage two ,supervised learning
class SimCLRStage2(nn.Module):
    def __init__(self, num_class):
        super(SimCLRStage2, self).__init__()#对继承自父类的属性进行初始化，并且用父类的初始化方法初始化继承的属性。
        # encoder
        self.f = SimCLRStage1().f #resnet50
        
        
        # classifier_Xception
        self.fc = Xception(num_classes=num_class)

        #
        ##################################
        for param in self.f.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.f(x)
        out = self.fc(x)
        return out


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, out_1, out_2, batch_size, temperature=0.5):
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        return (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()  

        
    


if __name__ == "__main__":
    for name, module in resnet50().named_children():
        print(name, module)

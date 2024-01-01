from torch import nn
import torch
import numpy as np
from net import SimCLRStage1
from Xception import Xception
from skimage import feature, exposure
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from Triplet import TripletAttention
import torch.nn.functional as F

class SimCLRStage2(nn.Module):
    def __init__(self, num_class):
        super(SimCLRStage2, self).__init__()
        # encoder
        self.f = SimCLRStage1().f # resnet50
        for param in self.f.parameters():
            param.requires_grad = False
        
        self.attention = TripletAttention()
        self.fc = Xception(num_classes=num_class) 
        
        self.bn = nn.BatchNorm2d(2050)

        '''
        # classifier
        self.classsifier = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                         nn.Linear(2049, 512, bias=False),
                                         nn.BatchNorm1d(512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, num_class, bias=True))
        '''

    def forward(self, x_device, x):
        x_device = self.f(x_device) 
        
        x_c = x_device.to(torch.device('cpu'))
        
        zeros = torch.zeros((x_device.shape[0],1,30,40)).to(torch.device('cuda:0'))
        x_device = torch.cat((x_device,zeros),1)
        x_device = torch.cat((x_device,zeros),1)

        trans = transforms.ToTensor()
        transgray = transforms.Grayscale(1)
        transPIL = transforms.ToPILImage()
        for i in range(x.shape[0]):
            img = x[i,:,:,:].to(torch.device('cpu')).numpy()
            img = np.transpose(img,(1,2,0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp_feature = feature.local_binary_pattern(img,P=8,R=1,method='default')
            lbp_feature = Image.fromarray(lbp_feature)
            lbp_feature = lbp_feature.resize((x_device.shape[-1],x_device.shape[-2]),resample=Image.Resampling.BICUBIC)
            lbp_feature = trans(lbp_feature)
            
            x_device[i,:,:,:] = torch.cat((torch.cat((lbp_feature,lbp_feature),0),x_c[i,:,:,:]),0).to(torch.device('cuda:0')) # 对纹理特征以及经过编码器结构的特征图进行拼接
            
        x_device = self.attention(x_device) 
        
        out = self.fc(x_device) # Xception


        return out




     

class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """
    def __init__(
            self,
            bins=1,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        #self.edges = torch.arange(bins + 1).float().cuda() / bins
        #self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight
        
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predo, target, epoch, batch, *args, **kwargs):
        
        
        bins = int(40 ** (((epoch-1)*100+batch+1)/20000))
        self.bins = bins
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(target,dtype=torch.float)

        pred = F.softmax(predo,dim=1)
        g = torch.abs(pred[torch.arange(pred.shape[0]),target].detach() - 1)
        n = 0  # n valid bins
        g_ = []
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1])
            print(i,'edges',inds)
            num_in_bin = inds.sum().item()
            g_.append(num_in_bin)
            if num_in_bin > 0:
                '''
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = 1 / self.acc_sum[i]
                else:
                '''
                weights[inds] = 1.0/num_in_bin
                
                n += 1
        if n > 0:
            weights = weights / n
        
        
        if epoch % 5 == 0 and batch % 10:
            file_handler = open('g.txt',mode='a')
            file_handler.write(str(epoch)+":"+str(batch)+":")
            file_handler.write(str(g_)+'\n')
            file_handler.close()
        print(str(epoch)+":"+str(batch)+":"+str(g_)+'\n')
        
        loss = self.loss_func(predo, target) * weights
        return loss.sum() * self.loss_weight
    
from torch import nn
import sys
import torch.nn.functional as F
import torch
sys.path.append('../kan_convolutional')

from kan_convolutional.KANConv import KAN_Convolutional_Layer
from kan_convolutional.KANLinear import KANLinear

class KKAN_Convolutional_Network(nn.Module):
    def __init__(self,label_num,device: str = 'cpu'):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size= (3,3),
            device = device
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size = (3,3),
            device = device
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 

        self.kan1 = KANLinear(
            625,
            label_num,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1],
        )


    def forward(self, x):
        x = self.conv1(x)
        #print("x.shape:", x.shape)
        x = self.pool1(x)
        #print("x.shape:", x.shape)
        x = self.conv2(x)
        #print("x.shape:", x.shape)
        x = self.pool1(x)
        #print("x.shape:", x.shape)
        x = self.flat(x)
        #print("x.shape:", x.shape)
        x = self.kan1(x)
        #print("x.shape:", x.shape)
        x = F.log_softmax(x, dim=1)

        return x

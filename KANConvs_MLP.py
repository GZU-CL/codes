from torch import nn
import sys
import torch.nn.functional as F
import torch
sys.path.append('../kan_convolutional')
from kan_convolutional.KANConv import KAN_Convolutional_Layer

class KANC_MLP(nn.Module):
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
        
        self.linear1 = nn.Linear(625, 256)
        self.linear2 = nn.Linear(256, label_num)


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
        x = self.linear1(x)
        #print("x.shape:", x.shape)
        x = self.linear2(x)
        #print("x.shape:", x.shape)
        x = F.log_softmax(x, dim=1)
        return x


# label_num = 12
#
# # 创建模型实例
# device = 'cpu'
# model = KANC_MLP(label_num, device)
#
# # 创建一个28x28的输入张量（批次大小为1，单通道图像）
# input_tensor = torch.randn(1, 1, 28, 28)  # (batch_size, channels, height, width)
#
# # 前向传播
# output = model(input_tensor)
#
# # 打印输出大小
# print("输出大小：", output.size())
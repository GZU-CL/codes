from torch import nn
import sys
import torch.nn.functional as F
import torch
sys.path.append('../kan_convolutional')

from kan_convolutional.KANLinear import KANLinear

class NormalConvsKAN(nn.Module):
    def __init__(self,label_num):
        super(NormalConvsKAN, self).__init__()
        # Convolutional layer, assuming an input with 1 channel (grayscale image)
        # and producing 16 output channels, with a kernel size of 3x3
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)

        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Flatten layer
        self.flatten = nn.Flatten()

        # KAN layer
        self.kan1 = KANLinear(
            245,
            label_num,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1])


    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print("x.shape:", x.shape)
        x = self.maxpool(x)
       # print("x.shape:", x.shape)
        x = F.relu(self.conv2(x))
        #print("x.shape:", x.shape)
        x = self.maxpool(x)
        #print("x.shape:", x.shape)
        x = self.flatten(x)
       # print("x.shape:", x.shape)
        x = self.kan1(x)
       # print("x.shape:", x.shape)
        x = F.log_softmax(x, dim=1)

        return x

# label_num = 12
#
# # 创建模型实例
# device = 'cpu'
# model = NormalConvsKAN(label_num)
#
# # 创建一个28x28的输入张量（批次大小为1，单通道图像）
# input_tensor = torch.randn(1, 1, 28, 28)  # (batch_size, channels, height, width)
#
# # 前向传播
# output = model(input_tensor)
#
# # 打印输出大小
# print("输出大小：", output.size())
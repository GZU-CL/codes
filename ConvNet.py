from torch import nn
import torch
import torch.nn.functional as F
import sys
# directory reach
# sys.path.append('../kan_convolutional')

class ConvNet(nn.Module):
    def __init__(self,label_num):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding='same')

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same')

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 7 * 7, 256) 
        self.fc2 = nn.Linear(256, label_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        print("x.shape:", x.shape)
        x = F.relu(self.conv2(x))
        print("x.shape:", x.shape)
        x = self.maxpool(x)
        print("x.shape:", x.shape)
        x = self.dropout1(x)
        print("x.shape:", x.shape)

        x = F.relu(self.conv3(x))
        print("x.shape:", x.shape)
        x = F.relu(self.conv4(x))
        print("x.shape:", x.shape)
        x = self.maxpool(x)
        print("x.shape:", x.shape)
        x = self.dropout2(x)
        print("x.shape:", x.shape)

        x = torch.flatten(x, 1)
        print("x.shape:", x.shape)
        x = F.relu(self.fc1(x))
        print("x.shape:", x.shape)
        x = self.dropout3(x)
        print("x.shape:", x.shape)
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)
        return x


label_num = 12

# 创建模型实例
device = 'cpu'
model = ConvNet(label_num)

# 创建一个28x28的输入张量（批次大小为1，单通道图像）
input_tensor = torch.randn(1, 1, 28, 28)  # (batch_size, channels, height, width)

# 前向传播
output = model(input_tensor)

# 打印输出大小
print("输出大小：", output.size())
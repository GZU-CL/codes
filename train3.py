import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import gzip
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torch import nn
import sys
import torch.nn.functional as F
from kan_convolutional.KANLinear import KANLinear
from architectures_28x28.SimpleModels import *


# 设置标签数量
label_num = 12
epochs = 10
transform = None  # 如果有需要的transform，可以设置
# 自定义数据集类
class DealDataset(Dataset):
    """
    读取数据、初始化数据
    """
    def __init__(self, data_name, label_name, transform=None):
        self.data, self.labels = self.load_data(data_name, label_name)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], int(self.labels[idx])
        img = img.copy()
        img = img.reshape(1, 1, 28 * 28)  # Ensure shape (1, 1, 784)
        if self.transform:
            img = self.transform(img)
        return img, label

    def load_data(self, data_name, label_name):
        with open(label_name, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with open(data_name, 'rb') as imgpath:
            data = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(labels), 28, 28)
        return data, labels

class CustomDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        self.data, self.labels = self.load_data(data_path, label_path)
        self.transform = transform
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def load_data(self, data_path, label_path):
        with open(data_path, 'rb') as imgpath:
            imgpath.read(16)  # 跳过图像文件头部的16个字节
            data = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(-1, 28, 28).copy()

        with open(label_path, 'rb') as lbpath:
            lbpath.read(8)  # 跳过标签文件头部的8个字节
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8).copy()

        return data, labels

# 图像预处理，转换为张量并标准化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载训练和测试数据
train_data_path = './data/12class/SessionAllLayers/train-images-idx3-ubyte'
train_label_path = './data/12class/SessionAllLayers/train-labels-idx1-ubyte'
test_data_path = './data/12class/SessionAllLayers/test-images-idx3-ubyte'
test_label_path = './data/12class/SessionAllLayers/test-labels-idx1-ubyte'

train_dataset = CustomDataset(train_data_path, train_label_path, transform=transform)
test_dataset = CustomDataset(test_data_path, test_label_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#一维卷积的
trainDataset = DealDataset(train_data_path,train_label_path,transform=transform)
testDataset = DealDataset(test_data_path,test_label_path,transform=transform)
train_loader1D = torch.utils.data.DataLoader(
    dataset=trainDataset,
    batch_size=32,
    shuffle=True
)

test_loader1D = torch.utils.data.DataLoader(
    dataset=testDataset,
    batch_size=32,
    shuffle=False
)


# class KAN1(nn.Module):
#     def __init__(self, label_num, device: str = 'cpu'):
#         super().__init__()
#         self.flat = nn.Flatten()
#         self.kan1 = KANLinear(
#             28 * 28,
#             label_num,
#             grid_size=10,
#             spline_order=3,
#             scale_noise=0.01,
#             scale_base=1,
#             scale_spline=1,
#             base_activation=nn.SiLU,
#             grid_eps=0.02,
#             grid_range=[0, 1],
#         )
#
#     def forward(self, x):
#         x = self.flat(x)
#         x = self.kan1(x)
#         x = F.log_softmax(x, dim=1)
#         return x
#

# Example usage
class OneCNN(nn.Module):
    def __init__(self, label_num):
        super(OneCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1,25), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1,25), padding=1)

        self.maxpool = nn.MaxPool2d((1, 3), 3, padding=0)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(87*64,1024)
        self.fc2 =nn.Linear(1024, label_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = F.log_softmax(x, dim=1)

        return x


# 训练函数
# def train(model, device, train_loader, optimizer, epoch, criterion):
#     model.to(device)
#     model.train()
#     train_loss = 0
#     start_time = time.time()
#     for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         train_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     avg_loss = train_loss / (batch_idx + 1)
#     end_time = time.time()
#     return avg_loss, end_time - start_time
#
# # 测试函数
# def test(model, device, test_loader, criterion):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     all_targets = []
#     all_predictions = []
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             _, predicted = torch.max(output.data, 1)
#             correct += (target == predicted).sum().item()
#             all_targets.extend(target.view_as(predicted).cpu().numpy())
#             all_predictions.extend(predicted.cpu().numpy())
#
#     precision = precision_score(all_targets, all_predictions, average='macro')
#     recall = recall_score(all_targets, all_predictions, average='macro')
#     f1 = f1_score(all_targets, all_predictions, average='macro')
#
#     test_loss /= len(test_loader.dataset)
#     accuracy = correct / len(test_loader.dataset)
#
#     return test_loss, accuracy, precision, recall, f1
#
# # 训练和测试模型函数
# def train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs, scheduler, method_name):
#     all_train_loss = []
#     all_test_loss = []
#     all_test_accuracy = []
#     all_test_precision = []
#     all_test_recall = []
#     all_test_f1 = []
#
#     for epoch in range(1, epochs + 1):
#         train_loss, training_time = train(model, device, train_loader, optimizer, epoch, criterion)
#         all_train_loss.append(train_loss)
#         test_loss, test_accuracy, test_precision, test_recall, test_f1 = test(model, device, test_loader, criterion)
#         all_test_loss.append(test_loss)
#         all_test_accuracy.append(test_accuracy)
#         all_test_precision.append(test_precision)
#         all_test_recall.append(test_recall)
#         all_test_f1.append(test_f1)
#         print(f"Using method: {method_name}\n"
#               f"End of Epoch {epoch}: training time: {training_time:.2f} seconds, "
#               f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.4f}, "
#               f"Accuracy: {test_accuracy:.2%}")
#         scheduler.step()
#
#     model.all_test_accuracy = all_test_accuracy
#     model.all_test_precision = all_test_precision
#     model.all_test_f1 = all_test_f1
#     model.all_test_recall = all_test_recall
#
#     return all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 训练函数
def train(model, device, train_loader, optimizer, epoch, criterion):
    model.to(device)
    model.train()
    train_loss = 0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = train_loss / (batch_idx + 1)
    end_time = time.time()
    return avg_loss, end_time - start_time

# 测试函数
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += (target == predicted).sum().item()
            all_targets.extend(target.view_as(predicted).cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    return test_loss, accuracy, precision, recall, f1

# 训练和测试模型函数
def train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs, scheduler, method_name):
    all_train_loss = []
    all_test_loss = []
    all_test_accuracy = []
    all_test_precision = []
    all_test_recall = []
    all_test_f1 = []

    for epoch in range(1, epochs + 1):
        train_loss, training_time = train(model, device, train_loader, optimizer, epoch, criterion)
        all_train_loss.append(train_loss)
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = test(model, device, test_loader, criterion)
        all_test_loss.append(test_loss)
        all_test_accuracy.append(test_accuracy)
        all_test_precision.append(test_precision)
        all_test_recall.append(test_recall)
        all_test_f1.append(test_f1)
        print(f"Using method: {method_name}\n"
              f"End of Epoch {epoch}: training time: {training_time:.2f} seconds, "
              f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.4f}, "
              f"Accuracy: {test_accuracy:.2%}")
        scheduler.step()

    model.all_test_accuracy = all_test_accuracy
    model.all_test_precision = all_test_precision
    model.all_test_f1 = all_test_f1
    model.all_test_recall = all_test_recall

    return all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义参数计数函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 定义参数计数函数
#一维卷积
model_OneCNN = OneCNN(label_num)
optimizer_OneCNN = optim.AdamW(model_OneCNN.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_OneCNN = optim.lr_scheduler.ExponentialLR(optimizer_OneCNN, gamma=0.8)
criterion_OneCNN = nn.CrossEntropyLoss()
method_name = "OneCNN"

train_and_test_models(model_OneCNN, device, train_loader1D, test_loader1D, optimizer_OneCNN, criterion_OneCNN, epochs, scheduler_OneCNN, method_name)
print(f"Total number of trainable parameters: {count_parameters(model_OneCNN)}")

#定义和训练不同的模型
# model_KAN1 = KAN1(label_num)
# model_KAN1.to(device)
# optimizer_KAN1 = optim.AdamW(model_KAN1.parameters(), lr=1e-3, weight_decay=1e-4)
# scheduler_KAN1 = optim.lr_scheduler.ExponentialLR(optimizer_KAN1, gamma=0.8)
# criterion_KAN1 = nn.CrossEntropyLoss()
# all_train_loss_KAN1, \
# all_test_loss_KAN1, \
# all_test_accuracy_KAN1, \
# all_test_precision_KAN1, \
# all_test_recall_KAN1, \
# all_test_f1_KAN1 = train_and_test_models(
#     model_KAN1,
#     device,
#     train_loader,
#     test_loader,
#     optimizer_KAN1,
#     criterion_KAN1,
#     epochs,
#     scheduler=scheduler_KAN1,
#     method_name="KAN"
# )

#普通MLP
# model_SimpleLinear = SimpleLinear(label_num)
# model_SimpleLinear.to(device)
# optimizer_SimpleLinear = optim.AdamW(model_SimpleLinear.parameters(), lr=1e-3, weight_decay=1e-4)
# scheduler_SimpleLinear = optim.lr_scheduler.ExponentialLR(optimizer_SimpleLinear, gamma=0.8)
# criterion_SimpleLinear = nn.CrossEntropyLoss()
# all_train_loss_SimpleLinear, \
# all_test_loss_SimpleLinear, \
# all_test_accuracy_SimpleLinear, \
# all_test_precision_SimpleLinear, \
# all_test_recall_SimpleLinear, \
# all_test_f1_SimpleLinear = train_and_test_models(
#     model_SimpleLinear,
#     device,
#     train_loader,
#     test_loader,
#     optimizer_SimpleLinear,
#     criterion_SimpleLinear,
#     epochs,
#     scheduler=scheduler_SimpleLinear,
#     method_name="SimpleLinear"
# )
# #小卷积
# model_SimpleCNN = SimpleCNN(label_num)
# model_SimpleCNN.to(device)
# optimizer_SimpleCNN = optim.AdamW(model_SimpleCNN.parameters(), lr=1e-3, weight_decay=1e-4)
# scheduler_SimpleCNN = optim.lr_scheduler.ExponentialLR(optimizer_SimpleCNN, gamma=0.8)
# criterion_SimpleCNN = nn.CrossEntropyLoss()
# all_train_loss_SimpleCNN, \
# all_test_loss_SimpleCNN, \
# all_test_accuracy_SimpleCNN, \
# all_test_precision_SimpleCNN, \
# all_test_recall_SimpleCNN, \
# all_test_f1_SimpleCNN = train_and_test_models(
#     model_SimpleCNN,
#     device,
#     train_loader,
#     test_loader,
#     optimizer_SimpleCNN,
#     criterion_SimpleCNN,
#     epochs,
#     scheduler=scheduler_SimpleCNN,
#     method_name="SimpleCNN"
# )
#
# # 定义参数计数函数
# # def count_parameters(model):
# #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# # 创建绘图空间
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
# # 绘制测试损失曲线
# ax1.plot(all_test_loss_KAN1, label='Loss KAN', color='black')
# ax1.plot(all_test_loss_SimpleCNN, label='Loss ConvNet(Small)', color='red')
# ax1.plot(all_test_loss_SimpleLinear, label='Loss 1 Layer & MLP', color='green')
#
#
# ax1.set_title('Loss Test vs Epochs')
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Loss')
# ax1.legend()
# ax1.grid(True)
# # 绘制模型参数数量与准确度的关系
# ax2.scatter(count_parameters(model_KAN1), max(all_test_accuracy_KAN1), color='black', label='KAN')
# ax2.scatter(count_parameters(model_SimpleCNN), max(all_test_accuracy_SimpleCNN), color='red', label='ConvNet (Small)')
# ax2.scatter(count_parameters(model_SimpleLinear), max(all_test_accuracy_SimpleLinear), color='green', label='1 Layer MLP')
#
#
# ax2.set_title('Number of Parameters vs Accuracy')
# ax2.set_xlabel('Number of Parameters')
# ax2.set_ylabel('Accuracy (%)')
# ax2.legend()
# ax2.grid(True)
#
# plt.tight_layout()
# plt.show()
#
# # 定义一个函数来突出显示 DataFrame 中的最大值
# def highlight_max(s):
#     is_max = s == s.max()
#     return ['font-weight: bold' if v else '' for v in is_max]
#
#
# # 创建一个 DataFrame 来存储结果
# accs = []
# precision = []
# recall = []
# f1s = []
# params_counts = []
#
#
# models = [model_KAN1,model_SimpleLinear, model_SimpleCNN]
#
# # 记录模型指标
# for i, m in enumerate(models):
#     index = np.argmax(m.all_test_accuracy)
#     params_counts.append(count_parameters(m))
#     accs.append(m.all_test_accuracy[index])
#     precision.append(m.all_test_precision[index])
#     recall.append(m.all_test_recall[index])
#     f1s.append(m.all_test_f1[index])
#
# # 创建 DataFrame
# df = pd.DataFrame({
#     "Test Accuracy": accs,
#     "Test Precision": precision,
#     "Test Recall": recall,
#     "Test F1 Score": f1s,
#     "Number of Parameters": params_counts
# }, index=["1 Layer MLP", "ConvNet (Small)","KAN1"])
# # 将 DataFrame 保存为 CSV 文件
# df.to_csv('experiment_28x281.csv', index=False)

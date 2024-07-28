import numpy as np
import torch
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from sklearn.metrics import precision_score, recall_score, f1_score
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
        img = img.reshape(1, 1, 28 * 28).astype(np.float32)  # Ensure shape (1, 1, 784) and type float32
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



import torch.nn as nn
import torch.nn.functional as F


class OneCNN(nn.Module):
    def __init__(self, label_num):
        super(OneCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 25), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 25), padding=1)
        self.maxpool = nn.MaxPool2d((1, 3), 3, padding=0)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()

        # 计算flatten层之前的特征图尺寸
        self._to_linear = None
        self.convs = nn.Sequential(self.conv1, nn.ReLU(), self.maxpool, self.conv2, nn.ReLU(), self.maxpool)
        self._initialize_flatten_shape()

        self.fc = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, label_num)

    def _initialize_flatten_shape(self):
        # 使用一个dummy输入来计算flatten层之前的特征图尺寸
        x = torch.rand(1, 1, 1, 784)
        x = self.convs(x)
        self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]

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
def train(model, device, train_loader, optimizer, epoch, criterion):
    model.to(device)
    model.train()
    train_loss = 0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device).float(), target.to(device)
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
            data, target = data.to(device).float(), target.to(device)
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

# 初始化和训练模型
label_num = 12
epochs = 10
transform = None
train_data_path = './data/12class/SessionAllLayers/train-images-idx3-ubyte'
train_label_path = './data/12class/SessionAllLayers/train-labels-idx1-ubyte'
test_data_path = './data/12class/SessionAllLayers/test-images-idx3-ubyte'
test_label_path = './data/12class/SessionAllLayers/test-labels-idx1-ubyte'

trainDataset = DealDataset(train_data_path,train_label_path,transform=transform)
testDataset = DealDataset(test_data_path,test_label_path,transform=transform)
train_loader1D = DataLoader(dataset=trainDataset, batch_size=32, shuffle=True)
test_loader1D = DataLoader(dataset=testDataset, batch_size=32, shuffle=False)

model_OneCNN = OneCNN(label_num)
optimizer_OneCNN = optim.AdamW(model_OneCNN.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_OneCNN = optim.lr_scheduler.ExponentialLR(optimizer_OneCNN, gamma=0.9)
criterion_OneCNN = nn.CrossEntropyLoss()
method_name = "OneCNN"

train_and_test_models(model_OneCNN, device, train_loader1D, test_loader1D, optimizer_OneCNN, criterion_OneCNN, epochs, scheduler_OneCNN, method_name)
print(f"Total number of trainable parameters: {count_parameters(model_OneCNN)}")

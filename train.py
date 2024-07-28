import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


from architectures_28x28.SimpleModels import *
from architectures_28x28.ConvNet import ConvNet
from architectures_28x28.KANConvs_MLP import KANC_MLP
from architectures_28x28.KKAN import KKAN_Convolutional_Network
from architectures_28x28.conv_and_kan import NormalConvsKAN
from architectures_28x28.KANmlp import KAN1

# 设置标签数量
label_num = 12
epochs = 20

# 自定义数据集类
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
#12分类流的所有层
# train_data_path = './data/12class/FlowAllLayers/train-images-idx3-ubyte'
# train_label_path = './data/12class/FlowAllLayers/train-labels-idx1-ubyte'
# test_data_path = './data/12class/FlowAllLayers/test-images-idx3-ubyte'
# test_label_path = './data/12class/FlowAllLayers/test-labels-idx1-ubyte'
#12分类流的L7
train_data_path = './data/12class/FlowL7/train-images-idx3-ubyte'
train_label_path = './data/12class/FlowL7/train-labels-idx1-ubyte'
test_data_path = './data/12class/FlowL7/test-images-idx3-ubyte'
test_label_path = './data/12class/FlowL7/test-labels-idx1-ubyte'
#12分类会话的所有层
# train_data_path = './data/12class/SessionAllLayers/train-images-idx3-ubyte'
# train_label_path = './data/12class/SessionAllLayers/train-labels-idx1-ubyte'
# test_data_path = './data/12class/SessionAllLayers/test-images-idx3-ubyte'
# test_label_path = './data/12class/SessionAllLayers/test-labels-idx1-ubyte'
#12分类会话的L7
# train_data_path = './data/12class/SessionL7/train-images-idx3-ubyte'
# train_label_path = './data/12class/SessionL7/train-labels-idx1-ubyte'
# test_data_path = './data/12class/SessionL7/test-images-idx3-ubyte'
# test_label_path = './data/12class/SessionL7/test-labels-idx1-ubyte'

train_dataset = CustomDataset(train_data_path, train_label_path, transform=transform)
test_dataset = CustomDataset(test_data_path, test_label_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

# 定义和训练不同的模型
# #普通MLP
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
# #一个kan层
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

#小卷积
model_SimpleCNN = SimpleCNN(label_num)
model_SimpleCNN.to(device)
optimizer_SimpleCNN = optim.AdamW(model_SimpleCNN.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_SimpleCNN = optim.lr_scheduler.ExponentialLR(optimizer_SimpleCNN, gamma=0.8)
criterion_SimpleCNN = nn.CrossEntropyLoss()
all_train_loss_SimpleCNN, \
all_test_loss_SimpleCNN, \
all_test_accuracy_SimpleCNN, \
all_test_precision_SimpleCNN, \
all_test_recall_SimpleCNN, \
all_test_f1_SimpleCNN = train_and_test_models(
    model_SimpleCNN,
    device,
    train_loader,
    test_loader,
    optimizer_SimpleCNN,
    criterion_SimpleCNN,
    epochs,
    scheduler=scheduler_SimpleCNN,
    method_name="SimpleCNN"
)
#中卷积
model_SimpleCNN_2 = SimpleCNN_2(label_num)
model_SimpleCNN_2.to(device)
optimizer_SimpleCNN_2 = optim.AdamW(model_SimpleCNN_2.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_SimpleCNN_2 = optim.lr_scheduler.ExponentialLR(optimizer_SimpleCNN_2, gamma=0.8)
criterion_SimpleCNN_2 = nn.CrossEntropyLoss()
all_train_loss_SimpleCNN_2, \
all_test_loss_SimpleCNN_2, \
all_test_accuracy_SimpleCNN_2, \
all_test_precision_SimpleCNN_2, \
all_test_recall_SimpleCNN_2, \
all_test_f1_SimpleCNN_2 = train_and_test_models(
    model_SimpleCNN_2,
    device,
    train_loader,
    test_loader,
    optimizer_SimpleCNN_2,
    criterion_SimpleCNN_2,
    epochs,
    scheduler=scheduler_SimpleCNN_2,
    method_name="SimpleCNN_2"
)
#四层大卷积
model_ConvNet = ConvNet(label_num)
model_ConvNet.to(device)
optimizer_ConvNet = optim.AdamW(model_ConvNet.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_ConvNet = optim.lr_scheduler.ExponentialLR(optimizer_ConvNet, gamma=0.8)
criterion_ConvNet = nn.CrossEntropyLoss()
all_train_loss_ConvNet, \
all_test_loss_ConvNet, \
 all_test_accuracy_ConvNet,\
 all_test_precision_ConvNet, \
 all_test_recall_ConvNet, \
 all_test_f1_ConvNet = train_and_test_models(
    model_ConvNet,
    device,
    train_loader,
    test_loader,
    optimizer_ConvNet,
    criterion_ConvNet,
    epochs,
    scheduler=scheduler_ConvNet,
    method_name="big ConvNet"
)

#kan卷积+普通mlp层
model_KANC_MLP= KANC_MLP(label_num,device = device)
model_KANC_MLP.to(device)
optimizer_KANC_MLP = optim.AdamW(model_KANC_MLP.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_KANC_MLP = optim.lr_scheduler.ExponentialLR(optimizer_KANC_MLP, gamma=0.8)
criterion_KANC_MLP = nn.CrossEntropyLoss()
all_train_loss_KANC_MLP, \
all_test_loss_KANC_MLP, \
all_test_accuracy_KANC_MLP, \
all_test_precision_KANC_MLP, \
all_test_recall_KANC_MLP, \
all_test_f1_KANC_MLP = train_and_test_models(
    model_KANC_MLP,
    device,
    train_loader,
    test_loader,
    optimizer_KANC_MLP,
    criterion_KANC_MLP,
    epochs,
    scheduler=scheduler_KANC_MLP,
    method_name="KANC_MLP"
)
#普通卷积+kan层
model_Convs_and_KAN= NormalConvsKAN(label_num)
model_Convs_and_KAN.to(device)
optimizer_Convs_and_KAN = optim.AdamW(model_Convs_and_KAN.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_Convs_and_KAN = optim.lr_scheduler.ExponentialLR(optimizer_Convs_and_KAN, gamma=0.8)
criterion_Convs_and_KAN = nn.CrossEntropyLoss()
all_train_loss_Convs_and_KAN, \
all_test_loss_Convs_and_KAN, \
all_test_accuracy_Convs_and_KAN, \
all_test_precision_Convs_and_KAN, \
all_test_recall_Convs_and_KAN, \
all_test_f1_Convs_and_KAN = train_and_test_models(
    model_Convs_and_KAN,
    device,
    train_loader,
    test_loader,
    optimizer_Convs_and_KAN,
    criterion_Convs_and_KAN,
    epochs,
    scheduler=scheduler_Convs_and_KAN,
    method_name="NormalConvsKAN"
)
#kan卷积+kan层
model_KKAN_Convolutional_Network = KKAN_Convolutional_Network(label_num,device = device)
model_KKAN_Convolutional_Network.to(device)
optimizer_KKAN_Convolutional_Network = optim.AdamW(model_KKAN_Convolutional_Network.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_KKAN_Convolutional_Network = optim.lr_scheduler.ExponentialLR(optimizer_KKAN_Convolutional_Network, gamma=0.8)
criterion_KKAN_Convolutional_Network = nn.CrossEntropyLoss()
all_train_loss_KKAN_Convolutional_Network, \
all_test_loss_KKAN_Convolutional_Network, \
all_test_accuracy_KKAN_Convolutional_Network, \
all_test_precision_KKAN_Convolutional_Network, \
all_test_recall_KKAN_Convolutional_Network, \
all_test_f1_KKAN_Convolutional_Network = train_and_test_models(
    model_KKAN_Convolutional_Network,
    device,
    train_loader,
    test_loader,
    optimizer_KKAN_Convolutional_Network,
    criterion_KKAN_Convolutional_Network,
    epochs,
    scheduler=scheduler_KKAN_Convolutional_Network,
    method_name="KKAN_Convolutional_Network"
)
# # 训练损失图表
# plt.figure()
# plt.plot(all_train_loss_SimpleCNN, label='SimpleCNN')
# plt.plot(all_train_loss_SimpleCNN_2, label='SimpleCNN_2')
# plt.plot(all_train_loss_SimpleLinear, label='SimpleLinear')
# plt.plot(all_train_loss_ConvNet, label='ConvNet')
# plt.plot(all_train_loss_KKAN_Convolutional_Network, label='KKAN_Convolutional_Network')
# plt.xlabel('Epoch')
# plt.ylabel('Training Loss')
# plt.title('Training Loss vs. Epoch')
# plt.legend()
# plt.show()
#
# # 测试精度图表
# plt.figure()
# plt.plot(all_test_accuracy_SimpleCNN, label='SimpleCNN')
# plt.plot(all_test_accuracy_SimpleCNN_2, label='SimpleCNN_2')
# plt.plot(all_test_accuracy_SimpleLinear, label='SimpleLinear')
# plt.plot(all_test_accuracy_ConvNet, label='ConvNet')
# plt.plot(all_test_accuracy_KKAN_Convolutional_Network, label='KKAN_Convolutional_Network')
# plt.xlabel('Epoch')
# plt.ylabel('Test Accuracy')
# plt.title('Test Accuracy vs. Epoch')
# plt.legend()
# plt.show()
#
# # 将模型结果保存到CSV文件中
# models = ['SimpleCNN', 'SimpleCNN_2', 'SimpleLinear', 'ConvNet', 'KKAN_Convolutional_Network']
# data = {
#     'Model': models,
#     'Accuracy': [max(model_SimpleCNN.all_test_accuracy), max(model_SimpleCNN_2.all_test_accuracy), max(model_SimpleLinear.all_test_accuracy), max(model_ConvNet.all_test_accuracy), max(model_KKAN_Convolutional_Network.all_test_accuracy)],
#     'Precision': [max(model_SimpleCNN.all_test_precision), max(model_SimpleCNN_2.all_test_precision), max(model_SimpleLinear.all_test_precision), max(model_ConvNet.all_test_precision), max(model_KKAN_Convolutional_Network.all_test_precision)],
#     'Recall': [max(model_SimpleCNN.all_test_recall), max(model_SimpleCNN_2.all_test_recall), max(model_SimpleLinear.all_test_recall), max(model_ConvNet.all_test_recall), max(model_KKAN_Convolutional_Network.all_test_recall)],
#     'F1 Score': [max(model_SimpleCNN.all_test_f1), max(model_SimpleCNN_2.all_test_f1), max(model_SimpleLinear.all_test_f1), max(model_ConvNet.all_test_f1), max(model_KKAN_Convolutional_Network.all_test_f1)],
# }
#
# results_df = pd.DataFrame(data)
# results_df.to_csv('model_performance.csv', index=False)
# print('Results saved to model_performance.csv')

# 定义参数计数函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 创建绘图空间
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
# 绘制测试损失曲线
ax1.plot(all_test_loss_SimpleCNN, label='Loss ConvNet(Small)', color='red')
#ax1.plot(all_test_loss_SimpleLinear, label='Loss 1 Layer & MLP', color='green')
ax1.plot(all_test_loss_SimpleCNN_2, label='Loss ConvNet(Medium)', color='yellow')
ax1.plot(all_test_loss_ConvNet, label='Loss ConvNet (Big)', color='purple')
ax1.plot(all_test_loss_KANC_MLP, label='Loss KANConv & MLP', color='blue')
ax1.plot(all_test_loss_Convs_and_KAN, label='Loss Conv & KAN', color='gray')
ax1.plot(all_test_loss_KKAN_Convolutional_Network, label='Loss KKAN', color='orange')
#ax1.plot(all_test_loss_KAN1, label='Loss 1 Layer KAN', color='black')

ax1.set_title('Loss Test vs Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)
# 绘制模型参数数量与准确度的关系
ax2.scatter(count_parameters(model_SimpleCNN), max(all_test_accuracy_SimpleCNN), color='red', label='ConvNet (Small)')
#ax2.scatter(count_parameters(model_SimpleLinear), max(all_test_accuracy_SimpleLinear), color='green', label='1 Layer MLP')
ax2.scatter(count_parameters(model_SimpleCNN_2), max(all_test_accuracy_SimpleCNN_2), color='yellow', label='ConvNet (Medium)')
ax2.scatter(count_parameters(model_ConvNet), max(all_test_accuracy_ConvNet), color='purple', label='ConvNet (Big)')
ax2.scatter(count_parameters(model_KANC_MLP), max(all_test_accuracy_KANC_MLP), color='blue', label='KANConv & MLP')
ax2.scatter(count_parameters(model_Convs_and_KAN), max(all_test_accuracy_Convs_and_KAN), color='grey', label='Convs & KAN')
ax2.scatter(count_parameters(model_KKAN_Convolutional_Network), max(all_test_accuracy_KKAN_Convolutional_Network), color='orange', label='KKAN')
#ax2.scatter(count_parameters(model_KAN1), max(all_test_accuracy_KAN1), color='black', label='1 Layer KAN')

ax2.set_title('Number of Parameters vs Accuracy')
ax2.set_xlabel('Number of Parameters')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 定义一个函数来突出显示 DataFrame 中的最大值
def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]


# 创建一个 DataFrame 来存储结果
accs = []
precision = []
recall = []
f1s = []
params_counts = []


models = [model_SimpleCNN,model_SimpleCNN_2, model_ConvNet, model_KANC_MLP, model_Convs_and_KAN, model_KKAN_Convolutional_Network]
# all_accuracys = [all_test_accuracy_SimpleLinear, all_test_accuracy_SimpleCNN, all_test_accuracy_ConvNet, all_test_accuracy_KANC_MLP, all_test_accuracy_Convs_and_KAN, all_test_accuracy_KKAN_Convolutional_Network]
# all_precision = [all_test_precision_SimpleLinear, all_test_precision_SimpleCNN, all_test_precision_ConvNet, all_test_precision_KANC_MLP, all_test_precision_Convs_and_KAN, all_test_precision_KKAN_Convolutional_Network]
# all_recall = [all_test_recall_SimpleLinear, all_test_recall_SimpleCNN, all_test_recall_ConvNet, all_test_recall_KANC_MLP, all_test_recall_Convs_and_KAN, all_test_recall_KKAN_Convolutional_Network]
# all_f1s = [all_test_f1_SimpleLinear, all_test_f1_SimpleCNN, all_test_f1_ConvNet, all_test_f1_KANC_MLP, all_test_f1_Convs_and_KAN, all_test_f1_KKAN_Convolutional_Network]


# 记录模型指标
for i, m in enumerate(models):
    index = np.argmax(m.all_test_accuracy)
    params_counts.append(count_parameters(m))
    accs.append(m.all_test_accuracy[index])
    precision.append(m.all_test_precision[index])
    recall.append(m.all_test_recall[index])
    f1s.append(m.all_test_f1[index])

# 创建 DataFrame
df = pd.DataFrame({
    "Test Accuracy": accs,
    "Test Precision": precision,
    "Test Recall": recall,
    "Test F1 Score": f1s,
    "Number of Parameters": params_counts
}, index=[ "ConvNet (Small)","ConvNet (Medium)", "ConvNet (Big)", "KANConv & MLP", "Simple Conv & KAN", "KANConv&KAN"])
# 将 DataFrame 保存为 CSV 文件
df.to_csv('flow_L7.csv', index=False)
df_styled = df.style.apply(highlight_max, subset=df.columns[:], axis=0).format('{:.3f}')

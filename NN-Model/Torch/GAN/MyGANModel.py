##  python 3.9
import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# 选择设备进行训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 超参数定义
latent_dim = 100  # 噪声向量的维度
batch_size = 64  # 批大小
image_size = 28 * 28  # MNIST 图片大小
epochs = 50  # 训练轮数
lr = 0.0002  # 学习率
sample_dir = 'gps'  # 生成图片的保存路径

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 将图片标准化到 [-1, 1]
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_size),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出 [0, 1] 之间的概率
        )

    def forward(self, img):
        return self.model(img)

# 实例化生成器和判别器
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 用于存储损失值
d_losses = []
g_losses = []

# 训练 GAN
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(train_loader):
        batch_size = real_imgs.size(0)

        # 训练判别器：最大化 log(D(x)) + log(1 - D(G(z)))
        real_imgs = real_imgs.view(batch_size, -1).to(device)  # 展平图像
        real_labels = torch.ones(batch_size, 1).to(device)  # 真实样本标签为 1
        fake_labels = torch.zeros(batch_size, 1).to(device)  # 假样本标签为 0

        # 训练判别器（真实样本）
        outputs = discriminator(real_imgs)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # 训练判别器（生成的假样本）
        z = torch.randn(batch_size, latent_dim).to(device)  # 生成噪声
        fake_imgs = generator(z)  # 生成假图像
        outputs = discriminator(fake_imgs.detach())  # 不更新生成器
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # 判别器总损失
        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器：最大化 log(D(G(z)))
        z = torch.randn(batch_size, latent_dim).to(device)  # 生成噪声
        fake_imgs = generator(z)  # 生成假图像
        outputs = discriminator(fake_imgs)  # 判别生成的图像
        g_loss = criterion(outputs, real_labels)  # 生成器的目标是让判别器认为生成的图像是真实的

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # 保存损失
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        if i % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
                  f'D(x): {real_score.mean().item():.4f}, D(G(z)): {fake_score.mean().item():.4f}')

    # 生成并保存图像
    with torch.no_grad():
        z = torch.randn(64, latent_dim).to(device)  # 生成64个噪声样本
        fake_imgs = generator(z).view(-1, 1, 28, 28)  # 调整生成图像的形状
        save_image(fake_imgs, os.path.join(sample_dir, f'fake_images_epoch_{epoch + 1}.png'), nrow=8, normalize=True)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses, label="G Loss")
plt.plot(d_losses, label="D Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

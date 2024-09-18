## python 3.9
import math
import torch
from torch import nn
import WE_Dataset as Dataset
from d2l import torch as d2l
import matplotlib.pyplot as plt

batch_size, max_window_size, num_noise_words = 512, 5, 5  # 设置批次大小，上下文窗口大小，每个正样本生成的噪声词
data_iter, vocab = Dataset.load_data_ptb(batch_size, max_window_size, num_noise_words)  # 生成数据迭代器和词汇表


# 跳元模型的前向传播
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


# 计算二元交叉熵损失
class SigmoidBCELoss(nn.Module):
    # 带掩码的二元交叉熵损失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)


loss = SigmoidBCELoss()

# 构建嵌入层，将词元索引映射为特征向量
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size))


def train(net, data_iter, lr, num_epochs, device="cuda:0"):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs])
    # plt.ion()
    # fig, ax = plt.subplots()
    # ax.set_xlabel('epoch')
    # ax.set_ylabel('loss')
    # ax.set_xlim(0, num_epochs)
    # ax.set_ylim(0, 1)
    # line, = ax.plot([], [])
    # 规范化的损失之和，规范化的损失数
    total_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_samples = 0
        num_batches = len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask) / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            epoch_loss += l.sum().item()
            epoch_samples += l.numel()

        # 记录每个epoch的平均损失
        epoch_avg_loss = epoch_loss/epoch_samples
        total_losses.append(epoch_avg_loss)
        # # 动态更新图像
        # line.set_xdata(range(1, len(total_losses) + 1))
        # line.set_ydata(total_losses)
        # ax.set_ylim(0, max(total_losses))
        # fig.canvas.draw()
        # plt.pause(0.5)

        print(f'Epoch {epoch + 1}, Loss: {epoch_avg_loss:.4f}')

    # 训练结束后绘制损失曲线
    plt.figure()
    plt.plot(range(1, num_epochs + 1), total_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()


# 训练跳元模型
train(net, data_iter, 0.002, 10)


# 输入词汇到训练好的模型，输出在词表中与输入词语义最相似的单词
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k + 1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 删除输入词
        print(f'cosine sim with \'{query_token}\'={float(cos[i]):.3f}: {vocab.to_tokens(i)}')


# 输入词汇到训练好的模型，输出在词表中与输入词语义最相似的单词
get_similar_tokens('man', 10, net[0])

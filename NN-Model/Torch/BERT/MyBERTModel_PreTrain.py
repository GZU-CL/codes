import numpy as np
from BERT_Preprocessing import *

# 加载数据集
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

# 定义BERT结构
net = BERTModel(len(vocab), 128, [128], 128, 256, 2, 2,
                0.2, 1000, 128, 128, 128, 128, 128, 128)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()


# 训练BERT模型
def train_bert(train_iter, net, loss, vocab_size, devices, num_epochs):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()

    # 总损失的保存列表
    mlm_losses, nsp_losses = [], []

    # 记录每个epoch的句子数
    epoch_sentence = 0

    for epoch in range(num_epochs):
        # 每个epoch中的损失记录
        mlm_loss_epoch, nsp_loss_epoch = 0, 0

        # 每个epoch执行多少次
        epoch_times = 0

        for tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                                                  pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            timer.stop()

            # 记录每一轮epoch的 MLM 和 NSP 损失
            mlm_losses.append(mlm_l.item())
            nsp_losses.append(nsp_l.item())

            # 记录当前的损失
            mlm_loss_epoch += mlm_l.item()
            nsp_loss_epoch += nsp_l.item()

            epoch_times += 1
            epoch_sentence += tokens_X.shape[0]
        # 每轮结束后打印平均损失
        print(
            f'epoch {epoch + 1}/{num_epochs}, MLM loss: {mlm_loss_epoch / epoch_times:.3f}, NSP loss: {nsp_loss_epoch / epoch_times:.3f}')

    # print(f'MLM loss {np.mean(mlm_losses):.3f}, NSP loss {np.mean(nsp_losses):.3f}')
    print(f'{epoch_sentence / timer.sum():.1f} sentence pairs/sec')

    # 使用 matplotlib 画出损失曲线
    plot_loss_curves(mlm_losses, nsp_losses)


# 开始训练
train_bert(train_iter, net, loss, len(vocab), devices, 5)


# 用BERT表示文本
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X


tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
encoded_text_crane = encoded_text[:, 2, :]
print("'crane' the first three elements represented by BERT in the sentence 'a crane is flying' are:\n",
      encoded_text_crane[0][:3])

tokens_b = ['a', 'crane', 'driver', 'came']
tokens_c = ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_b, tokens_c)
encoded_pair_crane = encoded_pair[:, 2, :]
print("'crane' the first three elements represented by BERT in the sentence pair 'a crane driver came,he just left' are:\n",
      encoded_pair_crane[0][:3])

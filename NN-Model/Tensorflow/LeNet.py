## python3.9
import tensorflow as tf
import tensorboard
from tensorflow import keras
from keras import Sequential, layers, losses, optimizers, datasets
import datetime

#数据预处理：
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


path = "G:\Learning\program\python\model\LeNet\dataset\mnist.npz"
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(path)
batch_size = 1000
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))  #转化为Dataset对象、随机打散、批训练、数据预处理、复制30份数据
train_db = train_db.shuffle(100000).batch(batch_size).map(preprocess).repeat(30)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(batch_size).map(preprocess)

#创建tensorboard环境：
current_time = datetime.datetime.now().strftime(('%Y%m%d-%H%M%S'))
log_dir = "logs/" + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

#通过Sequnential容器创建LeNet-5
network = Sequential([layers.Conv2D(6, kernel_size=3, strides=1),  #第一个卷积核，6个3x3的卷积核
                      layers.MaxPooling2D(pool_size=2, strides=2),  #最大池化层
                      layers.ReLU(),  #激活函数
                      layers.Conv2D(16, kernel_size=3, strides=1),  #第二个卷积核，16个3x3的卷积核
                      layers.MaxPooling2D(pool_size=2, strides=2),  #最大池化层
                      layers.ReLU(),  #激活函数
                      layers.Flatten(),  #压缩
                      layers.Dense(120, activation='relu'),  #全连接层，120个结点
                      layers.Dense(84, activation='relu'),  #全连接层，84个结点
                      layers.Dense(10, activation='relu')])  #全连接层，10个结点
network.build(input_shape=(1000, 28, 28, 1))  #构建网络模型
print(network.summary)  #统计网络信息

#创建损失函数的类
criteon = losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.RMSprop(0.01)

#训练10个epoch
for step, (x, y) in enumerate(train_db):
    with tf.GradientTape() as tape:
        x = tf.expand_dims(x, axis=3)  #插入通道维度
        out = network(x)  #前向计算，获取10类别的概率分布
        loss = criteon(y, out)  #计算交叉熵损失函数
    grads = tape.gradient(loss, network.trainable_variables)  #自动计算梯度
    optimizer.apply_gradients(zip(grads, network.trainable_variables))
    if step % 100 == 0:
        with summary_writer.as_default():
            tf.summary.scalar('loss', float(loss), step=step)
            print(f"Step {step}, Loss: {loss.numpy()}")
    #计算准确度
    if step % 100 == 0:
        correct, total = 0, 0
        for x, y in test_db:
            x = tf.expand_dims(x, axis=3)
            out = network(x)
            pred = tf.argmax(out, axis=-1)
            y = tf.cast(y, tf.int64)
            y = tf.argmax(y, axis=-1)
            correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
            total += x.shape[0]
        with summary_writer.as_default():
            tf.summary.scalar('acc', float(correct / total), step=step)

tf.saved_model.save(network, 'LeNet')

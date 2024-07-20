##    Python3.9    DataSet:MNIST
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import numpy as np
import datetime

#加载和预处理MNIST数据集
path = "G:\Learning\program\python\model\ResNet34\datasets\mnist.npz"
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path)
train_images = np.expand_dims(train_images, axis=-1).astype('float32') / 255.0
test_images = np.expand_dims(test_images, axis=-1).astype('float32') / 255.0

train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

#残差块定义
def resnet_block(inputs, filters, kernel_size=(3,3), stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or inputs.shape[-1] != filters:
        inputs = layers.Conv2D(filters, kernel_size=(1,1), strides=stride, padding='same')(inputs)
        inputs = layers.BatchNormalization()(inputs)

    x = layers.add([x, inputs])
    x = layers.ReLU()(x)
    return x

#创建tensorboard环境：
log_dir = "logs/" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

#构建ResNet34模型
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

for filters, blocks, stride in zip([64, 128, 256, 512], [3, 4, 6, 3], [1, 2, 2, 2]):
    for block in range(blocks):
        if block == 0:
            x = resnet_block(x, filters, stride=stride)
        else:
            x = resnet_block(x, filters, stride=1)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation='softmax')(x)

#创建模型
model = keras.Model(inputs, outputs)

#编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#训练模型

model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_split=0.2, callbacks=[tensorboard_callback])

#评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

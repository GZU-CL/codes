##    Python3.9    DataSet:MNIST
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import keras
import datetime

# 加载和预处理MNIST数据集
path="G:\Learning\program\python\model\GoogleNet\datasets\mnist.npz"
(x_train, y_train), (x_test, y_test) = mnist.load_data(path)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 定义Inception模块
def inception_module(x, filters):
    # Branch 1x1
    branch1x1 = layers.Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    # Branch 1x1 -> 3x3
    branch1x1_3x3 = layers.Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    branch1x1_3x3 = layers.Conv2D(filters[2], (3, 3), padding='same', activation='relu')(branch1x1_3x3)

    # Branch 1x1 -> 5x5
    branch1x1_5x5 = layers.Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    branch1x1_5x5 = layers.Conv2D(filters[4], (5, 5), padding='same', activation='relu')(branch1x1_5x5)

    # Branch Pooling -> 1x1
    branch_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = layers.Conv2D(filters[5], (1, 1), padding='same', activation='relu')(branch_pool)

    # Concatenate all branches
    outputs = layers.concatenate([branch1x1, branch1x1_3x3, branch1x1_5x5, branch_pool], axis=-1)
    return outputs

#创建tensorboard环境：
log_dir = "logs/" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

# 构建GoogleNet模型
def build_googlenet_model():
    inputs = layers.Input(shape=(28, 28, 1))

    x = layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add multiple Inception modules
    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = inception_module(x, [64, 128, 128, 32, 32, 32])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs, x)
    return model

# 编译和训练模型
model = build_googlenet_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2,callbacks=[tensorboard_callback])

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

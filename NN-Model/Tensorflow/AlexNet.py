##  python3.9
import tensorflow as tf
from tensorflow import keras
from keras import layers,models,datasets
import tensorboard
import datetime

#数据预处理
def preprocess(train, test):
    train = train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
    test = test.reshape((10000, 28, 28, 1)).astype('float32') / 255.0
    return train, test
    
path="G:\Learning\program\python\model\AlexNet\dataset\mnist.npz"
(train_images, train_labels), (test_images, test_labels)=datasets.mnist.load_data(path)

train_images,test_images=preprocess(train_images,test_images)
train_labels=keras.utils.to_categorical(train_labels,10)
test_labels=keras.utils.to_categorical(test_labels,10)

#创建tensorboard环境：
log_dir="logs/"+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback=keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1,update_freq='batch')

#构建模型
model=models.Sequential([layers.Conv2D(32,(3,3),strides=1,activation='relu',input_shape=(28,28,1)),
                         layers.BatchNormalization(),
                         layers.MaxPooling2D((2,2),strides=2),
                         layers.Conv2D(64,(3,3),padding='same',activation='relu'),
                         layers.BatchNormalization(),
                         layers.MaxPooling2D((2,2),strides=2),
                         layers.Conv2D(128,(3,3),padding='same',activation='relu'),
                         layers.Conv2D(128,(3,3),padding='same',activation='relu'),
                         layers.Conv2D(64,(3,3),padding='same',activation='relu'),
                         layers.MaxPooling2D((2,2),strides=2),
                         layers.Flatten(),
                         layers.Dense(512,activation='relu'),
                         layers.Dropout(0.5),
                         layers.Dense(512,activation='relu'),
                         layers.Dropout(0.5),
                         layers.Dense(10,activation='softmax')])

#编译模型
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#训练模型
model.fit(train_images,train_labels,epochs=5,batch_size=128,validation_split=0.2,callbacks=[tensorboard_callback])

#评估模型
test_loss,test_acc=model.evaluate(test_images,test_labels)
print(f'Test accuracy: {test_acc}')


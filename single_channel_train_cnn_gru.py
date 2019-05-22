# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 08:57:02 2018

@author: wen
"""

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import cv2
#from PIL import Image
#from PIL import ImageEnhance
from keras.utils import np_utils
from skimage.transform import resize
from print_loss_accuracy import LossHistory
#from keras.preprocessing.image import load_img, img_to_array
#from sklearn.model_selection import train_test_split

#读入图片名称总文档
data = pd.read_csv('train_cnngru.csv')
data.head(5)

#读取图片信息
path = './train_image_cnngru/'
X = []
count = 1
b = []
for img_name in data.Image_name:
    #image = Image.open(path + img_name)
    #截取主体图像
    #box = (0, 180, 1500, 1080)
    #image = image.crop(box)
    #灰度图
    #image = image.convert('L')
    #亮度增强
    #enh_bri = ImageEnhance.Brightness(image)
    #brightness = 1.5
    #image_brightened = enh_bri.enhance(brightness)
    #对比度增强
    #enh_con = ImageEnhance.Contrast(image_brightened)
    #contrast = 1.5
    #image_contrasted = enh_con.enhance(contrast)
    #image = np.array(image_contrasted)
    #image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    img = cv2.imread(path + img_name, 0)
    equ = cv2.equalizeHist(img) #使图片的像素趋近于高斯分布 
    image = cv2.GaussianBlur(equ, (5, 5), 0)
    a = resize(image, preserve_range = True, output_shape = (224, 224, 1)).astype(int)
    a = a / 255 #数据归一化
    #ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    b.append(a)
    if count% 5 == 0:
        X.append(b)
        b = []
    count += 1

X = np.array(X)

#独热编码
y = data.Label
dummy_y = np_utils.to_categorical(y)    #one hot encoding Classes
y = []
for i in range(0,dummy_y.shape[0]-2,5):
    y.append(dummy_y[i:i+5])
y = np.array(y)
#数据集分离
#X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2)
X_train = X[250:]
#增加一个维度
#X_train = np.expand_dims(X_train, axis=0) 
y_train = y[250:]
X_test = X[:250]
##增加一个维度
#X_test = np.expand_dims(X_test, axis=0) 
y_test = y[:250]

#搭建模型
#预处理
from keras.models import Sequential, Model  #快速开始序贯模型
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D, GRU, Input
#Dence:全连接层（对上一层的神经元进行全部连接，实现特征的非线性组合）
#Flatten返回一个折叠的一维数组
from keras.optimizers import SGD
from keras.layers.wrappers import TimeDistributed

def build_model(categories):
    input_shape = Input((224, 224, 1))
    cnn = Conv2D(filters=32,  kernel_size=3, padding='same', activation='relu')(input_shape)
    cnn = MaxPool2D(pool_size= 2, strides = 2)(cnn)
    cnn = Conv2D(filters=64,  kernel_size=3, padding='same', activation='relu')(cnn)
    cnn = MaxPool2D(pool_size= 2, strides = 2)(cnn)
    cnn = Conv2D(filters=128,  kernel_size=3, padding='same', activation='relu')(cnn)
    cnn = MaxPool2D(pool_size= 2, strides = 2)(cnn)
    cnn = Conv2D(filters=256,  kernel_size=3, padding='same', activation='relu')(cnn)
    cnn = MaxPool2D(pool_size= 2, strides = 2)(cnn)
    cnn = Conv2D(filters=512,  kernel_size=3, padding='same', activation='relu')(cnn)
    cnn = MaxPool2D(pool_size= 2, strides = 2)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(512)(cnn)
    cnn = Dropout(0.5)(cnn)
    x = Model(inputs = input_shape, outputs = cnn)
    x.summary()
    input_shape = (5, 224, 224, 1) # (seq_len, width, height, channel)
    model = Sequential()
    model.add(TimeDistributed(x, input_shape=input_shape))
    model.add(GRU(512, kernel_initializer='orthogonal', bias_initializer='ones', dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    #model.add(GRU(128, kernel_initializer='orthogonal', bias_initializer='ones', dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    model.add(Dense(categories, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy',optimizer = SGD(lr=0.01, momentum=0.9, clipnorm=1., clipvalue=0.5),metrics=['accuracy'])
    return model
model = build_model(4)
model.summary()
#编译模型
#SGD：随机梯度下降，随机梯度下降每次的权重更新只利用数据集中的一个样本来完成，momentum表示动量，动量项超参数γ<1一般是小于等于0.9
#加入动量可以使得梯度方向不变的维度上速度变快，梯度方向有所改变的维度上的更新速度变慢，这样就可以加快收敛并减小震荡。
#model.compile(optimizer = SGD(lr=1e-2, momentum = 0.9), loss='categorical_crossentropy', metrics = ['accuracy'])
#model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])
#记录训练的准确率与损失值
history = LossHistory()
#训练模型 
model.fit(X_train, y_train, batch_size = 16, epochs = 100, validation_data = (X_test, y_test), callbacks=[history])
#打印准确率与损失值的变化曲线
history.loss_plot('epoch')

#保存模型
model.save('model_cnn_gru1.h5') 

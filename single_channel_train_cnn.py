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
from sklearn.model_selection import train_test_split
from print_loss_accuracy import LossHistory

#读入图片名称总文档
data = pd.read_csv('train_cnn.csv')
data.head(5)

#读取图片信息
path = './train_image_cnn/'
X = []
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
    #ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    X.append(image)

#独热编码
y = data.Class
dummy_y = np_utils.to_categorical(y)    #one hot encoding Classes
#输入到自搭建的单层vgg16：224x224x1
image_ = []
for i in range(len(X)):
#    X[i] = np.array(X[i])
    a = resize(X[i], preserve_range = True, output_shape = (224, 224, 1)).astype(int)
    a = a / 255 #数据归一化
    image_.append(a)
X = np.array(image_)

#数据集分离
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2)

#搭建模型
#预处理
from keras.models import Sequential  #快速开始序贯模型
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D 
#Dence:全连接层（对上一层的神经元进行全部连接，实现特征的非线性组合）
#Flatten返回一个折叠的一维数组
#from keras.optimizers import SGD, Adam

#搭建卷积神经网络
#搭建全连接层
model = Sequential()
#Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
model.add(Conv2D(input_shape=(224, 224, 1), filters=32,  kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size= 2, strides = 2))

model.add(Conv2D(filters=64,  kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size= 2, strides = 2))

model.add(Conv2D(filters=128,  kernel_size=3, padding='same', activation='relu'))
#model.add(Conv2D(filters=128,  kernel_size=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size= 2, strides = 2))

model.add(Conv2D(filters=256,  kernel_size=3, padding='same', activation='relu'))
#model.add(Conv2D(filters=256,  kernel_size=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size= 2, strides = 2))

model.add(Conv2D(filters=512,  kernel_size=3, padding='same', activation='relu'))
#model.add(Conv2D(filters=512,  kernel_size=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size= 2, strides = 2))

model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(4, activation = 'softmax'))
model.summary()

#编译模型
#SGD：随机梯度下降，随机梯度下降每次的权重更新只利用数据集中的一个样本来完成，momentum表示动量，动量项超参数γ<1一般是小于等于0.9
#加入动量可以使得梯度方向不变的维度上速度变快，梯度方向有所改变的维度上的更新速度变慢，这样就可以加快收敛并减小震荡。
#model.compile(optimizer = SGD(lr=1e-2, momentum = 0.9), loss='categorical_crossentropy', metrics = ['accuracy'])
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])
#训练模型
history = LossHistory() 
model.fit(X_train, y_train, batch_size = 16, epochs = 100, validation_data = (X_test, y_test), callbacks=[history])
history.loss_plot('epoch')

#保存模型
model.save('model_cnn_short_threshold.h5') 

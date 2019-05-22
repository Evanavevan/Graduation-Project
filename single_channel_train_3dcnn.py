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
from sklearn.model_selection import train_test_split

#读入图片名称总文档
data = pd.read_csv('train_cnngru.csv')
data.head(5)

#读取图片数据集
path = './train_image_cnngru/'
X = []
count = 1
b = []
for img_name in data.Image_name:
    img = cv2.imread(path + img_name, 0)
    equ = cv2.equalizeHist(img)     #使图片的像素趋近于高斯分布 
    image = cv2.GaussianBlur(equ, (5, 5), 0)
    a = resize(image, preserve_range = True, output_shape = (112, 112, 1)).astype(int)
    a = a / 255 #数据归一化
    #ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    b.append(a)
    if count% 5 == 0:
        X.append(b)
        b = []
    count += 1

X = np.array(X)

#独热编码
y = data[data.Class.notnull()].Class
dummy_y = np_utils.to_categorical(y)    #one hot encoding Classes
#数据集分离
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2, random_state = 12)

#X_train = X[250:]
#增加一个维度
#X_train = np.expand_dims(X_train, axis=0) 
#y_train = dummy_y[250:]
#X_test = X[:250]
##增加一个维度
#X_test = np.expand_dims(X_test, axis=0) 
#y_test = dummy_y[:250]

#搭建模型
from keras.models import Sequential  #快速开始序贯模型
from keras.layers import Dense, Flatten, Dropout, Conv3D, MaxPool3D
#Dence:全连接层（对上一层的神经元进行全部连接，实现特征的非线性组合）
#Flatten返回一个折叠的一维数组
from keras.optimizers import SGD

def build_model(categories):
    model = Sequential()
    model.add(Conv3D(input_shape=(5, 112, 112, 1), filters=16, kernel_size=3, 
                     strides=1, padding='same', activation='relu'))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=(1,2,2)))
    model.add(Conv3D(filters=32, kernel_size=3, strides=1, padding='same', 
                     activation='relu'))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=(1,2,2)))
    model.add(Conv3D(filters=64, kernel_size=3, strides=1, padding='same', 
                     activation='relu'))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=(1,2,2)))
    model.add(Conv3D(filters=128, kernel_size=3, strides=1, padding='same', 
                     activation='relu'))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=(1,2,2)))
    
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(categories, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], 
                  optimizer = SGD(lr=0.01, momentum=0.9, clipnorm=1., clipvalue=0.5),
                  )
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
model.save('model_3dcnn.h5') 

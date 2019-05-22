# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:02:17 2019

@author: wen
"""

import numpy as np
import cv2
from skimage.transform import resize
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score

#载入测试数据
path = './test_image_cnngru/'
test = pd.read_csv("test_cnngru.csv")
#保存预测结果
image = []
b = []
d = []
array = []
c2d = []
c3d = []
count = 1

#预测测试集
for image_name in test[test.Image_name3.notnull()].Image_name3:
    #导入数据
    img = cv2.imread(path + image_name, 0)             
    equ = cv2.equalizeHist(img) #使图片的像素趋近于高斯分布 
    img = cv2.GaussianBlur(equ, (5, 5), 0)
    a = resize(img, preserve_range = True, output_shape = (112, 112, 1)).astype(int)
    c = resize(img, preserve_range = True, output_shape = (224, 224, 1)).astype(int)
    a = a / 255 #数据归一化 
    c = c / 255
    c = np.expand_dims(c, 0)
    b.append(a)
    d.append(c)
    if count % 5 == 0:
        #增加一个维度
        b = np.expand_dims(b, 0)
        c3d.append(b)
        c2d.append(d)
        b = []
        d = []
    count += 1

#载入模型
model = load_model('model_3dcnn.h5')
#测试模型
result = []
for i in c3d:
    pred = model.predict(i)
    result.append(np.argmax(pred))

model1 = load_model('model_cnn_short_threshold.h5')
for i in range(len(result)):
    pred = model1.predict(c2d[i][2])
    result1 = np.argmax(pred)
    if result1 == 3 and result[i] != 3:
            result[i] = 3

label = test[test.Class3.notnull()].Class3
label = np.array(label)

#输出测试集的准确率  
accuracy = accuracy_score(np.array(label), np.array(result))
print("The accuracy of the model in test is:", accuracy)

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 09:52:29 2019

@author: wen
"""

import numpy as np
import cv2
from skimage.transform import resize
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter

#载入测试数据
path = './test_image_cnngru/'
test = pd.read_csv("test_cnngru.csv")
#保存预测结果
image = []
b = []
array = []
c2d = []
c3d = []
count = 1

#混合预测测试集，1行为采用2dcnn
for image_name in test[test.Image_name1.notnull()].Image_name1:
    #导入数据
    #img = Image.open(path + 'frame%d.jpg' % i).convert('L')
    img = cv2.imread(path + image_name, 0)             
    equ = cv2.equalizeHist(img) #使图片的像素趋近于高斯分布 
    img = cv2.GaussianBlur(equ, (5, 5), 0)
    #ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    a = resize(img, preserve_range = True, output_shape = (112, 112, 1)).astype(int)
    c = resize(img, preserve_range = True, output_shape = (224, 224, 1)).astype(int)
    img = a / 255 #数据归一化  
    image = c / 255
    b.append(img)
    
    image = np.expand_dims(image, 0)
    array.append(image)
    
    if count% 5 == 0:
        #增加一个维度
        b = np.expand_dims(b, 0)
        c3d.append(b)
        c2d.append(array)
        b = []
        array = []
    count += 1
# np.array(c2d).shape
#2dcnn预测
#载入模型
model = load_model('model_cnn_short_threshold.h5')
#测试模型
result1 = []
for i in c2d:
    temp = []
    for j in i:
        pred = model.predict(j)
        temp.append(np.argmax(pred))
    # Counter(temp).most_common(1):返回出现频率最高的一个数
    temp = Counter(temp).most_common(1)[0][0] 
    result1.append(temp)

#3dcnn预测
#载入模型
model = load_model('model_3dcnn.h5')
#测试模型
result = []
for i in c3d:
    pred = model.predict(i)
    result.append(np.argmax(pred))

#对比，如果2dCNN为1的而3dCNN对应位置不为1的，则相应位置将会变为1
for i in range(len(result)):
    if result1[i] == 1 and result[i] != 1:
        result[i] = 1
    if result1[i] == 2 and result[i] != 2:
        result[i] = 2

label = test[test.Class1.notnull()].Class1

#输出测试集的准确率  
accuracy = accuracy_score(np.array(label), np.array(result))
print("The accuracy of the model in test is:", accuracy)





#混合预测整个测试集，1行为采用2dcnn
for image_name in test.Image_name:
    #导入数据
    #img = Image.open(path + 'frame%d.jpg' % i).convert('L')
    img = cv2.imread(path + image_name, 0)             
    equ = cv2.equalizeHist(img) #使图片的像素趋近于高斯分布 
    img = cv2.GaussianBlur(equ, (5, 5), 0)
    #ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    a = resize(img, preserve_range = True, output_shape = (112, 112, 1)).astype(int)
    c = resize(img, preserve_range = True, output_shape = (224, 224, 1)).astype(int)
    img = a / 255 #数据归一化  
    image = c / 255
    b.append(img)
    
    image = np.expand_dims(image, 0)
    array.append(image)
    
    if count% 5 == 0:
        #增加一个维度
        b = np.expand_dims(b, 0)
        c3d.append(b)
        c2d.append(array)
        b = []
        array = []
    count += 1
# np.array(c2d).shape
#2dcnn预测
#载入模型
model = load_model('model_cnn_short_threshold.h5')
#测试模型
result1 = []
for i in c2d:
    temp = []
    for j in i:
        pred = model.predict(j)
        temp.append(np.argmax(pred))
    # Counter(temp).most_common(1):返回出现频率最高的一个数
    temp = Counter(temp).most_common(1)[0][0] 
    result1.append(temp)

#3dcnn预测
#载入模型
model = load_model('model_3dcnn.h5')
#测试模型
result = []
for i in c3d:
    pred = model.predict(i)
    result.append(np.argmax(pred))

#对比，如果2dCNN为1的而3dCNN对应位置不为1的，则相应位置将会变为1
for i in range(len(result)):
    if result1[i] == 1 and result[i] != 1:
        result[i] = 1
    if result1[i] == 2 and result[i] != 2:
        result[i] = 2

label = test[test.Class.notnull()].Class

#输出测试集的准确率  
accuracy = accuracy_score(np.array(label), np.array(result))
print("The accuracy of the model in test is:", accuracy)
# The accuracy of the model in test is: 0.9210526315789473
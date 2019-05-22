# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:03:52 2018

@author: wen
"""

import numpy as np
import cv2
from skimage.transform import resize
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg 
#from PIL import Image
#from PIL import ImageEnhance

#图片处理
def presolve(image):
    #截取主体图像
    #box = (0, 180, 1500, 1080)
    #image = image.crop(box)
    #灰度图
    #image = image.convert('L')
    #亮度增强
    #enh_bri = ImageEnhance.Brightness(image)
    #brightness = 1.5
    #image_brightened = enh_bri.enhance(brightness)
    ##对比度增强
    #enh_con = ImageEnhance.Contrast(image_brightened)
    #contrast = 1.5
    #image_contrasted = enh_con.enhance(contrast)
    #image = np.array(image_contrasted)
    equ = cv2.equalizeHist(image) #使图片的像素趋近于高斯分布 
    image = cv2.GaussianBlur(equ, (5, 5), 0)
    #ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    a = resize(image, preserve_range = True, output_shape = (224, 224, 1)).astype(int)
    image = a / 255 #数据归一化
    #增加一个维度
    img = np.expand_dims(image, 0)
    return img
#可视化展示 
def plot_images_labels_prediction(image, label, prediction, idx, label_dict, count, num=10):
    fig = plt.gcf()
    fig.set_size_inches(16, 5)
    if num > 10:
        num = 10
    for i in range(num):
        ax = plt.subplot(2, 5, 1+i)
        ax.imshow(image[idx])
        
        title = 'true:' + label_dict[label[idx]]
        if len(prediction) > 0:
            title += '\n'+'pred:' + label_dict[prediction[idx]]
            
        ax.set_title(title, fontsize = 10)
        idx += 1
    plt.savefig('cnnshow%d' %count + '.png')
    plt.show()

#构建标签
#label = np.array(['safe_driving', 'hands_off_the_wheel', 'playing_cellphone', 'harassing_others'])
#载入模型
model = load_model('model_cnn_short_threshold.h5')
#载入测试数据
path = './test_image_cnn/'
test = pd.read_csv("test_cnn.csv")
#保存预测结果
result = []
image = []

#预测测试集
for image_name in test[test.Class == 3].Image_name:
    #导入数据
    #img = Image.open(path + 'frame%d.jpg' % i).convert('L')
    img = cv2.imread(path + image_name, 0)
    #image.append(mpimg.imread(path + image_name))             
    img = presolve(img)
    #测试模型
    pred = model.predict(img)
    result.append(np.argmax(pred))
    #print(label[np.argmax(pred)])

#可视化预测结果
#定义标签字典，每一个数字所代表的的图像类别名称
label_dict = {0:'safe_driving', 1:'hands_off_the_wheel', 2:'playing_cellphone', 3:'harassing_others' }
index = [7, 37, 44, 82, 122, 132, 140, 166, 187, 249]
count = 1
plot_images_labels_prediction(image, test.Class, result, 0, label_dict, count)
count = 2
plot_images_labels_prediction(image, test.Class, result, 35, label_dict, count)
count = 3
plot_images_labels_prediction(image, test.Class, result, 80, label_dict, count)
count = 4
plot_images_labels_prediction(image, test.Class, result, 240, label_dict, count)
count = 5
plot_images_labels_prediction(image, test.Class, result, 160, label_dict, count)
#输出测试集的准确率    
accuracy = accuracy_score(np.array(test[test.Class == 3].Class), np.array(result))
print("The accuracy of the model in test is:", accuracy)

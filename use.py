# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 08:49:23 2019

@author: wen
"""

import numpy as np
import cv2
from skimage.transform import resize
from keras.models import load_model
from collections import Counter
import threading
import tensorflow as tf

class Gestrue(object):
    def __init__(self):
        self.model1 = load_model('model_cnn_short_threshold.h5')
        self.model2 = load_model('model_3dcnn.h5')
        self.label = {0:'safe driving', 1:'hands off the wheel', 2:'playing cellphone', 3:'harassing others'}
        self.image_size1 = 224
        self.image_size2 = 112
        self.result = -1
        # 每隔3帧读取一次图像
        self.timeF = 6
        # 保存回话视图，以便模型全局调用
        self.graph = tf.get_default_graph()
    
    def camera(self):
        count = 0
        image = []
        cam = cv2.VideoCapture(1)
        while cam.isOpened(): 
            success, img = cam.read()
            if(success != True):
                break
            count += 1
            if count % self.timeF == 0:
                image.append(img)
                count = 0
            if len(image) == 5:
                # self.predict(image)
                # 创立线程，防止视频卡顿
                t = threading.Thread(target=Gestrue.predict, args=(self, image))
                t.start()
                # 等待上一线程结束
                # t.join()
                image = []
            if self.result != -1:
                # 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                cv2.putText(img, self.label[self.result], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  #显示名字
            cv2.imshow('img', img)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()
        
    def predict(self, image):
        image_1 = []
        image_2 = []
        
        for img in image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(img) #使图片的像素趋近于高斯分布 
            img = cv2.GaussianBlur(equ, (5, 5), 0)
            a = resize(img, preserve_range = True, output_shape = (self.image_size1, self.image_size1, 1)).astype(int)
            b = resize(img, preserve_range = True, output_shape = (self.image_size2, self.image_size2, 1)).astype(int)
            a = a / 255 #数据归一化  
            b = b / 255
            # 2dcnn的输入为（1，224，224，1）
            a = np.expand_dims(a, 0)
            image_1.append(a)
            image_2.append(b)
        image_2 = np.expand_dims(image_2, 0)
        
        # 调用线程要加上这句话，否则报错
        with self.graph.as_default():
            # 2dcnn
            temp = []
            for j in image_1:
                pred = self.model1.predict(j)
                temp.append(np.argmax(pred))
            # Counter(temp).most_common(1):返回出现频率最高的一个数
            result1 = Counter(temp).most_common(1)[0][0] 
                
            # 3dcnn
            result = np.argmax(self.model2.predict(image_2))
            
        #对比，如果2dCNN为1的而3dCNN对应位置不为1的，则相应位置将会变为1
        if result1 == 1 and result != 1:
            self.result = 1
        if result1 == 2 and result != 2:
            self.result = 2

if __name__ == "__main__":
   Gestrue().camera()         
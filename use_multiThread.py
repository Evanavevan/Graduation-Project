# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 08:49:23 2019

@author: wen
"""

import numpy as np
import cv2
from skimage.transform import resize
from keras.models import load_model
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
        self.result1 = -1
        # 每隔3帧读取一次图像
        self.timeF = 6
        # 保存回话视图，以便模型全局调用
        self.graph = tf.get_default_graph()
    
    def camera(self):
        count = 0
        image = []
        cam = cv2.VideoCapture(1)
        # 录制视频
        # 视频编码
        """
        FourCC全称Four-Character Codes，代表四字符代码 (four character code), 它是一个32位的标示符，
        其实就是typedef unsigned int FOURCC;是一种独立标示视频数据流格式的四字符代码。
        因此cv2.VideoWriter_fourcc()函数的作用是输入四个字符代码即可得到对应的视频编码器。
        #cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi
        #cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'),该参数是MPEG-1编码类型，文件名后缀为.avi
        #cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),该参数是MPEG-4编码类型，文件名后缀为.avi
        #cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'),该参数是Ogg Vorbis,文件名后缀为.ogv
        #cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),该参数是Flash视频，文件名后缀为.flv
        """
        # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        # outputPath = 'output1.avi'
        # 30fps, 1920*1080 size
        # out = cv2.VideoWriter(outputPath, fourcc, 30.0, (1920,1080))
        while cam.isOpened(): 
            success, img = cam.read()
            if not success:
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
                image = []
            if self.result != -1:
                # 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                cv2.putText(img, self.label[self.result], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  #显示名字
            # out.write(img)
            cv2.imshow('img', img)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()
        
    def predict(self, image):
        self.image_1 = []
        self.image_2 = []
        
        for img in image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(img) #使图片的像素趋近于高斯分布 
            img = cv2.GaussianBlur(equ, (5, 5), 0)
            t1 = threading.Thread(target=Gestrue.resize_img1, args=(self, img))
            t2 = threading.Thread(target=Gestrue.resize_img2, args=(self, img))
            t1.start()
            t2.start()
            # 等待上一线程结束
            t1.join()
            t2.join()            
        self.image_2 = np.expand_dims(self.image_2, 0)
        
        # 预测每种行为的概率
        t1 = threading.Thread(target=Gestrue.cnn_3d, args=(self, ))
        t2 = threading.Thread(target=Gestrue.cnn_2d, args=(self, ))
        t1.start()
        t2.start()
        # 等待上一线程结束
        t1.join()
        t2.join()
        
        #对比，如果2dCNN为1的而3dCNN对应位置不为1的，则相应位置将会变为1
        if self.result1 == 1 and self.result != 1:
            self.result = 1
            
    def resize_img1(self, img):
        a = resize(img, preserve_range = True, output_shape = (self.image_size1, self.image_size1, 1)).astype(int)
        a = a / 255 #数据归一化
        # 2dcnn的输入为（1，224，224，1）
        a = np.expand_dims(a, 0)
        self.image_1.append(a)
    
    def resize_img2(self, img):
        b = resize(img, preserve_range = True, output_shape = (self.image_size2, self.image_size2, 1)).astype(int)
        b = b / 255
        self.image_2.append(b)
        
    def cnn_3d(self):
        # 调用线程要加上这句话，否则报错
        with self.graph.as_default():
            # 3dcnn
            self.result = np.argmax(self.model2.predict(self.image_2))
        
    def cnn_2d(self):
        with self.graph.as_default():
            # 2dcnn
            pred = self.model1.predict(self.image_1[2])
            self.result1 = np.argmax(pred)
           
               
if __name__ == "__main__":
   Gestrue().camera()         
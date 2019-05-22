# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:59:55 2018

@author: wen
"""
import cv2
import matplotlib.pyplot as plt

#读取视频并从中提取帧，将其保存为图像
count = 1281
videoFile = 'video_7.mp4'
#VideoCapture()中参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
cap = cv2.VideoCapture(videoFile)
cap.isOpened()
#frameRate = cap.get(5)
#获取视频属性
#cap.get( propId )：propId 的值为 0 - 18，19个值并不是每个都可以进行修改
"""
参数	                        值	功能/意义
CV_CAP_PROP_POS_MSEC	    0	视频文件的当前位置（以毫秒为单位）或视频捕获时间戳。
CV_CAP_PROP_POS_FRAMES	    1	基于0的索引将被解码/捕获下一帧。
CV_CAP_PROP_POS_AVI_RATIO	2	视频文件的相对位置：0 - 电影的开始，电影的1 - 结束。
CV_CAP_PROP_FRAME_WIDTH	    3	视频每一帧的宽。
CV_CAP_PROP_FRAME_HEIGHT	4	视频每一帧的高。
CV_CAP_PROP_FPS	            5	视频的帧速。
CV_CAP_PROP_FOURCC	        6	4个字符表示的视频编码器格式。
CV_CAP_PROP_FRAME_COUNT	    7	视频的帧数。
CV_CAP_PROP_FORMAT	        8	byretrieve（）返回的Mat对象的格式。
CV_CAP_PROP_MODE	        9	指示当前捕获模式的后端特定值。
CV_CAP_PROP_BRIGHTNESS	    10	图像的亮度（仅适用于相机）。
CV_CAP_PROP_CONTRAST	    11	图像对比度（仅适用于相机）。
CV_CAP_PROP_SATURATION	    12	图像的饱和度（仅适用于相机）。
CV_CAP_PROP_HUE	            13	图像的色相（仅适用于相机）。
CV_CAP_PROP_GAIN	        14	图像的增益（仅适用于相机）。
CV_CAP_PROP_EXPOSURE	    15	曝光（仅适用于相机）。
CV_CAP_PROP_CONVERT_RGB	    16	表示图像是否应转换为RGB的布尔标志。
CV_CAP_PROP_WHITE_BALANCE	17	目前不支持
CV_CAP_PROP_RECTIFICATION	18	立体摄像机的整流标志（注意：只有当前支持DC1394 v 2.x后端）
"""
"""
函数名：cv2.isOpened()
功  能：返回一个布尔值（ True / False ），检查是否初始化成功，成功返回 True
返回值：布尔值
"""
c = 1
timeF = 3
#path = './1/'
while(cap.isOpened()):
#    frameId = cap.get(1)
    #cap.read()按帧读取视频，ret,frame是获cap.read()方法的两个返回值。
    #其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。
    #frame就是每一帧的图像，是个三维矩阵。
    ret, frame = cap.read()
    if(ret != True):
        break
    #math.floor() 返回数字的下舍整数。
    if (c % timeF == 0):
#    if(frameId % math.floor(frameRate) == 0):
        filename = "frame%d.jpg" % count; count += 1
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite()：保存图像，第一个参数是保存的路径及文件名，第二个是图像矩阵
        cv2.imwrite(filename, frame)
#        cv2.imwrite(filename, gray)
    c += 1
cap.release()
print("Done!")

img = plt.imread('frame0.jpg')
plt.axis('off')
plt.imshow(img)



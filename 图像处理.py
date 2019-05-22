# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:13:09 2018

@author: wen
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance
from skimage.transform import resize

#图像去噪
"""
cv2.fastNlMeansDenoising（） - 使用单个灰度图像
cv2.fastNlMeansDenoisingColored（） - 使用彩色图像
h：参数决定滤波器强度。较高的h值可以更好地消除噪声，但也会删除图像的细节 (10 is ok)
hForColorComponents：与h相同，但仅适用于彩色图像。 （通常与h相同）
templateWindowSize：应该是奇数。 （recommended 7）
searchWindowSize：应该是奇数。 （recommended 21）
"""
path = './image/'
img = cv2.imread(path + 'frame405.jpg')
plt.axis('off')
plt.imshow(img)
dst = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
plt.imshow(dst)

"""
cv2.fastNlMeansDenoisingMulti（） - 用于在短时间内捕获的图像序列（灰度图像）
cv2.fastNlMeansDenoisingColoredMulti（） - 与上面相同，但用于彩色图像。
多数应用于视频提取出来的照片进行处理
第一个参数是嘈杂帧的列表。 
第二个参数imgToDenoiseIndex指定我们需要去噪的帧，因为我们在输入列表中传递了frame的索引。 
第三个是temporalWindowSize，它指定了用于去噪的附近帧的数量。 在这种情况下，使用总共temporalWindowSize帧，其中中心帧是要去噪的帧
"""

#图像增强
#原始图像
path = './image/'
image = Image.open(path + 'frame604.jpg')
image.show()
#图像裁剪
box = (0, 180, 1500, 1080)
image = image.crop(box)
#转化为灰度图
image = image.convert('L')
#手绘体
image = np.array(image).astype('float')

depth = 10.
grad = np.gradient(image)
grad_x, grad_y = grad
 
grad_x = grad_x*depth/100.
grad_y = grad_y*depth/100.
A = np.sqrt(grad_y**2+grad_y**2+1)
uni_x = grad_x/A
uni_y = grad_y/A
uni_z = 1./A
 
vec_el = np.pi/2.2
vec_az = np.pi/4
dx = np.cos(vec_el)*np.cos(vec_az)
dy = np.cos(vec_el)*np.sin(vec_az)
dz = np.sin(vec_el)
 
b = 255*(dx*uni_x+dy*uni_y+dz*uni_z)
b = b.clip(0, 225)

im = Image.fromarray(b.astype('uint8'))
im.show()
 
#亮度增强
enh_bri = ImageEnhance.Brightness(image)
brightness = 1.5
image_brightened = enh_bri.enhance(brightness)
image_brightened.show()
 
#色度增强
enh_col = ImageEnhance.Color(image)
color = 1.5
image_colored = enh_col.enhance(color)
image_colored.show()
 
#对比度增强
enh_con = ImageEnhance.Contrast(image)
contrast = 2.0
image_contrasted = enh_con.enhance(contrast)
image_contrasted.show()
 
#锐度增强
enh_sha = ImageEnhance.Sharpness(image)
sharpness = 3.0
image_sharped = enh_sha.enhance(sharpness)
image_sharped.show()

#边缘检测
img = cv2.imread(path + 'frame569.jpg', 0)
"""
cv2.GaussianBlur()
InputArray src: 输入图像，可以是Mat类型，图像深度为CV_8U、CV_16U、CV_16S、CV_32F、CV_64F。 
OutputArray dst: 输出图像，与输入图像有相同的类型和尺寸。 
Size ksize: 高斯内核大小，这个尺寸与前面两个滤波kernel尺寸不同，ksize.width和ksize.height可以不相同但是这两个值必须为正奇数，如果这两个值为0，他们的值将由sigma计算。 
double sigmaX: 高斯核函数在X方向上的标准偏差 
double sigmaY: 高斯核函数在Y方向上的标准偏差，如果sigmaY是0，则函数会自动将sigmaY的值设置为与sigmaX相同的值，如果sigmaX和sigmaY都是0，这两个值将由ksize.width和ksize.height计算而来。
"""
img = np.array(img)
img = cv2.GaussianBlur(img, (5, 5), 0)
"""
cv2.Canny()
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是阈值1；
第三个参数是阈值2。
其中较大的阈值2用于检测图像中明显的边缘，但一般情况下检测的效果不会那么完美，边缘检测出来是断断续续的。所以这时候用较小的第一个阈值用于将这些间断的边缘连接起来。
可选参数中apertureSize就是Sobel算子的大小。apertureSize默认为3
而L2gradient参数是一个布尔值，如果为真，则使用更精确的L2范数进行计算（即两个方向的倒数的平方和再开方），否则使用L1范数（直接将两个方向导数的绝对值相加）。
"""
edges = cv2.Canny(img, 50, 150, apertureSize = 3)
cv2.imshow('img', img)
cv2.imshow('edegs', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#修复水印
img = cv2.imread(path + 'frame0.jpg')
hight, width, depth = img.shape[0:3]

#图片二值化处理，把[240, 240, 240]~[255, 255, 255]以外的颜色变成0
"""
cv2.inRange()
第一个参数：原图
第二个参数：图像中低于这个lower_red的值，图像值变为0
第三个参数：图像中高于这个upper_red的值，图像值变为0
而在lower_red～upper_red之间的值变成255
"""
thresh = cv2.inRange(img, np.array([240, 240, 240]), np.array([255, 255, 255]))

#创建形状和尺寸的结构元素
kernel = np.ones((3, 3), np.uint8)

#扩张待修复区域
"""
dilate()函数可以对输入图像用特定结构元素进行膨胀操作，该结构元素确定膨胀操作过程中的邻域的形状，各点像素值将被替换为对应邻域上的最大值
"""
hi_mask = cv2.dilate(thresh, kernel, iterations=1)
"""
cv2.inpaint()
第一个参数：需要输入的图像，要求为8位单通道或者三通道图像
第二个参数：修复掩膜， 八位单通道，非零像素表示要修补的区域。
第三个参数：需要修补的每个点的圆形邻域，为修复算法的参考半径
flags:基于快速行进算法 cv2.INPAINT_TELEA
"""
specular = cv2.inpaint(img, hi_mask, 5, flags=cv2.INPAINT_TELEA)
cv2.namedWindow("Image", 0)
cv2.resizeWindow("Image", int(width / 2), int(hight / 2))
cv2.imshow("Image", img)

cv2.namedWindow("newImage", 0)
cv2.resizeWindow("newImage", int(width / 2), int(hight / 2))
cv2.imshow("newImage", specular)
cv2.waitKey(0)
cv2.destroyAllWindows()

#固定阈值二值化
#ret, dst = cv2.threshold(src, thresh, maxval, type)
#src： 输入图，只能输入单通道图像，通常来说为灰度图
#dst： 输出图
#thresh： 阈值
#maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
#type：二值化操作的类型，包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV
path = './image/'
img = cv2.imread(path + 'frame604.jpg', 0)

image = np.array(image_contrasted)
image = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
ret, binary = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY)
cv2.imshow('image', img)
cv2.imshow('binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.axis('off')
plt.imshow(binary)

cv2.imwrite('binary2.jpg', binary)

#自适应阈值二值化
#当同一幅图像上的不同部分的具有不同亮度时，我们需要采用自适应阈值。此时的阈值是根据图像上的每一个小区域计算与其对应的阈值。因此在同一幅图像上的不同区域采用的是不同的阈值，从而使我们能在亮度不同的情况下得到更好的结果。
#dst = cv2.adaptiveThreshold(src, maxval, thresh_type, type, Block Size, C)
#src： 输入图，只能输入单通道图像，通常来说为灰度图
#dst： 输出图
#maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
#thresh_type： 阈值的计算方法，包含以下2种类型：
#           cv2.ADAPTIVE_THRESH_MEAN_C: 阈值取自相邻区域的平均值
#           cv2.ADAPTIVE_THRESH_GAUSSIAN_C:阈值取值相邻区域的加权和，权重为一个高斯窗口
#type：二值化操作的类型，与固定阈值函数相同，包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV.
#Block Size： 图片中分块的大小
#C ：阈值计算方法中的常数项,阈值就等于平均值或者加权平均值减去这个常数
path = './image/'
img = cv2.imread(path + 'frame604.jpg', 0)
img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 121, 2)
cv2.imwrite('binary.png', binary)
plt.axis('off')
plt.imshow(img)

#Otsu’s 二值化
#在使用全局阈值时，我们就是随便给了一个数来做阈值，那我们怎么知道我们选取的这个数的好坏呢？答案就是不停的尝试。如果是一副双峰图像（简单来说双峰图像是指图像直方图中存在两个峰）呢？我们岂不是应该在两个峰之间的峰谷选一个值作为阈值？这就是 Otsu 二值化要做的。
#简单来说就是对一副双峰图像自动根据其直方图计算出一个阈值。（对于非双峰图像，这种方法得到的结果可能会不理想）。
#这里用到到的函数还是 cv2.threshold()，但是需要多传入一个参数（flag）：cv2.THRESH_OTSU。这时要把阈值设为 0。然后算法会找到最优阈值，这个最优阈值就是返回值 retVal。
path = './image/'
img = cv2.imread(path + 'frame604.jpg', 0)
img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
ret3, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('binary1.png', th3)


#直方图均衡化：使图像的亮度分布接近于正态分布
path = './train_image_cnngru/'
img = cv2.imread(path + 'frame50.jpg', 0)
equ = cv2.equalizeHist(img) #使图片的像素趋近于高斯分布 
cv2.imwrite('res.png',equ)

image = cv2.fastNlMeansDenoising(equ, None, 10, 7, 21)
ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imwrite('binary3.png', resize(binary, preserve_range = True, output_shape = (224, 224, 1)))


path = './train_image_cnngru/'
for i in range(100, 1000, 100):
    img = cv2.imread(path + ('frame'+ str(i) +'.jpg'), 0)
    equ = cv2.equalizeHist(img) #使图片的像素趋近于高斯分布 
    
    #image = cv2.fastNlMeansDenoising(equ, None, 10, 7, 21)
    image = cv2.GaussianBlur(equ, (5, 5), 0)
    ret1, binary1 = cv2.threshold(image, 143, 255, cv2.THRESH_BINARY)
    #ret2, binary2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite('binary_'+str(i)+'.png', resize(binary1, preserve_range = True, output_shape = (224, 224, 1)))
    cv2.imwrite('img_'+str(i)+'.png', resize(img, preserve_range = True, output_shape = (224, 224, 1)))
    #cv2.imwrite('binary-'+str(i)+'.png', binary2)

#path = './train_image_cnngru/'
for i in range(100, 101, 100):
    img = cv2.imread('equ-'+str(i)+'.jpg', 0)
    equ = cv2.equalizeHist(img) #使图片的像素趋近于高斯分布 
    
    #image = cv2.fastNlMeansDenoising(equ, None, 10, 7, 21)
    image = cv2.GaussianBlur(equ, (5, 5), 0)
    #ret1, binary1 = cv2.threshold(image, 143, 255, cv2.THRESH_BINARY)
    #ret2, binary2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imwrite('binary_'+str(i)+'.png', resize(binary1, preserve_range = True, output_shape = (224, 224, 1)))
    #cv2.imwrite('img_'+str(i)+'.png', resize(img, preserve_range = True, output_shape = (224, 224, 1)))
    cv2.imwrite('equ1-'+str(i)+'.jpg', image)
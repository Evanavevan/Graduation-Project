# coding: utf-8

import numpy as np
import cv2
from skimage.transform import resize
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

#可视化展示    
def plot_images_labels_prediction(image, label, prediction, idx, label_dict, count, num=10):
    fig = plt.gcf()
    fig.set_size_inches(16, 5)
    index = (count-1)*10
    if num > 10:
        num = 10
    for i in range(num):
        ax = plt.subplot(2, 5, 1+i)
        ax.imshow(image[index])
        
        title = 'true:' + label_dict[label[idx]]
        if len(prediction) > 0:
            title += '\n'+'pred:' + label_dict[prediction[idx]]
            
        ax.set_title(title, fontsize = 10)
        idx += 1
        index += 1
    plt.savefig('cnngrushow%d' %count + '.png')
    plt.show()

#载入测试数据
path = './test_image_cnngru/'
test = pd.read_csv("test_cnngru.csv")
#保存预测结果
image = []
b = []
t = []
count = 1

#预测测试集
for image_name in test[test.Image_name1.notnull()].Image_name1:
    #导入数据
    #img = Image.open(path + 'frame%d.jpg' % i).convert('L')
    img = cv2.imread(path + image_name, 0)             
    equ = cv2.equalizeHist(img) #使图片的像素趋近于高斯分布 
    img = cv2.GaussianBlur(equ, (5, 5), 0)
    #ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    a = resize(img, preserve_range = True, output_shape = (224, 224, 1)).astype(int)
    img = a / 255 #数据归一化
    b.append(img)
    if count% 5 == 0:
        #增加一个维度
        b = np.expand_dims(b, 0)
        t.append(b)
        b = []
    count += 1

#载入模型
model = load_model('model_cnn_gru.h5')
#测试模型
result = []
for i in t:
    pred = model.predict(i)
    for j in pred[0]:
        result.append(np.argmax(j))
        
Image_name = list(test.Image_name[30:40])+list(test.Image_name[70:80])+list(test.Image_name[750:760])+\
list(test.Image_name[1360:1370])+list(test.Image_name[1590:1600])+list(test.Image_name[1680:1690])
for image_name in Image_name:
    image.append(mpimg.imread(path + image_name))
#可视化预测结果
#定义标签字典，每一个数字所代表的的图像类别名称
label_dict = {0:'safe_driving', 1:'hands_off_the_wheel', 2:'playing_cellphone', 3:'harassing_others' }
start = 1
plot_images_labels_prediction(image, test.Label, result, 30, label_dict, start)
start = 2
plot_images_labels_prediction(image, test.Label, result, 70, label_dict, start)
start = 3
plot_images_labels_prediction(image, test.Label, result, 750, label_dict, start)
start = 4
plot_images_labels_prediction(image, test.Label, result, 1360, label_dict, start)
start = 5
plot_images_labels_prediction(image, test.Label, result, 1590, label_dict, start)
start = 6
plot_images_labels_prediction(image, test.Label, result, 1680, label_dict, start)

#输出测试集的准确率    
accuracy = accuracy_score(np.array(test[test.Label1.notnull()].Label1), np.array(result))
print("The accuracy of the model in test is:", accuracy)

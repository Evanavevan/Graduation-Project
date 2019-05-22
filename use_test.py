# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 22:07:52 2019

@author: wen
"""

import cv2
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(1)
i = 0
while cam.isOpened():
    success, img = cam.read()
    if success == False:
        break
#    cv2.imwrite("capture%d" %i + ".jpg", img)
#    i += 1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img)
    cv2.imshow('img', img)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

img = cv2.imread('capture26.jpg', 0)
equ = cv2.equalizeHist(img)
cv2.imshow('img', equ)

img = plt.imread('capture26.jpg')
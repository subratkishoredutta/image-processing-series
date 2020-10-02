# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 23:36:47 2020

@author: Asus
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img =cv2.imread('noise.jpg')

cv2.imshow("noisy image",img)

blurimage = cv2.blur(img,(3,3))
cv2.imshow("blur",blurimage)
cv2.waitKey(0)
cv2.destroyAllWindows()

gauss1 = cv2.GaussianBlur(img,(3,3),1)
gauss2 = cv2.GaussianBlur(img,(3,3),0)
cv2.imshow("gaublur",gauss1)
cv2.waitKey(0)
cv2.destroyAllWindows()

DOG=gauss1-gauss2
cv2.imshow("DOG",DOG)
cv2.waitKey(0)
cv2.destroyAllWindows()

median = cv2.medianBlur(img,7)
cv2.imshow("median",median)
cv2.waitKey(0)
cv2.destroyAllWindows()


##edge detection

img = cv2.imread("neuron.jpg")
canny = cv2.Canny(img,100,200)

cv2.imshow("original",img)
cv2.imwrite("canny.jpg",canny)
cv2.imshow("canny",canny)
cv2.waitKey(0)
cv2.destroyAllWindows()



















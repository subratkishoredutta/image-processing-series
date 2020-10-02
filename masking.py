# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:28:34 2020

@author: Asus
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

cv2.namedWindow("Trackbars")

def nothing(x):
    pass


cv2.createTrackbar("L-H","Trackbars",0,179,nothing)
cv2.createTrackbar("L-S","Trackbars",0,255,nothing)
cv2.createTrackbar("L-V","Trackbars",0,255,nothing)
cv2.createTrackbar("U-H","Trackbars",179,179,nothing)
cv2.createTrackbar("U-S","Trackbars",255,255,nothing)
cv2.createTrackbar("U-V","Trackbars",255,255,nothing)
img = cv2.imread("path")##give the path to the image

while True:
     
     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
     
     l_h = cv2.getTrackbarPos("L-H","Trackbars")
     l_s = cv2.getTrackbarPos("L-S","Trackbars")
     l_v = cv2.getTrackbarPos("L-V","Trackbars")
     u_h = cv2.getTrackbarPos("U-H","Trackbars")
     u_s = cv2.getTrackbarPos("U-S","Trackbars")
     u_v = cv2.getTrackbarPos("U-V","Trackbars")
     
     lower_blue = np.array([l_h,l_s,l_v])
     upper_blue = np.array([u_h,u_s,u_v])
     
     mask = cv2.inRange(hsv, lower_blue, upper_blue)
     
     cv2.imshow('img',img)
     cv2.imshow('mask',mask)
     #cv2.imwrite('mask.jpg',mask)
     if cv2.waitKey(10) & 0xFF==ord('q'):
         break


cv2.destroyAllWindows()


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask =cv2.inRange(hsv,np.array([28,111,190]),np.array([179,255,255]))
cv2.imshow("mask",mask)
cv2.imwrite("mask.jpg",mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


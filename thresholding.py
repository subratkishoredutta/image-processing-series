# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:36:37 2020
q
@author: Asus
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('data.tiff')



bnw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('image',img)
cv2.imshow('bnw',bnw)
cv2.waitKey(0)
cv2.destroyAllWindows()

threshold=127

_,fig1= cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)

_,fig2= cv2.threshold(img,threshold,255,cv2.THRESH_BINARY_INV)

_,fig3= cv2.threshold(img,threshold,255,cv2.THRESH_TOZERO)

_,fig4= cv2.threshold(img,threshold,255,cv2.THRESH_TRUNC)

cv2.imshow('image1',fig1)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image2',fig2)

cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow('image3',fig3)

cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow('image4',fig4)

cv2.waitKey(0)
cv2.destroyAllWindows()





img2 = cv2.imread('otsu.tiff',0)

plt.imshow(img2,cmap="gray")

_,OTSU1 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_,OTSU2= cv2.threshold(img2,0,255,cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
_,OTSU3= cv2.threshold(img2,0,255,cv2.THRESH_TOZERO+ cv2.THRESH_OTSU)
_,OTSU4= cv2.threshold(img2,0,255,cv2.THRESH_TRUNC+ cv2.THRESH_OTSU)




cv2.imshow('binaryOTSU1',OTSU1)
cv2.imshow('biinvOTSU2',OTSU2)
cv2.imshow('tozeroOTSU3',OTSU3)
cv2.imshow('truncOTSU4',OTSU4)


cv2.waitKey(0)


cv2.destroyAllWindows()
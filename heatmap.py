# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:45:00 2021

@author: subrat
"""
#movement localisation using heatmaps
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
from builtins import range,input
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import scipy as sp
from PIL import Image
import os
import cmapy

resnet = ResNet50(input_shape = (224,224,3), weights='imagenet', include_top=True)
resnet.summary()
activation_layer = resnet.get_layer('activation_49')
model = Model(inputs=resnet.input, output=activation_layer.output)
final_dense =  resnet.get_layer('fc1000')
W = final_dense.get_weights()[0]

camera=cv2.VideoCapture(0)

while True:
        _,frame1=camera.read()
        frame1=cv2.resize(frame1,(224,224))
        _,frame2=camera.read()
        frame2=cv2.resize(frame2,(224,224))
        
        diff=cv2.absdiff(frame1,frame2)
        diff=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(diff, (5, 5), 0)##negates the noise
        ret,diff = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
        diff=np.reshape(diff,(224,224,1))
        z=np.zeros((224,224,3))
        z=z+diff
        diff=np.array(z,dtype=np.uint8)
        x = preprocess_input(np.expand_dims(diff, 0))
        fmaps = model.predict(x)[0]
        probs = resnet.predict(x)
        pred = np.argmax(probs[0])
        w = W[:, pred]
        cam = fmaps.dot(w)
        cam = sp.ndimage.zoom(cam, (32,32), order=1)
        cam=np.array(cam,dtype=np.uint8)*3
        cam=np.reshape(cam,(224,224,1))
        z=np.zeros((224,224,3))
        z=z+cam
        X=np.array(z,dtype=np.uint8)
        X = cv2.applyColorMap(X, cmapy.cmap('inferno'))
        
        img=cv2.resize(frame2,(700,500))
        X=cv2.resize(X,(700,500))
        fin=cv2.addWeighted(img,0.4,X,0.6,0)
        cv2.imshow('original',img)
        cv2.imshow('activationMap',X)
        cv2.imshow('frame',fin)
        
        key=cv2.waitKey(1)
        if key==ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
        
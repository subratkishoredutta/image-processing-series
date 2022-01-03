# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 10:57:18 2022

@author: Asus
"""

import cv2
import mediapipe as mp

cap=cv2.VideoCapture(0)

mp_drawing_styles = mp.solutions.drawing_styles
draw=mp.solutions.drawing_utils
pose=mp.solutions.pose
hands = mp.solutions.hands
face_mesh = mp.solutions.face_mesh
fm=face_mesh.FaceMesh()
ha=hands.Hands()
p=pose.Pose()

x,y,z=1,0,0

while(True):
    _,img=cap.read()
    img=cv2.resize(img, (800, 800))
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    han=ha.process(rgb)
    facm=fm.process(rgb)
    res=p.process(rgb)
    mask=img*0
    if x==1:
        if res.pose_landmarks:
                draw.draw_landmarks(mask,res.pose_landmarks,pose.POSE_CONNECTIONS)
    if y==1:
                if han.multi_hand_landmarks:
                    for hand_landmarks in han.multi_hand_landmarks:
                        draw.draw_landmarks(mask,hand_landmarks,hands.HAND_CONNECTIONS)
    if z==1:
                if facm.multi_face_landmarks:
                     for face_landmarks in facm.multi_face_landmarks:
                        draw.draw_landmarks(
                            mask,
                            landmark_list=face_landmarks,
                            connections=face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        
                        draw.draw_landmarks(mask,landmark_list=face_landmarks,connections=face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        
                        #draw.draw_landmarks(mask,landmark_list=face_landmarks,connections=face_mesh.FACEMESH_IRISES,landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

    cv2.imshow('motion',mask)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
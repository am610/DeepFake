# @Ayan Mitra, 2020
# ayanmitra375@gmail.com
#********************************************************** 
#********************************************************** 
from tensorflow.keras.models import load_model
import imageio
from IPython.display import HTML
import dlib
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib.animation as animation
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import warnings
warnings.filterwarnings("ignore")
#********************************************************** 
#**********************************************************


#********************************************************** 
# !!!!!!!
# Download the Model file from :
# #https://drive.google.com/file/d/1T8hj8ww2xmW1rzDt_rJfo1l069md3GiY/view?usp=sharing
# !!!!!!!
#********************************************************** 


#********************************************************** 
#********************************************************** 
# Based on the detection of individual frames per video. If they are flagged as real or fake. 
def result(a,b):
  r = a/b
  if r<0.7:
    print('Fake Video')
  else:
    print('Real Video')  
#********************************************************** 

def video_test(path):
    warnings.filterwarnings("ignore")
    pr_data = []
    hypo=[]
    count=0;total=0;
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(path)
    frameRate = cap.get(5)
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if ret != True:
            break
        if frameId % ((int(frameRate)+1)*1) == 0:
            face_rects, scores, idx = detector.run(frame, 0)
            for i, pos in enumerate(face_rects):
                left = pos.left()
                top = pos.top()
                right = pos.right()
                bottom = pos.bottom()
                total+=1
                crop_img = frame[top:bottom, left:right]
                if (np.shape(crop_img)[0]==0):
                  continue
                elif (np.shape(crop_img)[1]==0): 
                  continue
                elif (np.shape(crop_img)[2]==0): 
                  continue   
                data = img_to_array(cv2.resize(crop_img, (128, 128))).flatten() / 255.0
                #print(np.shape(data))
                data = data.reshape(-1, 128, 128, 3)
                #print(model2.predict_classes(data))
                hypo.append(model2.predict_classes(data)[0])
                if(model2.predict_classes(data)==1):
                  count+=1

    result(count,total)
#********************************************************** 
#********************************************************** 

# LOAD THE MODEL :
#!!!!! SPECIFY THE LOCAL PATH OF THE MODEL CORRECTLY AFTER 
# DOWNLOADING IT FROM THE GOOGLE DRIVE MENTIONED ABOVE

model2 = load_model(path_to_model+'model-2.h5')
#********************************************************** 

# INPUT THE VIDEO PATH HERE

user = input("Input Video Path :") #                                                                         
path = str(user)

video_test(path)

#********************************************************** 

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from scipy.interpolate import splprep, splev

# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
# from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
# from utils import INPUT_SHAPE, batch_generator
# import argparse
# import os

def crop_image(x):
    crop=x[53:160,0:320]
    return crop

def flip_image(img,angle):
    angle=-angle
    return cv2.flip(img,1),angle

def translate_image(img,x,y):
    height, width = img.shape[:2]
    T = np.float32([[1, 0, x], [0, 1, y]])
    img_translation = cv2.warpAffine(img, T, (width, height))
    return img_translation

def brightness(img,factor):
    change=0.5+factor*1
    img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img[:,:,2]=img[:,:,2]*change
    return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

def rand_blur(img):
    size=1+int(random.random()*10)
    if(size%2==0):
        size+=1
    img=cv2.GaussianBlur(img,(size,size),0)
    return img

def add_random_shadow(img, w_low=0.6, w_high=0.85):
    
    cols, rows = (img.shape[0], img.shape[1])
    
    top_y = np.random.random_sample() * rows
    bottom_y = np.random.random_sample() * rows
    bottom_y_right = bottom_y + np.random.random_sample() * (rows - bottom_y)
    top_y_right = top_y + np.random.random_sample() * (rows - top_y)
    if np.random.random_sample() <= 0.5:
        bottom_y_right = bottom_y - np.random.random_sample() * (bottom_y)
        top_y_right = top_y - np.random.random_sample() * (top_y)    
    poly = np.asarray([[ [top_y,0], [bottom_y, cols], [bottom_y_right, cols], [top_y_right,0]]], dtype=np.int32)
        
    mask_weight = np.random.uniform(w_low, w_high)
    origin_weight = 1 - mask_weight
    
    mask = np.copy(img).astype(np.int32)
    cv2.fillPoly(mask, poly, (0, 0, 0))
    
    return cv2.addWeighted(img.astype(np.int32), origin_weight, mask, mask_weight, 0).astype(np.uint8)

def readcsv():
    drive_log=pd.read_csv("train2/drive.csv")
    img=cv2.imread(drive_log.iloc[0]['loc'])
    crp=add_random_shadow(img)
    cv2.imshow("img",crp)
    cv2.waitKey(0)
    inputs=[]
    outputs=[]

readcsv()

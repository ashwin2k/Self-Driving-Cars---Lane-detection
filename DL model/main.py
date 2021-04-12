import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten,Input
from keras.applications import ResNet50

import argparse
import os
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

def roi_mask():
    stencil = np.zeros((160,320), dtype='uint8')
    #DEFINE the ROI boundary
    polygon = np.array([[56,50], [266,50], [318,125], [0,125]])
    cv2.fillConvexPoly(stencil, polygon, 1)
    return stencil
def get_roi(img):
    stencil=roi_mask()
    for x in range(0,3):
        img[:, :, x] = cv2.bitwise_and(img[:, :, x] , img[:, :, x], mask=stencil)
    return img
def getBirdEye(img):
    road_coordinates=np.float32([[56,50], [266,50], [318,125], [0,125]])
    dest_coordinates=np.float32([[0,0], [160,0], [160,320], [0,320]])
    dest_matrix=cv2.getPerspectiveTransform(road_coordinates,dest_coordinates)
    result = cv2.warpPerspective(img, dest_matrix, (160,320))
    return result

def smoothenContour(contours):
    smoothened = []
    for contour in contours:
        x,y = contour.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        tck, u = splprep([x,y], u=None, s=1.0, per=1)
        u_new = np.linspace(u.min(), u.max(), 50)
        x_new, y_new = splev(u_new, tck, der=0)
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
        smoothened.append(np.asarray(res_array, dtype=np.int32))
    return smoothened
    
    
def morph_close(img):
    kernal=np.ones((3,3),np.uint8)
    closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernal)
    return closing
def read_image(img_path,rand_shadow=False):
    img = cv2.imread(img_path)
    print("SHAPE")
    print(img.shape)
    if(rand_shadow):
        img=add_random_shadow(img)

    # img1=get_roi(img)
    # blur = cv2.blur(img,(5,5))
    # blur0=cv2.medianBlur(blur,5)
    # blur1= cv2.GaussianBlur(blur0,(5,5),0)
    # blur2= cv2.bilateralFilter(blur1,9,75,75)
    # hsv = cv2.cvtColor(blur2,cv2.COLOR_BGR2HSV)
    # lower_gray = np.array([0, 5, 50], np.uint8)
    # upper_gray = np.array([179, 50, 255], np.uint8)
    # mask = cv2.inRange(hsv, lower_gray, upper_gray)
    # res = cv2.bitwise_and(img,img, mask= mask)
    # res=morph_close(res)
    # # #Get contours
    # res=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY);
    # res= cv2.blur(res,(3,3))
    # birdeye=getBirdEye(res)
    # dim,contours, hierarchy = cv2.findContours(birdeye, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours=smoothenContour(contours)
    # areas = [cv2.contourArea(c) for c in contours]
    # max_index = np.argmax(areas)
    # mask_image=np.zeros_like(birdeye)
    # cv2.drawContours(mask_image, contours, max_index, (255,255,255), -1)
    return img

def flip_image(img,angle):
    angle=-angle
    return cv2.flip(img,1),angle

def generate_image(raw_image_path,steer,ind):
    print("GENERATING FOR"+str(ind))
    m_image=read_image(raw_image_path)
    # shadow_image=read_image(raw_image_path,True)
    flipped_image,flip_steer=flip_image(m_image,steer)
    # flipped_image_shadow,flip_steer_shadow=flip_image(shadow_image,steer)
    
    # m_image= np.expand_dims(m_image, axis=2)
    # flipped_image= np.expand_dims(flipped_image, axis=2)

    images=np.array([m_image,flipped_image])
    angles=np.array([steer,flip_steer])
    return images,angles

def get_dataset():
    drive_log=pd.read_csv("train2/drive.csv")
    # drive_log=drive_log[:5]
    print(drive_log)
    arr_x=np.array([])
    arr_y=np.array([])
    for index,row in drive_log.iterrows():
        inp,oup=generate_image(row['loc'],float(row['angle']),index)
        
        if(index==0):
            arr_x=inp
            arr_y=oup
        arr_x=np.append(arr_x,inp,axis=0)
        arr_y=np.append(arr_y,oup,axis=0)
    print("FINAL SIZE:")
    print(arr_x.shape)
    return arr_x,arr_y

def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160,320,3)))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

def train_data():
    X,Y=get_dataset()
    model=nvidia_model()
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))
    print(model.summary())
    print(X.shape)
    history = model.fit(X, Y, nb_epoch=25, validation_split=0.15, batch_size=24, shuffle=1)
    # model.save('./saved_model/my_model_2')
    model.save('./saved_model/model6.h5')

train_data()


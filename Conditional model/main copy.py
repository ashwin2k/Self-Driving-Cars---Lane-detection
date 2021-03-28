import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
def roi_mask():
    stencil = np.zeros((720,1280), dtype='uint8')
    #DEFINE the ROI boundary
    polygon = np.array([[0,687], [450,166], [707,168], [1280,687]])
    cv2.fillConvexPoly(stencil, polygon, 1)
    return stencil
def get_roi(img):
    stencil=roi_mask()
    for x in range(0,3):
        img[:, :, x] = cv2.bitwise_and(img[:, :, x] , img[:, :, x], mask=stencil)
    return img
def getBirdEye(img):
    road_coordinates=np.float32([[230,380],[980,380],[1280,687],[0,687]])
    dest_coordinates=np.float32([[0,0], [1280,0], [1280,720], [0,720]])
    dest_matrix=cv2.getPerspectiveTransform(road_coordinates,dest_coordinates)
    result = cv2.warpPerspective(img, dest_matrix, (1280,720))
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
    kernal=np.ones((12,12),np.uint8)
    closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernal)
    return closing
def read_image():
    img = cv2.imread('./images/6.jpeg')
    # img=cv2.resize(img,(1280,720))
    # img1=get_roi(img)
    # blur = cv2.blur(img,(5,5))
    # blur0=cv2.medianBlur(blur,5)
    # blur1= cv2.GaussianBlur(blur0,(5,5),0)
    # blur2= cv2.bilateralFilter(blur1,9,75,75)
    # hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
    # lower_gray = np.array([0, 0, 50], np.uint8)
    # upper_gray = np.array([179, 50, 255], np.uint8)
    # mask = cv2.inRange(hsv, lower_gray, upper_gray)
    # res = cv2.bitwise_and(img,img, mask= mask)
    # res=morph_close(res)
    # #Get contours
    # res=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY);
    # res= cv2.blur(res,(5,5))
    # birdeye=getBirdEye(res)

    # contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours=smoothenContour(contours)
    # areas = [cv2.contourArea(c) for c in contours]
    # max_index = np.argmax(areas)
    # cv2.drawContours(res, contours, max_index, (255,255,102), 3)
    cv2.imshow("s",img)
    cv2.waitKey(0)

read_image()
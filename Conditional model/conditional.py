import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.interpolate import splprep, splev

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def sliding_window(img, nwindows=15, margin=150, minpix = 1, draw_windows=True):
    global left_a, left_b, left_c,right_a, right_b, right_c 
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 30.5/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/720 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    car_pos = img.shape[1]/2
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center = (car_pos - lane_center_position) * xm_per_pix / 10

    left_fitx = leftx[0] * ploty ** 2 + leftx[1] * ploty + leftx[2]
    right_fitx = rightx[0] * ploty ** 2 + rightx[1] * ploty + rightx[2]

    radius1 = round((float(left_curverad) + float(right_curverad))/2.,2)
    print(l_fit_x_int )

    if right_fitx[0] - right_fitx[-1] > 60:
        curve_direction = 'Left Curve'
        radius=-5729.57795/radius1
    elif right_fitx[-1] - right_fitx[0] > 60:
        curve_direction = 'Right Curve'
        radius=5729.57795/radius1
    else:
        curve_direction = 'Straight'
        radius=5729.57795/radius1
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad, left_fitx[0] - left_fitx[-1],curve_direction,radius)

def roi_mask():
    stencil = np.zeros((720,1280), dtype='uint8')
    #DEFINE the ROI boundary
    #ORDER: BOTTOM LEFT, TOP LEFT, TOP RIGHT, BOTTOM RIGHT
    polygon = np.array([[89,411], [459,195], [810,195], [1280,411]]) 
    cv2.fillConvexPoly(stencil, polygon, 1)
    return stencil
def get_roi(img):
    stencil=roi_mask()
    for x in range(0,3):
        img[:, :, x] = cv2.bitwise_and(img[:, :, x] , img[:, :, x], mask=stencil)
    # img=img[225:375,185:702]
    return img
def getBirdEye(img):

    #ORDER: TOP LEFT, TOP RIGHT, BOTTOM RIGHT, BOTTOM LEFT
    road_coordinates=np.float32([[459,195], [810,195], [1280,411],[89,411]])
    dest_coordinates=np.float32([[0,0], [720,0], [720,1280], [0,1280]])
    dest_matrix=cv2.getPerspectiveTransform(road_coordinates,dest_coordinates)
    result = cv2.warpPerspective(img, dest_matrix, (720,1280))
    return result
def get_normal_view(img):
    road_coordinates=np.float32([[459,195], [810,195], [1280,411],[89,411]])
    dest_coordinates=np.float32([[0,0], [720,0], [720,1280], [0,1280]])
    dest_matrix=cv2.getPerspectiveTransform(dest_coordinates,road_coordinates)
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
    
def draw_line(img):
    n_windowsize=5
    points=[]
    n_row_start=0
    while(n_row_start+5 != 1280):
        pts=[0,0]
        for i in range(0,5):
            row=img[n_row_start+i,:]
            ls = [i for i, e in enumerate(row) if e != 0]
            midpt_X=(ls[0]+ls[-1])/2
            # first=(midpt,n_row_start+i)
            pts=[pts[0]+midpt_X,pts[1]+n_row_start+i]
        pts=[int(pts[0]/5),int(pts[1]/5)]
        points.append(pts)
        n_row_start+=5
    
    points = np.array(points).reshape((-1,1,2))
    return points
def morph_close(img):
    kernal=np.ones((12,12),np.uint8)
    closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernal)
    return closing
def read_image():
    img = cv2.imread('/home/ashwin/edison-mk1/Conditional model/images/3.jpg')
    img=cv2.resize(img,(1280,720))
    img_org=img.copy()
    img1=get_roi(img)
    blur = cv2.blur(img1,(5,5))
    blur0=cv2.medianBlur(blur,5)
    blur1= cv2.GaussianBlur(blur0,(5,5),0)
    blur2= cv2.bilateralFilter(blur1,9,75,75)
    hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 5, 50], np.uint8)
    upper_gray = np.array([179, 50, 255], np.uint8)
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    res = cv2.bitwise_and(img1,img1, mask= mask)    
    res=morph_close(res)
    # # # #Get contours
    res=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY);
    res= cv2.blur(res,(5,5))

    birdeye=getBirdEye(res)  

    contours, hierarchy = cv2.findContours(birdeye, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours=smoothenContour(contours)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    empty_mat=np.zeros_like(birdeye)
    empty_mat=cv2.cvtColor(empty_mat,cv2.COLOR_GRAY2BGR)
    max_contour=contours[max_index]
    max_contour = cv2.approxPolyDP(max_contour, 0.009 * cv2.arcLength(max_contour, True), True)
    cv2.drawContours(empty_mat, [max_contour], -1, (0,255,0), -1)
    # cv2.imshow("ss",img)
    # out_img, curves, lanes, ploty= sliding_window(empty_mat) 
    # curverad=get_curve(out_img, curves[0],curves[1])
    empty_mat_gray=empty_mat[:,:,1]
    center_pts=draw_line(empty_mat_gray)
    print(center_pts)
    polyline=cv2.polylines(empty_mat,[center_pts],False,(255,0,0),3)
    normal_perspective=get_normal_view(empty_mat)
    final=cv2.addWeighted(img_org,1,normal_perspective,0.5,0)
    cv2.imwrite("/home/ashwin/edison-mk1/Conditional model/results/result.jpg",final)
    
    cv2.imshow("Final2",final)
    cv2.waitKey(5000)

read_image()

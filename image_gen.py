import numpy as np
import cv2 as cv
import pickle
import glob
from tracker import tracker

dist_pickle =  pickle.load(open('camera_cal/cal_pickle.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']
# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Define a function that thresholds the S-channel of HLS and the V channel of HSV 
def color_thresh(img, sthresh=(0, 255), vthresh=(0,255)):
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1

    return output

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), max(0,int(center-width)):min(int(center+width), img_ref.shape[1])] = 1
    return output

images = glob.glob('test_images/test*.jpg')


for idx, fname in enumerate(images):
    img = cv.imread(fname)
    img = cv.undistort(img,mtx,dist,None,mtx)

    #create 
    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=12, thresh_max=255)
    grady = abs_sobel_thresh(img, orient='y', thresh_min=25, thresh_max=255)
    c_binary = color_thresh(img, sthresh=(100,255), vthresh=(50,255))
    preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255
    #result = preprocessImage
    

    #defining perspective, transformation area
    img_size = (img.shape[1], img.shape[0])
    #define shape of trapezoid
    bot_width = .76 
    mid_width = .08 
    height_pct = .62 
    bottom_trim= .935 #trim car hood
    src = np.float32([[img.shape[1]*(.5-mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5+mid_width/2), img.shape[0]*height_pct], [img.shape[1] *(.5+bot_width/2), img.shape[0]*bottom_trim], [img.shape[1]*(.5-bot_width/2), img.shape[0]*bottom_trim] ])
    offset = img_size[0]*.25
    dst = np.float32([[offset,0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

    #source and destination of perspective transform
    M = cv.getPerspectiveTransform(src,dst)
    #inverse to put it back 
    Minv = cv.getPerspectiveTransform(dst,src)
    warped = cv.warpPerspective(preprocessImage, M, img_size, flags=cv.INTER_LINEAR)
    #result = warped

    window_width = 25
    window_height = 80

    curve_centers = tracker(Mywindow_width = window_width, Mywindow_height=window_height, Mymargin = 25, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor = 15)

    window_centroids = curve_centers.find_window_centroids(warped)

    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    l_x = []
    r_x = []
    # loop through 
    for level in range(0, len(window_centroids)):
        #draws window area
        l_x.append(window_centroids[level][0])
        r_x.append(window_centroids[level][1])
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0],level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1],level)

        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    #draw resulting lines    
    template = np.array(r_points+l_points, np.uint8) #add L and R  pixels together 
    zero_channel = np.zeros_like(template) #to pass through 0 in the R and B color channels below
    template = np.array(cv.merge((zero_channel, template, zero_channel)), np.uint8) #green template
    warpage = np.array(cv.merge((warped,warped,warped)), np.uint8) 
    #result = cv.addWeighted(warpage, 1, template, 0.5,0.0)

    #draw lane lines 
    yvals = range(0,warped.shape[0])

    res_yvals = np.arange(warped.shape[0] - (window_height/2),0,-window_height)

    l_fit = np.polyfit(res_yvals, l_x, 2)
    l_fit_x = l_fit[0]*yvals*yvals + l_fit[1]*yvals + l_fit[2]
    l_fit_x = np.array(l_fit_x, np.int32)

    r_fit = np.polyfit(res_yvals, r_x, 2)
    r_fit_x = r_fit[0]*yvals*yvals + r_fit[1]*yvals + r_fit[2]
    r_fit_x = np.array(r_fit_x, np.int32)

    l_lane = np.array(list(zip(np.concatenate((l_fit_x-window_width/2,l_fit_x[::-1]+window_width/2), axis=0), np.concatenate((yvals,yvals[::-1]), axis=0))), np.int32)
    r_lane = np.array(list(zip(np.concatenate((r_fit_x-window_width/2,r_fit_x[::-1]+window_width/2), axis=0), np.concatenate((yvals,yvals[::-1]), axis=0))), np.int32) 
    middle_marker = np.array(list(zip(np.concatenate((l_fit_x+window_width/2,r_fit_x[::-1]+window_width/2), axis=0), np.concatenate((yvals,yvals[::-1]), axis=0))), np.int32)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv.fillPoly(road,[l_lane], color=[255,0,0])
    cv.fillPoly(road,[r_lane], color=[0,0,255])
    cv.fillPoly(road_bkg,[middle_marker], color=[0,255,0])
    #use this background to make lines darker
    cv.fillPoly(road_bkg, [l_lane], color=[255,255,255])
    cv.fillPoly(road_bkg, [r_lane], color=[255,255,255])

    #swap perspective back w/ inverse of above
    road_warped = cv.warpPerspective(road,Minv,img_size,flags=cv.INTER_LINEAR)
    road_warped_bkg = cv.warpPerspective(road_bkg, Minv, img_size, flags=cv.INTER_LINEAR)

    base = cv.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv.addWeighted(base, 1.0, road_warped, 1.0, 0.0)

    ym_per_pix = curve_centers.ym_per_pix
    xm_per_pix = curve_centers.xm_per_pix

    # check out curvature of left lane (l_x)
    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(l_x,np.float32)*xm_per_pix, 2)
    curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1]) **2)**1.5)/ np.absolute(2*curve_fit_cr[0])

    #calculate the offset
    camera_center = (l_fit_x[-1] + r_fit_x[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    #positive/negative tells us if it's left or right
    if center_diff <= 0:
        side_pos = 'right'

    cv.putText(result, 'Radius of curvature = '+str(round(curverad,3))+'(m)',(50,50) , cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv.putText(result, 'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center', (50,100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #result = road_warped

    write_name = './test_images/center_display_'+str(idx)+'.jpg'
    cv.imwrite(write_name,result)
'''	 camera_calibration.py

Usage:	python camera_calibration.py board_w board_h number_of_views
	
This program reads a chessboard's width and height, collects requested number of views and calibrates the camera.

This is a little modified version of the example 11-1 given in the book "Learning OpenCV: Computer Vision with the OpenCV Library". 

Converted to Python by Abid.K	--mail me at abidrahman2@gmail.com	
https://github.com/abidrahmank/OpenCV-Python/blob/master/Other_Examples/camera_calibration.py
'''

################################################################################################


import numpy as np
import cv2 as cv
import glob
import pickle

import time,sys
cheX = 9
cheY = 6
objp = np.zeros((cheY*cheX, 3), np.float32)
objp[:,:2] = np.mgrid[0:cheX,0:cheY].T.reshape(-1,2)

images = glob.glob('camera_cal/calibration*.jpg')
print(len(images))
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
	print (fname)
	img = cv.imread(fname)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the chessboard corners
	ret, corners = cv.findChessboardCorners(gray, (cheX,cheY), None)

	# If found, add object points, image points
	if ret == True:
		objpoints.append(objp)
		imgpoints.append(corners)

		# Draw and display the corners
		cv.drawChessboardCorners(img, (cheX,cheY), corners, ret)
		#write_name = 'corners_found'+str(idx)+'.jpg'
		#cv.imwrite(write_name, img)
		#cv.imshow('img', img)
		#cv.waitKey(500)

#cv2.destroyAllWindows()

# Test undistortion on an image
img = cv.imread('camera_cal/calibration3.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img_size = (img.shape[1], img.shape[0])
print (img_size)
# Do camera calibration given object points and image points
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


dst = cv.undistort(img, mtx, dist, None, mtx)
write_name = './test_images/calibrated.jpg'
cv.imwrite(write_name,dst)
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_cal/cal_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,5))
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=20)
# ax2.imshow(dst)
# ax2.set_title('Undistorted Image', fontsize=20)
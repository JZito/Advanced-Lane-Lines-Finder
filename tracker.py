import numpy as np
import cv2 as cv2
np.set_printoptions(threshold=np.nan)

class tracker():

	def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym=1, My_xm=1, Mysmooth_factor=15):

		self.recent_centers = []

		self.window_width = Mywindow_width

		self.window_height = Mywindow_height 

		self.margin = Mymargin #pixel distance in both directions to move window

		self.ym_per_pix = My_ym # m / pixel on y axis

		self.xm_per_pix = My_xm # m / pixel on x axis

		self.smooth_factor = Mysmooth_factor

	def find_window_centroids2(self,binary_warped):
	    # Assuming you have created a warped binary image called "binary_warped"
	    # Take a histogram of the bottom half of the image
	    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
	    # Create an output image to draw on and  visualize the result
	    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	    
	   
	    # Find the peak of the left and right halves of the histogram
	    # These will be the starting point for the left and right lines
	    midpoint = np.int(histogram.shape[0]/2)
	    leftx_base = np.argmax(histogram[:midpoint])
	    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	    # Choose the number of sliding windows
	    nwindows = 9
	    # Set height of windows
	    window_height = np.int(binary_warped.shape[0]/nwindows)
	    # Identify the x and y positions of all nonzero pixels in the image
	    nonzero = binary_warped.nonzero()
	    nonzeroy = np.array(nonzero[0])
	    nonzerox = np.array(nonzero[1])
	    # Current positions to be updated for each window
	    leftx_current = leftx_base
	    rightx_current = rightx_base
	    # Set the width of the windows +/- margin
	    margin = 100
	    # Set minimum number of pixels found to recenter window
	    minpix = 50
	    # Create empty lists to receive left and right lane pixel indices
	    left_lane_inds = []
	    right_lane_inds = []

	    # Step through the windows one by one
	    for window in range(nwindows):
	        # Identify window boundaries in x and y (and right and left)
	        win_y_low = binary_warped.shape[0] - (window+1)*window_height
	        win_y_high = binary_warped.shape[0] - window*window_height
	        win_xleft_low = leftx_current - margin
	        win_xleft_high = leftx_current + margin
	        win_xright_low = rightx_current - margin
	        win_xright_high = rightx_current + margin
	        # Draw the windows on the visualization image
	        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
	        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
	        # Identify the nonzero pixels in x and y within the window
	        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
	        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
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
	    if len(leftx) == 0:
	        left_fit =[]
	    else:
	        left_fit = np.polyfit(lefty, leftx, 2)
	    
	    if len(rightx) == 0:
	        right_fit =[]
	    else:
	        right_fit = np.polyfit(righty, rightx, 2)
	    

	    
	    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


	    return np.array(left_fit), np.array(right_fit),out_img

	#function for identifying lane segments
	def find_window_centroids(self, warped):

		window_width = self.window_width
		window_height = self.window_height
		margin = self.margin 

		window_centroids = [] #the left and right centroid positions for this level
		window = np.ones(window_width) #template
		#find starting pos for L and R lanes by squashing 
		l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
		l_center = np.argmax(np.convolve(window,l_sum))-window_width/2

		r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
		r_center = np.argmax(np.convolve(window, r_sum))-window_width/2+int(warped.shape[1]/2)
		#print(l_center, r_center)
		window_centroids.append((l_center, r_center))

		for level in range(1, (int)(warped.shape[0]/window_height)):
			#print(l_min_index, " , ", l_max_index, " , ", r_min_index , " , " , r_max_index)
			image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
			#use convolution to create histogram style 1d arrays, clever idea from udacity live help video
			conv_signal = np.convolve(window, image_layer)
			
			offset = window_width/2
			l_min_index = int(max(l_center+offset-margin,0))
			l_max_index = int(min(l_center+offset+margin, warped.shape[1]))
			l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
			#print(l_min_index)
			r_min_index = int(max(r_center+offset-margin,0))
			r_max_index = int(min(r_center+offset+margin, warped.shape[1]))
			r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

			window_centroids.append((l_center,r_center))

		self.recent_centers.append(window_centroids)

		return np.average(self.recent_centers[-self.smooth_factor:], axis=0)
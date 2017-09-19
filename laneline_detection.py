import glob
import numpy as np
import matplotlib.image as mpimg
import cv2
import copy

def calibrate():
    
    images = glob.glob('./camera_cal/*.jpg')

    #number of x and y corners
    nx = 9
    ny = 6

    #map the coordinates of corners in this 2D image
    imgpoints = [] #2D points in image plane

    # to the 3D coordinates of the real world, undistorted chessboard corners
    objpoints = [] #3D points in real world space

    #prepare object points, like (0, 0, 0) (2, 0, 0) ... (8, 5, 0)
    objp = np.zeros((nx * ny, 3), np.float32) 
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) # x, y coordinates

    for path in images:

        img = mpimg.imread(path)

        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        #draw the corners
        if ret == True:
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

            imgpoints.append(corners)
            objpoints.append(objp)

    # gets the first image to gets the shape
    img_sample = mpimg.imread(images[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_sample.shape[0:2], None, None)
    
    return ret, mtx, dist, rvecs, tvecs

# calibrates and exposes the variables for the other methods
ret, mtx, dist, rvecs, tvecs = calibrate()

# given two points and a new_x, we define an y equivalent
# good to change the size of line
def find_y_line(x_min, x_max, y_min, y_max, new_x):
    del_y = y_min - y_max
    del_x = x_max - x_min
    slope = del_y / del_x

    b = y_min - (x_max * slope)
    
    # y = mx + b
    y = (slope * new_x) + b
    return y

def extract_points_mask(img):
    imshape = img.shape
    
    rows = imshape[0]
    cols = imshape[1]

    x_bl = int(.15 * cols)
    x_tl = int(.469 * cols)
    x_tr = int(.532 * cols)
    x_br = int(.875 * cols)

    y_top = int(0.62 * rows)
    
    return x_bl, x_tl, x_tr, x_br, y_top

def warp(img, src, dest):
    
    img_size = (img.shape[1], img.shape[0])
    
    M = cv2.getPerspectiveTransform(src, dest)
    Minv = cv2.getPerspectiveTransform(dest, src)
    
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, Minv

def undist(img):
    return cv2.undistort(img, mtx, dist, None, mtx)



# gradients

def abs_sobel_thresh(img, orient='x', thresh=(0, 255), use_S_channel=False, sobel_kernel=3):
    
    img_src = None
    
    if use_S_channel:
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        img_src = hls[:,:,2]
    else:
        img_src = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = None
    if orient == 'x':
        sobel = cv2.Sobel(img_src, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(img_src, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        raise Exception('The param ' + orient + ' is unknow')
        
    abs_sobel = np.absolute(sobel)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1        
    return binary_output

def hls_threshold(img, thresh=(0, 255)):
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
   
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output




# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
        # polynomial coefficients of the last n fit
        self.recent_fit = []
        
        # number of rollbacks in a row
        self.n_rollbacks = 0
    
    # x values of polynomial
    def add_xfitted(self, xfitted):
        self.recent_xfitted.append(xfitted)
        # matem n ultimos elementos
        n = 5
        self.recent_xfitted = self.recent_xfitted[-n:]
        self.bestx = np.average(self.recent_xfitted, axis=0)
        
    def set_current_fit(self, current_fit):
        self.current_fit = current_fit
        
        self.recent_fit.append(current_fit)
        
        # matem n ultimos elementos
        n = 5
        self.recent_fit = self.recent_fit[-n:]
        self.best_fit = np.average(self.recent_fit, axis=0)
        
    def get_indexes_of_line(self, img, nonzero, x_base):
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        minpx = 100
        margin = 100

        lines_inds = []
        
        if self.detected == False:
            self.detected = True
            
            cur_x_base = x_base
            
            height_img = img.shape[0]
            nwindows = 10
            window_height = height_img//nwindows

            for i in range(nwindows):

                # finds the left top point in the rectangle
                y_top = height_img - ((i + 1) * window_height)
                # finds the right bottom point in the rectangle
                y_bottom = height_img - (i * window_height)


                # defines the left window
                win_x_low = cur_x_base - margin
                win_x_high = cur_x_base + margin

                # where True we gets the indices
                good_ind = ((nonzerox >= win_x_low) & (nonzerox <= win_x_high) & 
                                  (nonzeroy >= y_top) & (nonzeroy <= y_bottom)).nonzero()[0]

                # stores the indexes of the pixels that make up the line
                lines_inds.append(good_ind)

                # recenter
                if len(good_ind) > minpx:
                    cur_x_base = np.int(np.mean(nonzerox[good_ind]))

            lines_inds = np.concatenate(lines_inds)
        else:
            # onde tiver pixel (nonzeroy) definimos uma fronteira de pesquisa
            search_area_x_low = self.best_fit[0]*nonzeroy**2 + \
                                   self.best_fit[1]*nonzeroy + \
                                   self.best_fit[2] - margin
            search_area_x_high = self.best_fit[0]*nonzeroy**2 + \
                                   self.best_fit[1]*nonzeroy + \
                                   self.best_fit[2] + margin

            # now we don't need to build the windows
            lines_inds = ((nonzerox >= search_area_x_low) & (nonzerox <= search_area_x_high)).nonzero()[0]
            
        # gets the indexes of pixel that make up the line
        self.allx = nonzerox[lines_inds]
        self.ally = nonzeroy[lines_inds]
        
    def sanity_check(self, other_curverad, bases_left_right=(0, 0)):
        good_measurement = True
        
        # checking that they have similar curvature
        min_curvature_diff = self.radius_of_curvature - (self.radius_of_curvature/2)
        max_curvature_diff = self.radius_of_curvature + (self.radius_of_curvature/2)
        
        if (other_curverad < min_curvature_diff) | (other_curverad > max_curvature_diff):
            good_measurement = False
            
        # checking that they are separated by approximately the right distance horizontally
        xm_per_pix = 3.7/700
        dist_hor = (bases_left_right[1] - bases_left_right[0]) * xm_per_pix
        if dist_hor > 4 or dist_hor < 3:
            print('irregular distance horizontally', dist_hor)
            good_measurement = False
            
        if good_measurement:
            # if we gets a good measurement we reset the number of roolbacks
            self.n_rollbacks = 0
            
        return good_measurement
    
    def rollback(self, previous_state):
        self.detected = previous_state.detected  
        self.recent_xfitted = previous_state.recent_xfitted
        self.bestx = previous_state.bestx
        self.best_fit = previous_state.best_fit
        self.current_fit = previous_state.current_fit
        self.radius_of_curvature = previous_state.radius_of_curvature
        self.line_base_pos = previous_state.line_base_pos
        self.diffs = previous_state.diffs
        self.allx = previous_state.allx
        self.ally = previous_state.ally
        self.recent_fit = previous_state.recent_fit
        
        self.n_rollbacks += 1
        
        if self.n_rollbacks > 15:
            # init from stratch
            print('init search from stratch')
            self.n_rollbacks = 0
            self.detected = False

def detect_lines(image, left_line, right_line):
    
    # allow rollback in case of bad detection
    left_previous_state = copy.deepcopy(left_line)
    right_previous_state = copy.deepcopy(right_line)
    
    image = cv2.undistort(image, mtx, dist, None, mtx)
    
    grad_x = abs_sobel_thresh(image, thresh=(20, 130), sobel_kernel=5)
    #grad_x = abs_sobel_thresh(img_sobel, thresh=(20, 130), sobel_kernel=9)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]

    thresh = (100, 200)
    h_binary = np.zeros_like(H)
    h_binary[(H > thresh[0]) & (H <= thresh[1])] = 1


    #project
    #s_tresh = hls_threshold(image, thresh=(130, 255))
    s_tresh = hls_threshold(image, thresh=(90, 255))

    combined = np.zeros_like(s_tresh)
    combined[((s_tresh == 1) & (s_tresh != h_binary)) | ((grad_x == 1) ) ] = 1

    imshape = image.shape

    rows = imshape[0]
    cols = imshape[1]

    x_bl, x_tl, x_tr, x_br, y_top = extract_points_mask(image)

    # points that will be used in source points to transform
    #offset = 40
    offset = -15
    new_left_x = x_tl + offset
    new_left_y = find_y_line(x_bl, x_tl, y_top, rows, new_left_x)
    new_right_x = x_tr - offset
    new_right_y = find_y_line(x_br, x_tr, y_top, rows, new_right_x)

    src = np.float32([
        [x_bl, rows], 
        [new_left_x, new_left_y], 
        [new_right_x, new_right_y], 
        [x_br, rows] 
    ])

    # increase the size of line (mask area)
    increase_value = 150
    x_left = x_bl + increase_value
    x_right = x_br - increase_value

    dest = np.float32([
        [x_left, rows], 
        [x_left, 0],
        [x_right, 0], 
        [x_right, rows]
    ])

    warped_img, Minv = warp(combined, src, dest)
     
    half_index = warped_img.shape[0]//2
    histogram = np.sum(warped_img[half_index:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # gets the nonzero indices in an array (1 dim) inner a tuple (0=dim1 and 1=dim2)
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # get indexes of line
    left_line.get_indexes_of_line(warped_img, nonzero, leftx_base)
    right_line.get_indexes_of_line(warped_img, nonzero, rightx_base)
   
    # gets a vector of coefficients of deegre 2
    # pass the y, x and the deegre
    left_line.set_current_fit(np.polyfit(left_line.ally, left_line.allx, 2))
    right_line.set_current_fit(np.polyfit(right_line.ally, right_line.allx, 2))

    # generates x and y values
    start = 0
    stop = warped_img.shape[0]-1
    nsamples = warped_img.shape[0]
    # returns an array of numbers for y
    ploty = np.linspace(start, stop, nsamples)

    # polynomial f(y) = a*yˆ2 + b*y + c
    left_line.add_xfitted(left_line.best_fit[0]*ploty**2 + \
                          left_line.best_fit[1]*ploty + \
                          left_line.best_fit[2])
    right_line.add_xfitted(right_line.best_fit[0]*ploty**2 + \
                          right_line.best_fit[1]*ploty + \
                          right_line.best_fit[2])

    # measuring the curvature

    # maximum y-value, corresponding to the bottom of the image
    y_eval = warped_img.shape[0]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/warped_img.shape[0] # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # polynomial coefficients with meters
    left_fit_m = np.polyfit(left_line.ally*ym_per_pix, left_line.allx*xm_per_pix, 2)
    right_fit_m = np.polyfit(right_line.ally*ym_per_pix, right_line.allx*xm_per_pix, 2)

    y_eval_m = y_eval*ym_per_pix

    # calculates the new radii of curvature
    left_line.radius_of_curvature = ((1 + (2*left_fit_m[0]*y_eval_m + left_fit_m[1])**2)**1.5) / np.absolute(2*left_fit_m[0])
    right_line.radius_of_curvature = ((1 + (2*right_fit_m[0]*y_eval_m + right_fit_m[1])**2)**1.5) / np.absolute(2*right_fit_m[0])
    
    polycolor = (0,255, 0)    
    
    if (left_line.sanity_check(
            right_line.radius_of_curvature, bases_left_right=(leftx_base,rightx_base)) == False):
        
        if left_previous_state.bestx is not None: # if we have previous state
            left_line.rollback(left_previous_state)
        if right_previous_state.bestx is not None: # if we have previous state
            right_line.rollback(right_previous_state)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_(pts), polycolor)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
   
    # print texts

    # curvature
    mean_curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) // 2
    
    # distance from center
    center_point_lane = (leftx_base + rightx_base) / 2
    center_image = cols // 2
    distance = (center_image - center_point_lane) * xm_per_pix
    
    ## for debug (shows center of lane and the distance from center)
    ## draw_lines(result, [[center_image, 0, center_image, rows]], color=(255, 0, 0))
    ## draw_lines(result, [[int(center_point_lane), 0, int(center_point_lane), rows]], color=(0, 0, 255))
    
    
    text_curvature = "Radius of Curvature = {0:.0f}(m)".format(mean_curvature)
    text_distance = "Vehicle is {0:.2f}m {1} of center".format(
        abs(distance), "right" if distance > 0 else "left")
    
    cv2.putText(result, text_curvature, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(result, text_distance, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
    return result
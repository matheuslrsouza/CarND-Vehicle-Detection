import numpy as np
import cv2
from skimage.feature import hog

# functions extracted from Udacity's lessons
def convert_color(img, conv='RGB'):
    
    if conv != 'RGB':
        if conv == 'HSV':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif conv == 'LUV':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif conv == 'HLS':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif conv == 'YUV':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif conv == 'YCrCb':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: 
        converted = np.copy(img)

    return converted

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        '''
        print('params hog')
        print('orient', orient)
        print('pixels_per_cell', pix_per_cell)
        print('cell_per_block', cell_per_block) 
        print('vis', vis)
        print('feature_vec', feature_vec)
        '''

        return features

def bin_spatial(img, size=(32, 32), feature_vector=False):
    
    if feature_vector == True:        
        color1 = cv2.resize(img[:,:,0], size).ravel()
        color2 = cv2.resize(img[:,:,1], size).ravel()
        color3 = cv2.resize(img[:,:,2], size).ravel()
        return np.hstack((color1, color2, color3))
    else:
        return cv2.resize(img, size)
                        
def color_hist(img, nbins=32, bins_range=(0, 255)):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


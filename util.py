# Thai Thien
# 1351040
from __future__ import division
import math
import numpy as np
from scipy import signal
import scipy.stats as st
import cv2
def test():
    print('success')

## convert color image to grayscale image
def bgr2gray(img):
    ret = np.zeros((img.shape[0], img.shape[1]),
              dtype = img.dtype)

    for (x,y), v in np.ndenumerate(ret):
        intensity = img[x,y,0]/3 + img[x,y,1]/3 + img[x,y,2]/3
        ret[x,y] = intensity

    return ret

# return n x n Gaussian Mask with sigma
def calGaussianKernel(sigma,n):
    if (sigma==0):
        sigma = 0.000001
    sigma2 = sigma*sigma
    ret = np.zeros((n,n))
    fraction = (1)/(2*math.pi*sigma2)
    for (x, y),v in np.ndenumerate(ret):
        xx = x - n//2
        xx = xx*xx
        yy = y - n//2
        yy = yy*yy
        ef = (xx+yy)/(2*sigma2)
        ef = -ef
        ret[x,y] = fraction * math.exp(ef)
    return ret

## calculate
def calGaussianKernel2(sigma, n):
    interval = (2*sigma+1.)/(n)
    x = np.linspace(-sigma-interval/2., sigma+interval/2., n+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def smooth_one_channel(value,dim,img, kernel):
    ret = signal.convolve2d(img, kernel, mode='same')
    return ret

def smooth_verry_slow(value,dim,img):
    mask =  calGaussianKernel2(value,dim)
    ret0 = smooth_one_channel(value,dim,img[:,:,0],mask)
    ret1 = smooth_one_channel(value, dim, img[:, :, 1], mask)
    ret2 = smooth_one_channel(value, dim, img[:, :, 2], mask)
    ret = img.copy()
    ret[:, :, 0] = ret0
    ret[:, :, 0] = ret1
    ret[:, :, 0] = ret2
    return ret

def smooth_fast(value, dim, img):
    value = 255 - value
    kernel = calGaussianKernel2(5, dim)
    ret = cv2.filter2D(img,-1,kernel)
    return ret

def smooth(value,dim,img):
    kernel = calGaussianKernel2(value*0.1, dim)
    ret = signal.convolve2d(img, kernel, mode='same')
    ret = ret.astype(np.uint8)
    return ret


def filter(img):
    ret = img.copy()
    kernel = np.ones((3,3))/9
    ret = signal.convolve2d(img, kernel, mode='same')
    ret = ret.astype(np.uint8)
    print(ret.dtype)
    print(img.dtype)
    return ret

def derivative(img, mode):
    if (mode == 'x'):
        sobel = np.matrix([[-1,0,1],[-2,0,2],[-1,0,1]])
    else:
        sobel = np.matrix([[1,2,1],[0,0,0],[-1,-2,-1]])
    ret = signal.convolve2d(img,sobel,mode='same')
    #   ret = ret.astype(np.uint8) # normalize
    return ret

''' m - show the magnitude of the gradient normalized to the range [0,255]. The gradient is
computed based on the x and y derivatives of the image.'''
def magnitude(img, mode = 1):
    if (mode == 0):
        derivative_x = derivative(img,'x').astype(np.float64)
        derivative_y = derivative(img,'y').astype(np.float64)
        print('derivative')
    else:
        derivative_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        derivative_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        print ('cv2 sobel')
    ret = cv2.magnitude(derivative_x,derivative_y)
    ret = ret.astype(np.uint8)
    return ret
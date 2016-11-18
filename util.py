# Thai Thien
# 1351040

import numpy as np
import cv2
def test():
    print('success')

def bgr2gray(img):
    ret = np.zeros((img.shape[0], img.shape[1]),
              dtype = img.dtype)

    for (x,y), v in np.ndenumerate(ret):
        intensity = img[x,y,0]/3 + img[x,y,1]/3 + img[x,y,2]/3
        ret[x,y] = intensity

    return ret

def smooth_cv(value, img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return blur

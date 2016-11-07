import cv2

from matplotlib import pyplot as plt
import numpy as np

status = 0

# load image
cat = cv2.imread('cat.jpg')

cur_img = cat.copy()

# display result

cv2.imshow('current',cur_img)
#for r in result:
#   cv2.imshow('result'+str(index),r)
#   index+=1



while(1):
    key = cv2.waitKey()
    if key == ord('i'):
        cur_img = cat.copy()
        print 'i'
    elif key == ord('g'):
        cur_img = cv2.cvtColor(cur_img,cv2.COLOR_BGR2GRAY)
    elif key == ord('c'):
        b, g, r = cv2.split(cat)
        if status == 0:
            channel = b
            status = 1
        elif status == 1:
            channel = g
            status = 2
        elif status == 2:
            channel = r
            status = 0
        print (status)
        cur_img = np.zeros((cat.shape[0], cat.shape[1], 3),
              dtype = cat.dtype)
        cur_img[:,:,status] = channel

    elif key == ord('w'):
        cv2.imwrite('img.png',cur_img)
        print 'w'
    elif key == ord('q'):
        print 'q'
        quit()

    cv2.imshow('current', cur_img)
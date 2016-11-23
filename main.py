# Thai Thien
# 1351040

import cv2
import util

from matplotlib import pyplot as plt
import numpy as np
global cur_img # global value cur_img
global grayscale
cur_img = None
cat = None


def nothing(a):
    print (a)

def main():
    global cur_img
    global cat
    global grayscale
    status = 0
    help = '''
i - Show original image
w - Save file as img.png into current directory
s - Smooth image. Drag the top bar to change the amount
S - A better way to smooth image. Drag the top bar to change the amount
G or g - turn image into grayscale.
c - display image in green, red, blue
x - Sobel filter in x direction
y - Sobel filter in y direction
M or m - display magnitude of gradient.
r - rotate mode. Drag the track bar to rotate the image.
q - quit
    '''
    # load image
    cat = cv2.imread('cat.jpg')
    grayscale = cv2.cvtColor(cat,cv2.COLOR_BGR2GRAY)
    cur_img = cat.copy()

    cv2.imshow('image',cur_img)

    while(1):
        cv2.namedWindow('image')
        key = cv2.waitKey()

        if key == ord('i'):
            cur_img = cat.copy()
            print 'i'

        elif key == ord('g'): #
            cur_img = cv2.cvtColor(cat,cv2.COLOR_BGR2GRAY)

        elif key == ord('G'):
            cur_img = util.bgr2gray(cat)

        elif key == ord('s'):
            def callback(value):
                # use global variable because we can only pass in one parameter
                global cur_img
                global cat
                cur_img = cv2.GaussianBlur(cat, (3, 3), value)
                cv2.imshow('image', cur_img)
                if (value==0):
                    cv2.imshow('image',cat) # display original image when value = 0
            cv2.createTrackbar('Smooth',"image",0,255, callback)

        elif key == ord('S'):
            def callback(value):
                # use global variable because we can only pass in one parameter
                global cur_img
                global cat
                cur_img = util.smooth(value,10,grayscale)
                cv2.imshow('image', cur_img)
                if (value==0):
                    cv2.imshow('image',cat) # display original image when value = 0
            cv2.createTrackbar('Smooth',"image",0,100, callback)

        elif key == ord('x'):
            cur_img = util.derivative(grayscale,'x', True)


        elif key == ord('y'):
            cur_img = util.derivative(grayscale,'y', True)


        elif key == ord('m'):
            cur_img = util.magnitude(grayscale,0)
        elif key == ord('M'):
            cur_img = util.magnitude(grayscale,1)

        elif key == ord('r'):
            def callback(value):
                # use global variable because we can only pass in one parameter
                global grayscale
                global cur_img
                cur_img = util.nohole_rotation(grayscale,value)
                cv2.imshow('image', cur_img)

            cv2.createTrackbar('Rotation',"image",0,360, callback)


        elif key == ord('c'): #
            c1 = cat[:,:,0]
            c2 = cat[:,:,1]
            c3 = cat[:,:,2]
            if status == 0:
                channel = c1
                status = 1
            elif status == 1:
                channel = c2
                status = 2
            elif status == 2:
                channel = c3
                status = 0

            cur_img = np.zeros((cat.shape[0], cat.shape[1], 3),
                  dtype = cat.dtype)
            cur_img[:,:,status] = channel

        elif key == ord('w'):
            cv2.imwrite('img.png',cur_img)
            print 'w'

        elif key == ord('h'):
            print help

        elif key == ord('q'):
            print 'q'
            quit()

        cv2.imshow('image', cur_img)



if __name__ == '__main__':
    main()


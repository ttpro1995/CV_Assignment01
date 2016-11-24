# Thai Thien
# 1351040

import pytest
import cv2
import util
class TestUtil():

    def test_bgr2gray(self):
        cat = cv2.imread('cat.jpg')
        cur_img = util.bgr2gray(cat)
        assert cur_img.dtype =='uint8'

    def test_derivative(self):
        cat = cv2.imread('cat.jpg')
        cat = util.bgr2gray(cat)
        cur_img_x = util.derivative(cat, mode='x')
        cur_img_y = util.derivative(cat, mode='y')

    def test_smooth(self):
        cat = cv2.imread('cat.jpg')
        cat = util.bgr2gray(cat)
        cur_img_1 = util.smooth(30, 3, cat)
        cur_img_2 = util.smooth(60, 5, cat)
        cur_img_3 = util.smooth(0, 3, cat)
        assert cur_img_1.dtype == 'uint8'
        assert cur_img_2.dtype == 'uint8'
        assert cur_img_3.dtype == 'uint8'

    def test_rotate(self):
        cat = cv2.imread('cat.jpg')
        cat = util.bgr2gray(cat)
        for i in range(0,360):
            ret = util.rotation(cat,i)

    def test_plot(self):
        cat = cv2.imread('cat.jpg')
        grayscale = util.bgr2gray(cat)
        cur_img = util.plotGradVec(grayscale, n = 20)
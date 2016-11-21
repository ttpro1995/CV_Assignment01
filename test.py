# Thai Thien
# 1351040

import unittest
from lib import cv2
import util
class TestStringMethods(unittest.TestCase):

    def test_bgr2gray(self):
        cat = cv2.imread('cat.jpg')
        cur_img = util.bgr2gray(cat)
        self.assertEquals(cur_img.dtype,'uint8')

    def test_derivative(self):
        cat = cv2.imread('cat.jpg')
        cat = util.bgr2gray(cat)
        cur_img_x = util.derivative(cat, mode='x')
        cur_img_y = util.derivative(cat, mode='y')
        self.assertEquals(cur_img_x.dtype, 'uint8')
        self.assertEquals(cur_img_y.dtype, 'uint8')

    def test_smooth(self):
        cat = cv2.imread('cat.jpg')
        cat = util.bgr2gray(cat)
        cur_img_1 = util.smooth(30, 3, cat)
        cur_img_2 = util.smooth(60, 5, cat)
        cur_img_3 = util.smooth(0, 3, cat)
        self.assertEquals(cur_img_1.dtype, 'uint8')
        self.assertEquals(cur_img_2.dtype, 'uint8')
        self.assertEquals(cur_img_3.dtype, 'uint8')

if __name__ == '__main__':
    unittest.main()
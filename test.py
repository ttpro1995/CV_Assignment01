# Thai Thien
# 1351040

import unittest
import cv2
import util
class TestStringMethods(unittest.TestCase):

    def test_bgr2gray(self):
        global cur_img
        global cat
        cat = cv2.imread('cat.jpg')
        cur_img = util.bgr2gray(cat)

if __name__ == '__main__':
    unittest.main()
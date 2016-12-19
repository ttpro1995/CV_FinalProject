from detector import Detector
from lbp import LBP
import cv2

HARRIS = 'harris'
BLOB = 'blob'
DOG = 'dog'
class FeatureExtractor:
    def __init__(self):
        self._lbp = LBP()
        self._detector = Detector()
        self._sift = cv2.SIFT()

    def extract(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        des = self._lbp.compute_img(img)
        return des


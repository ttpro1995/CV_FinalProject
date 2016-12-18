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

    def extract(self, img, detector_type = HARRIS):
        if (detector_type==HARRIS):
            kp, img = self._detector.harris(img)
        elif (detector_type==BLOB):
            kp, img = self._detector.blob(img)
        elif (detector_type == DOG):
            kp, img = self._detector.dog(img)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self._sift.compute(gray_img,kp)
        return des


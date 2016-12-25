import cv2
from skimage.feature import local_binary_pattern
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.radius = 3
        self.n_point = self.radius*8
        self.METHOD = 'uniform'

    def extract(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray_img, self.n_point, self.radius, self.METHOD)
        n_bins = lbp.max() + 1
        hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
        return hist


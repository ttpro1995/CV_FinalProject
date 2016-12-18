# Thai Thien
# 1351040
import numpy as np
import cv2

class LBP:
    def compute(self, img, keypoints):
        img = np.asarray(img)
        img = (1 << 7) * (img[0:-2, 0:-2] >= img[1:-1, 1:-1]) \
            + (1 << 6) * (img[0:-2, 1:-1] >= img[1:-1, 1:-1]) \
            + (1 << 5) * (img[0:-2, 2:] >= img[1:-1, 1:-1]) \
            + (1 << 4) * (img[1:-1, 2:] >= img[1:-1, 1:-1]) \
            + (1 << 3) * (img[2:, 2:] >= img[1:-1, 1:-1]) \
            + (1 << 2) * (img[2:, 1:-1] >= img[1:-1, 1:-1]) \
            + (1 << 1) * (img[2:, :-2] >= img[1:-1, 1:-1]) \
            + (1 << 0) * (img[1:-1, :-2] >= img[1:-1, 1:-1])
        res = []
        for x in keypoints:
            rows = int(x.pt[1] - 1)
            cols = int(x.pt[0] - 1)
            size = int(x.size)
            rows_range = (max(0, rows - size), min(img.shape[0], rows + size + 1))
            cols_range = (max(0, cols - size), min(img.shape[1], cols + size + 1))
            window = img[rows_range[0]:rows_range[1], cols_range[0]:cols_range[1]].flatten()
            hist = np.histogram(window, bins=range(257))
            res.append(hist[0])
        res = np.array(res)
        res = np.reshape(res, (-1, 256))
        res = np.uint8(res)
        return keypoints, res


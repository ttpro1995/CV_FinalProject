# Thai Thien
# 1351040

import cv2
import numpy as np
import cv2

class Detector:
    def __init__(self):
        pass

    def harris(self, img):
        '''
        Harris detector
        :param img: an color image
        :return: feature, image with feature marked corner
        '''

        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        dst = cv2.cornerHarris(gray_img, 2, 3, 0.04)
        result_img = img.copy() # deep copy image
        # result is dilated for marking the corners, not important
        # dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        result_img[dst > 0.01 * dst.max()] = [0, 0, 255]

        # for each dst larger than threshold, make a keypoint out of it
        keypoints = np.argwhere(dst > 0.01 * dst.max())
        keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints]

        return (keypoints, result_img)

    def blob(self, img):
        '''
        Blob detector
        :param img:
        :return: feature, image with circle
        '''

        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0;
        params.maxThreshold = 12000;

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 1000

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.2

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        blob_detector = cv2.SimpleBlobDetector(params)
        print('gray_img type ', gray_img.dtype)
        keypoints = blob_detector.detect(gray_img)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return (keypoints, im_with_keypoints)

    def dog(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g1 = cv2.GaussianBlur(gray_img,(7,7),1)
        g2 = cv2.GaussianBlur(gray_img,(7,7),5)
        dif = cv2.absdiff(g1, g2)
        ret, dif = cv2.threshold(dif, 24, 255, cv2.THRESH_BINARY)
        # dif = cv2.dilate(dif, None)
        result_img = img.copy()  # deep copy image
        result_img[dif > 0.01 * dif.max()] = [0, 0, 255]

        sift = cv2.SIFT()
        kp = sift.detect(img)

        return kp, result_img
# Thai Thien 1351040

import cv2
import copyright
import numpy as np
from os import listdir
from os.path import isfile, join, splitext
import os.path

class PreProcessor:
    def __init__(self, face_xml, eye_xml, mouth_xml, nose_xml):
        self.face_cascade = cv2.CascadeClassifier(face_xml)
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)
        self.nose_cascade = cv2.CascadeClassifier(nose_xml)
        self.mouth_cascade = cv2.CascadeClassifier(mouth_xml)
        self.kernel = np.ones((5, 5), np.float32) / 25
    '''
    Convert image to black scale
    detect face in image
    return a array of face size 96x96
    '''
    def process_file(self, filename, size_dim, landmark = False):
        """
        :param filename: raw data file name
        :return:
        ret: an array each element is tuple (roi_face, roi_eyes, roi_nose, roi_mouth)
        """
        print ('process file ', filename)
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Color -> grayscale
        gray = cv2.GaussianBlur(gray,(5,5),0)  # blur gaussian
        ret = []
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_eyes = []
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (size_dim, size_dim))   # resize image to 96x96
            roi_face = roi_gray

            if (landmark):
                # detect eye
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey,ew,eh) in eyes:
                    roi_eye = roi_face[ey:ey + eh, ex:ex + ew]
                    roi_eyes.append(roi_eye)

                # detect nose
                nose = self.nose_cascade.detectMultiScale(roi_gray)
                nx, ny, nw, nh = nose[0]
                roi_nose = roi_face[ny:ny + nh, nx:nx + nw]

                # detect mouth
                mouth = self.mouth_cascade.detectMultiScale(roi_gray)
                mx, my, mw, mh = mouth[0]
                roi_mouth = roi_face[my:my + mh, mx:mx + mw]

                sample = (roi_face, roi_eyes, roi_nose, roi_mouth)
            else:
                sample = (roi_face, None)

            ret.append(sample)
        return ret

    def display_ret(self,ret):
        """
        Display the result of preprocess_file function
        :param ret: output of preprocess_file function
        :return: nothing
        """
        face, eyes, nose, mouth = ret[0]
        cv2.imshow('face',face)
        cv2.imshow('left_eye', eyes[0])
        cv2.imshow('right_eye', eyes[1])
        cv2.imshow('nose', nose)
        cv2.imshow('mouth', mouth)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def preprocess(self, in_dir, out_dir, size_dim):
        inputs = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]
        for filename in inputs:
            outputs = self.process_file(in_dir+'/'+filename, size_dim)
            for output_img in outputs:
                output_img = output_img[0] # only the face
                cv2.imwrite(out_dir+'/'+filename, output_img)

    def preprocess_landmark(self, in_dir, out_dir):
        """
        Preprocess file and get landmark of eye, nose, mouth
        :param in_dir:
        :param out_dir:
        :return:
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        inputs = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]
        for filename in inputs:
            name, extension = splitext(filename)
            subdir = out_dir+'/'+name # contain landmark
            if not os.path.exists(out_dir+'/'+name):
                os.makedirs(out_dir+'/'+name)
            outputs = self.process_file(in_dir+'/'+filename, size_dim=96, landmark=True)
            for roi_face, roi_eyes, roi_nose, roi_mouth in outputs:
                cv2.imwrite(out_dir+'/'+filename, roi_face)
                cv2.imwrite(subdir + '/' + 'eye0.tiff', roi_eyes[0])
                cv2.imwrite(subdir + '/' + 'eye1.tiff', roi_eyes[1])
                cv2.imwrite(subdir + '/' + 'nose.tiff', roi_nose)
                cv2.imwrite(subdir + '/' + 'mouth.tiff', roi_mouth)


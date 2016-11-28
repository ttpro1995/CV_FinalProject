# Thai Thien 1351040

import cv2
import copyright
from os import listdir
from os.path import isfile, join

class PreProcessor:
    def __init__(self, pretrain_dataset):
        self.face_cascade = cv2.CascadeClassifier(pretrain_dataset)

    '''
    Convert image to black scale
    detect face in image
    return a array of face size 96x96
    '''
    def process_file(self, filename):
        print ('process file ', filename)
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret = []
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (96, 96))   # resize image to 96x96
            ret.append(roi_gray)
        return ret

    def preprocess(self, in_dir, out_dir):
        inputs = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]
        for filename in inputs:
            outputs = self.process_file(in_dir+'/'+filename)
            for output_img in outputs:
                cv2.imwrite(out_dir+'/'+filename, output_img)




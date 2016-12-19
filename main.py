# Thai Thien 1351040

from preprocessor import PreProcessor
import cv2
import copyright
from os import listdir
import os
from os.path import isfile, join
import util
from sklearn.model_selection import train_test_split
import numpy as np

from feature_extraction.extractor import FeatureExtractor

PROCESSED_DATA_DIR = './ProcessedData'
dataset_path = 'numpy_dataset'



def main():
    CASCADE_CLASSIFIER_FILE = 'haarcascade_frontalface_default.xml'
    # m_pre = PreProcessor(CASCADE_CLASSIFIER_FILE)
    # m_pre.preprocess('./RawData', './ProcessedData')
    data = []
    labels = []
    if os.path.isfile(dataset_path):
        dataset = np.load(dataset_path)
        data = dataset['data']
        labels = dataset['labels']

    else:
        # extraction
        feature_extractor = FeatureExtractor()
        files = [f for f in listdir(PROCESSED_DATA_DIR) if isfile(join(PROCESSED_DATA_DIR, f))]
        for file in files:
            img = cv2.imread(PROCESSED_DATA_DIR+'/'+file)
            feaVec = feature_extractor.extract(img)
            data.append(feaVec)
            label = util.jaffe_labeling(file)
            labels.append(label)
        np.savez(dataset_path, data=data, labels = labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.33, random_state = 42)





    print ('breakpoint')


if __name__ == '__main__':
    main()


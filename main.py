# Thai Thien 1351040

from preprocessor import PreProcessor
import cv2
import copyright
from os import listdir
from os.path import isfile, join
import util

from feature_extraction.extractor import FeatureExtractor

PROCESSED_DATA_DIR = './ProcessedData'
def main():
    CASCADE_CLASSIFIER_FILE = 'haarcascade_frontalface_default.xml'
    # m_pre = PreProcessor(CASCADE_CLASSIFIER_FILE)
    # m_pre.preprocess('./RawData', './ProcessedData')

    # extraction
    feature_extractor = FeatureExtractor()
    files = [f for f in listdir(PROCESSED_DATA_DIR) if isfile(join(PROCESSED_DATA_DIR, f))]
    data = []
    labels = []
    for file in files:
        img = cv2.imread(PROCESSED_DATA_DIR+'/'+file)
        feaVec = feature_extractor.extract(img, detector_type='dog')
        data.append(feaVec)
        label = util.jaffe_labeling(file)
        labels.append(label)

    print ('breakpoint')


if __name__ == '__main__':
    main()


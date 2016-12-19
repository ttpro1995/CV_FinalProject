# Thai Thien 1351040

from preprocessor import PreProcessor
import cv2
import copyright
from os import listdir
from os.path import isfile, join
import util
from sklearn.model_selection import train_test_split

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
        feaVec = feature_extractor.extract(img)
        data.append(feaVec)
        label = util.jaffe_labeling(file)
        labels.append(label)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.33, random_state = 42)



    print ('breakpoint')


if __name__ == '__main__':
    main()


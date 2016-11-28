# Thai Thien 1351040

from preprocessor import PreProcessor
import cv2
import copyright

def main():
    CASCADE_CLASSIFIER_FILE = 'haarcascade_frontalface_default.xml'
    m_pre = PreProcessor(CASCADE_CLASSIFIER_FILE)
    m_pre.preprocess('./RawData', './ProcessedData')

if __name__ == '__main__':
    main()


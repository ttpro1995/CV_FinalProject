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
import sklearn.svm as svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
import optunity
import optunity.metrics

from feature_extraction.extractor import FeatureExtractor

PROCESSED_DATA_DIR = './ProcessedData'
dataset_path = 'numpy_dataset'

def evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    print (clf)
    # evaluate svm
    print ('evaluate on training set')
    pred_train = clf.predict(X_train)
    correct = np.equal(pred_train, y_train)
    n_correct = len(correct[correct == True])
    acc = float(n_correct) / len(y_train)
    print ('correct %d on total %d' % (n_correct, len(y_train)))
    print ('acc = ', acc)

    # evaluate svm
    print ('evaluate on test set')
    pred_test = clf.predict(X_test)
    correct = np.equal(pred_test, y_test)
    n_correct = len(correct[correct == True])
    acc = float(n_correct) / len(y_test)
    print ('correct %d on total %d' % (n_correct, len(y_test)))
    print ('acc = ', acc)



def main():
    CASCADE_CLASSIFIER_FILE = 'haarcascade_frontalface_default.xml'
    #m_pre = PreProcessor(CASCADE_CLASSIFIER_FILE)
    #m_pre.preprocess('./RawData', './ProcessedData')
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

    # data = normalize(data, norm='l1', axis=0)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.20)

    # hyps tuning
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5 ],
                         'C': [1, 10, 100, 1000, 10000, 100000], 'decision_function_shape':['ovo']},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000,10000, 100000], 'decision_function_shape':['ovo']}]

    scores = ['precision', 'recall']

    # train svm
    #clf = svm.SVC(C=1000, gamma=1, kernel='linear', decision_function_shape = 'ovo', verbose=False, class_weight='balanced')

    #clf.fit(X_train, y_train)
    #print (clf.n_support_.shape)
    #evaluate_classifier(clf, X_train, X_test, y_train, y_test)

    for score in scores:
        print ('######################')
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        evaluate_classifier(clf,X_train,X_test,y_train,y_test)
        print()

    print ('breakpoint')


if __name__ == '__main__':
    main()


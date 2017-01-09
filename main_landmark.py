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

from feature_extraction.extractor import FeatureExtractor

PROCESSED_GROUP_DATA_DIR = './ProcessedDataGroup'
RAW_DATA_DIR = './RawData'
dataset_path = 'numpy_dataset'
FACE_XML = 'haarcascade_frontalface_default.xml'
MOUTH_XML = 'haarcascade_mcs_mouth.xml'
NOSE_XML = 'haarcascade_mcs_nose.xml'
EYE_XML = 'haarcascade_eye.xml'





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

    '''
     Prerpocessing:
     Crop face in each raw data image and resize to 96x96
     The preprocessed image are saved into ./PreprocessedData
     raw image in ./RawData folder
     processed image would be output to ./ProcessedData
     Please create an empty ProcessedData folder if it is not exist
    '''
    ### Uncomment this block to do preprocessing
    # m_pre = PreProcessor(FACE_XML,EYE_XML,MOUTH_XML,NOSE_XML)
    # m_pre.preprocess(RAW_DATA_DIR, PROCESSED_GROUP_DATA_DIR, size_dim=96)
    # m_pre.preprocess_landmark(RAW_DATA_DIR, PROCESSED_GROUP_DATA_DIR)
    ######################################################################

    data = []
    labels = []
    #if os.path.isfile(dataset_path):
    if (False):
        print ('load old feature')
        dataset = np.load(dataset_path)
        data = dataset['data']
        labels = dataset['labels']
    else:
        # extraction feature of each part (eyes, nose, mouth and concat feature vector)
        print ('Extract feature')
        feature_extractor = FeatureExtractor()
        folders = [f for f in listdir(PROCESSED_GROUP_DATA_DIR) if os.path.isdir(PROCESSED_GROUP_DATA_DIR+"/"+f)]
        for folder in folders:
            feaVec = feature_extractor.extract_group(PROCESSED_GROUP_DATA_DIR+"/"+folder)
            data.append(feaVec)
            label = util.jaffe_labeling(folder)
            labels.append(label)
        # np.savez(dataset_path, data=data, labels = labels)


    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.20)

    # hyps tuning
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5 ],
                         'C': [1, 10, 100, 1000, 10000, 100000], 'decision_function_shape':['ovo']},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000,10000, 100000], 'decision_function_shape':['ovo']}
                        ]

    tuned_param_rbf = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5 ],
                         'C': [1, 10, 100, 1000, 10000, 100000], 'decision_function_shape':['ovo']}
                        ]
    tuned_params_poly = [{'kernel': ['poly'], 'gamma': [1e-3, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5 ],
                         'C': [1, 10, 100, 1000, 10000, 100000], 'decision_function_shape':['ovo'],
                          'degree':[1,2,3]}
                        ]

    tuned_param_linear = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000,10000, 100000], 'decision_function_shape':['ovo']}]

    tuned_params = [tuned_param_linear, tuned_param_rbf, tuned_params_poly]
    scores = ['precision', 'recall']
    score = 'recall'

    # train svm
    #clf = svm.SVC(C=1000, gamma=1, kernel='linear', decision_function_shape = 'ovo', verbose=False, class_weight='balanced')

    #clf.fit(X_train, y_train)
    #print (clf.n_support_.shape)
    #evaluate_classifier(clf, X_train, X_test, y_train, y_test)

    for score in scores:
        for tuned in tuned_params:
            print ('###############################################################')
            print("# Tuning hyper-parameters for %s using kernel %s" % (score ,tuned[0]['kernel']))
            print()

            clf = GridSearchCV(svm.SVC(C=1), tuned, cv=5,
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
            print('###################################################')

    print ('breakpoint')


if __name__ == '__main__':
    main()


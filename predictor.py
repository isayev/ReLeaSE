from __future__ import print_function
from __future__ import division

import numpy as np

from sklearn.ensemble import RandomForestRegressor as RFC
from sklearn.externals import joblib
from sklearn import metrics
from data import Data

data = Data()


class RandomForestQSAR(object):
    def __init__(self, n_estimators=100, n_ensemble=5):
        super(RandomForestQSAR, self).__init__()
        self.n_estimators = n_estimators
        self.n_ensemble = n_ensemble
        self.classifiers = []
        for i in range(n_ensemble):
            self.classifiers.append(RFC(n_estimators=n_estimators))

    def load_model(self, path):
        self.classifiers = []
        for i in range(self.n_ensemble):
            clf = joblib.load(path + str(i) + '.pkl')
            self.classifiers.append(clf)

    def save_model(self, path):
        assert self.n_ensemble == len(self.classifiers)
        for i in range(self.n_ensemble):
            joblib.dump(self.classifiers[i], path + str(i) + '.pkl')

    def fit_model(self):
        auc = []
        for i in range(self.n_ensemble):
            train_sm = np.concatenate(data.cross_val_data[:i] + data.cross_val_data[(i + 1):])
            test_sm = data.cross_val_data[i]
            train_labels = np.concatenate(data.cross_val_labels[:i] + data.cross_val_labels[(i + 1):])
            test_labels = data.cross_val_labels[i]
            fp_train = data.get_fp(train_sm)
            fp_test = data.get_fp(test_sm)
            self.classifiers[i].fit(fp_train, train_labels.ravel())
            predicted = self.classifiers[i].predict(fp_test)
            fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted)
            auc.append(metrics.auc(fpr, tpr))
        return auc

    def predict(self, smiles, average=True):
        fps = data.get_fp(smiles)
        assert len(smiles) == len(fps)
        clean_smiles = []
        clean_fps = []
        nan_smiles = []
        for i in range(len(fps)):
            if np.isnan(sum(fps[i])):
                nan_smiles.append(smiles[i])
            else:
                clean_smiles.append(smiles[i])
                clean_fps.append(fps[i])
        clean_fps = np.array(clean_fps)
        prediction = []
        for clf in self.classifiers:
            prediction.append(clf.predict(clean_fps))
        prediction = np.array(prediction)
        if average:
            prediction = prediction.mean(axis=1)
        to_return = np.concatenate((np.array(clean_smiles), prediction), axis=1)
        return to_return, nan_smiles

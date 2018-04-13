from __future__ import print_function
from __future__ import division

import numpy as np

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.externals import joblib
from sklearn import metrics

from data import get_fp


class RandomForestQSAR(object):
    def __init__(self, model_type='classifier', n_estimators=100, n_ensemble=5):
        super(RandomForestQSAR, self).__init__()
        self.n_estimators = n_estimators
        self.n_ensemble = n_ensemble
        self.model = []
        self.model_type = model_type
        if self.model_type == 'classifier':
            for i in range(n_ensemble):
                self.model.append(RFC(n_estimators=n_estimators))
        elif self.model_type == 'regressor':
            for i in range(n_ensemble):
                self.model.append(RFR(n_estimators=n_estimators))
        else:
            raiseValueError('invalid value for argument')

    def load_model(self, path):
        self.model = []
        for i in range(self.n_ensemble):
            m = joblib.load(path + str(i) + '.pkl')
            self.model.append(m)

    def save_model(self, path):
        assert self.n_ensemble == len(self.model)
        for i in range(self.n_ensemble):
            joblib.dump(self.model[i], path + str(i) + '.pkl')

    def fit_model(self, data, cross_val_data, cross_val_labels):
        eval_metrics = []
        for i in range(self.n_ensemble):
            train_sm = np.concatenate(cross_val_data[:i] + cross_val_data[(i + 1):])
            test_sm = cross_val_data[i]
            train_labels = np.concatenate(cross_val_labels[:i] + cross_val_labels[(i + 1):])
            test_labels = cross_val_labels[i]
            fp_train = get_fp(train_sm)
            fp_test = get_fp(test_sm)
            self.model[i].fit(fp_train, train_labels.ravel())
            predicted = self.model[i].predict(fp_test)
            if self.model_type == 'classifier':
                fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted)
                eval_metrics.append(metrics.auc(fpr, tpr))
                metrics_type = 'AUC'
            elif self.model_type == 'regressor':
                r2 = metrics.r2_score(test_labels, predicted)
                eval_metrics.append(r2)
                metrics_type = 'R^2 score'
        return eval_metrics, metrics_type

    def predict(self, smiles, average=True):
        fps = get_fp(smiles)
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
        if len(clean_fps) > 0:
            for m in self.model:
                prediction.append(m.predict(clean_fps))
            prediction = np.array(prediction)
            if average:
                prediction = prediction.mean(axis=0)
        assert len(clean_smiles) == len(prediction)
        #to_return = np.concatenate((np.array(clean_smiles).reshape(-1, 1), prediction.reshape(-1, 1)), axis=1)
        return clean_smiles, prediction, nan_smiles

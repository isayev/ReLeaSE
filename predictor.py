from __future__ import print_function
from __future__ import division

import numpy as np

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn import metrics

from data import get_fp, get_desc, normalize_desc, cross_validation_split


class RandomForestQSAR(object):
    def __init__(self, model_type='classifier', n_estimators=100, n_ensemble=5):
        super(RandomForestQSAR, self).__init__()
        self.n_estimators = n_estimators
        self.n_ensemblea = n_ensemble
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
        return clean_smiles, prediction, nan_smiles


class VanillaQSAR(object):
    def __init__(self, model_instance=None, model_params=None,
                 model_type='classifier', ensemble_size=5, normalization=False):
        super(VanillaQSAR, self).__init__()
        self.model_instance = model_instance
        self.model_params = model_params
        self.ensemble_size = ensemble_size
        self.model = []
        self.normalization = normalization
        if model_type not in ['classifier', 'regressor']:
            raise InvalidArgumentError("model type must be either"
                                       "classifier or regressor")
        self.model_type = model_type
        if isinstance(self.model_instance, list):
            assert(len(self.model_instance) == self.ensemble_size)
            assert(isinstance(self.model_params, list))
            assert(len(self.model_params) == self.ensemble_size)
            for i in range(self.ensemble_size):
                self.model.append(self.model_instance[i](**model_params[i]))
        else:
            for _ in range(self.ensemble_size):
                self.model.append(self.model_instance(**model_params))
        if self.normalization:
            self.desc_mean = [0]*self.ensemble_size
        self.metrics_type = None

    def fit_model(self, data, cv_split='stratified'):
        eval_metrics = []
        x = data.x
        if self.model_type == 'classifier' and data.binary_y is not None:
            y = data.binary_y
        else:
            y = data.y
        cross_val_data, cross_val_labels = cross_validation_split(x=x, y=y,
                                                                  split=cv_split,
                                                                  n_folds=self.ensemble_size)
        for i in range(self.ensemble_size):
            train_x = np.concatenate(cross_val_data[:i] +
                                     cross_val_data[(i + 1):])
            test_x = cross_val_data[i]
            train_y = np.concatenate(cross_val_labels[:i] +
                                     cross_val_labels[(i + 1):])
            test_y = cross_val_labels[i]
            if self.normalization:
                train_x, desc_mean = normalize_desc(train_x)
                self.desc_mean[i] = desc_mean
                test_x, _ = normalize_desc(test_x, desc_mean)
            self.model[i].fit(train_x, train_y.ravel())
            predicted = self.model[i].predict(test_x)
            if self.model_type == 'classifier':
                eval_metrics.append(metrics.f1_score(test_y, predicted))
                self.metrics_type = 'F1 score'
            elif self.model_type == 'regressor':
                r2 = metrics.r2_score(test_y, predicted)
                eval_metrics.append(r2)
                self.metrics_type = 'R^2 score'
            else:
                raise RuntimeError()
        return eval_metrics, self.metrics_type

    def load_model(self, path):
        # TODO: add iterable path object instead of static path 
        self.model = []
        for i in range(self.ensemble_size):
            m = joblib.load(path + str(i) + '.pkl')
            self.model.append(m)
        if self.normalization:
            arr = np.load(path + 'desc_mean.npy')
            self.desc_mean = arr

    def save_model(self, path):
        assert self.n_ensemble == len(self.model)
        for i in range(self.n_ensemble):
            joblib.dump(self.model[i], path + str(i) + '.pkl')
        if self.normalization:
            np.save(path + 'desc_mean.npy', self.desc_mean)

    def predict(self, objects=None, average=True, get_features=None,
                **kwargs):
        objects = np.array(objects)
        invalid_objects = []
        processed_objects = []
        if get_features is not None:
            x, processed_indices, invalid_indices = get_features(objects,
                                                                 **kwargs)
            processed_objects = objects[processed_indices]
            invalid_objects = objects[invalid_indices]
        else:
            x = objects
        if len(x) == 0:
            processed_ojects = []
            prediction = []
            invalid_objects = objects
        else:
            prediction = []
            for i in range(self.ensemble_size):
                m = self.model[i]
                if self.normalization:
                    x, _ = normalize_desc(x, self.desc_mean[i])
                prediction.append(m.predict(x))
            prediction = np.array(prediction)
            if average:
                prediction = prediction.mean(axis=0)
        return processed_objects, prediction, invalid_objects


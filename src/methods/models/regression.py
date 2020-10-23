"""Implementation of various regression models."""
import abc
import logging
import numpy as np
from sklearn.svm import SVR
from sklearn import linear_model, kernel_ridge
from six import with_metaclass

from .base import SKLearnModel
from ...util.container_ops import get_dim_size

__all__ = ['AveragePredictor', 'KernelRidgeRegression', 'LassoRegression',
    'LinearRegression', 'RidgeRegression', 'RidgeCVRegression', 'SVMRegression', 'ZeroPredictor']

__author__ = 'Otilia Stretcu'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# SKLearn models.
class SKLearnRegressionModel(with_metaclass(abc.ABCMeta, SKLearnModel)):
    """Wrapper for SKLearn models."""
    def __init__(self, model):
        super(SKLearnRegressionModel, self).__init__(model)
        self._predict = self._sk_learner.predict

    @abc.abstractmethod
    def __str__(self):
        pass

    def predict(self, inputs):
        predictions = self._sk_learner.predict(inputs)

        # Do this to make sure predictions always has at least 2 dimensions
        # (because we want the feature dimension to exist).
        if len(predictions.shape) == 1:
            return predictions[...,  np.newaxis]
        return predictions


class LinearRegression(SKLearnRegressionModel):
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1):
        linear_regression = linear_model.LinearRegression(
            fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X,
            n_jobs=n_jobs)
        super(LinearRegression, self).__init__(linear_regression)

    def __str__(self):
        return 'lin_reg_%r_%r_%r_%d' % (
            self._sk_learner.fit_intercept, self._sk_learner.normalize,
            self._sk_learner.copy_X, self._sk_learner.n_jobs)

    @staticmethod
    def class_name():
        return 'linear_reg'


class LassoRegression(SKLearnRegressionModel):
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000, tol=0.0001,
                 warm_start=False, positive=False, random_state=None,
                 selection='cyclic'):
        lasso = linear_model.Lasso(
            alpha=alpha, fit_intercept=fit_intercept, normalize=normalize,
            precompute=precompute, copy_X=copy_X, max_iter=max_iter, tol=tol,
            warm_start=warm_start, positive=positive, random_state=random_state,
            selection=selection)
        super(LassoRegression, self).__init__(lasso)

    def __str__(self):
        return 'lasso_%6.4f_%r_%r_%r_%r_%d_%6.4f_%r_%r_%s_%s' % (
            self._sk_learner.alpha, self._sk_learner.fit_intercept,
            self._sk_learner.normalize, self._sk_learner.precompute,
            self._sk_learner.copy_X, self._sk_learner.max_iter,
            self._sk_learner.tol, self._sk_learner.warm_start,
            self._sk_learner.positive, self._sk_learner.random_state,
            self._sk_learner.selection)

    @staticmethod
    def class_name():
        return 'lasso'


class RidgeRegression(SKLearnRegressionModel):
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=None, tol=0.001, solver='auto',
                 random_state=None):
        ridge_reg = linear_model.Ridge(
            alpha=alpha, fit_intercept=fit_intercept, normalize=normalize,
            copy_X=copy_X, max_iter=max_iter, tol=tol, solver=solver,
            random_state=random_state)
        super(RidgeRegression, self).__init__(ridge_reg)

    def __str__(self):
        return 'ridge_reg_%s.4f_%r_%r_%r_%s_%6.4f_%s_%s' % (
            str(self._sk_learner.alpha), self._sk_learner.fit_intercept,
            self._sk_learner.normalize, self._sk_learner.copy_X,
            str(self._sk_learner.max_iter), self._sk_learner.tol,
            str(self._sk_learner.solver), str(self._sk_learner.random_state))

    @staticmethod
    def class_name():
        return 'ridge'

    def parameters(self):
        return self._sk_learner.coef_


class RidgeCVRegression(SKLearnRegressionModel):
    def __init__(self, alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False,
                 scoring=None, cv=None, gcv_mode=None, store_cv_values=False):
        ridge_reg = linear_model.RidgeCV(
            alphas=alphas, fit_intercept=fit_intercept, normalize=normalize,
            scoring=scoring, cv=cv, gcv_mode=gcv_mode, store_cv_values=store_cv_values)
        super(RidgeCVRegression, self).__init__(ridge_reg)

    def __str__(self):
        print("cv_values_ shape: ", self._sk_learner.cv_values_.shape)
        print("alpha_: ", self._sk_learner.alpha_)
        return 'ridgecv_reg_%s.4f_%r_%r' % (
            str(self._sk_learner.alphas), self._sk_learner.fit_intercept,
            self._sk_learner.normalize, )

    @staticmethod
    def class_name():
        return 'ridgecv'


class KernelRidgeRegression(SKLearnRegressionModel):
    def __init__(self, alpha=1.0, kernel='linear', gamma=None, degree=3,
                 coef0=1, kernel_params=None):
        ridge_reg = kernel_ridge.KernelRidge(
            alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0,
            kernel_params=kernel_params)
        super(KernelRidgeRegression, self).__init__(
            ridge_reg)

    def __str__(self):
        return 'kernel_ridge_reg_%6.4f_%s_%s_%d_%6.4f_%s' % (
            self._sk_learner.alpha, self._sk_learner.kernel,
            str(self._sk_learner.gamma), self._sk_learner.degree,
            self._sk_learner.coef0, str(self._sk_learner.kernel_params))

    @staticmethod
    def class_name():
        return 'kernel_ridge'


class SVMRegression(SKLearnRegressionModel):
    def __init__(self, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1):

        model = SVR(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                    tol=tol, C=C, epsilon=epsilon, shrinking=shrinking,
                    cache_size=cache_size, verbose=verbose, max_iter=max_iter)
        super(SVMRegression, self).__init__(model)

    def __str__(self):
        return 'svm_reg_%s_%d_%s_%6.4f_%6.4f_%.2f_%.4f_%s_%d_%s_%d' % (
            self._sk_learner.kernel, self._sk_learner.degree,
            str(self._sk_learner.gamma), self._sk_learner.coef0,
            self._sk_learner.tol, self._sk_learner.C, self._sk_learner.epsilon,
            str(self._sk_learner.shrinking), self._sk_learner.cache_size,
            str(self._sk_learner.verbose), self._sk_learner.max_iter)

    def copy(self):
        return SVMRegression(
            kernel=self._sk_learner.kernel, degree=self._sk_learner.degree,
            gamma=self._sk_learner.gamma, coef0=self._sk_learner.coef0,
            tol=self._sk_learner.tol, C=self._sk_learner.C,
            epsilon=self._sk_learner.epsilon,
            shrinking=self._sk_learner.shrinking,
            cache_size=self._sk_learner.cache_size,
            verbose=self._sk_learner.verbose,
            max_iter=self._sk_learner.max_iter)

    @staticmethod
    def class_name():
        return 'svm'


# Baseline models.
class AveragePredictor(with_metaclass(abc.ABCMeta)):
    """Model that always predicts the average of the training data for each feature."""
    def __init__(self):
        self.avg = None
        super(AveragePredictor, self).__init__()

    def train(self, inputs, outputs):
        self.avg = np.mean(outputs, axis=0)[None, ...]

    def predict(self, inputs):
        predictions = np.repeat(self.avg, get_dim_size(inputs, 0), axis=0)
        return predictions

    def parameters(self):
        return []

    def __str__(self):
        return 'avg_pred'

    @staticmethod
    def class_name():
        return 'avg_pred'


class ZeroPredictor(with_metaclass(abc.ABCMeta)):
    """Model that always predicts 0."""
    def __init__(self):
        self.num_output_feat = None
        super(ZeroPredictor, self).__init__()

    def train(self, inputs, outputs):
        self.num_output_feat = get_dim_size(outputs, -1)

    def predict(self, inputs):
        predictions = np.zeros((get_dim_size(inputs, 0), self.num_output_feat))
        return predictions

    def parameters(self):
        return []

    def __str__(self):
        return 'zero_pred'

    @staticmethod
    def class_name():
        return 'zero_pred'

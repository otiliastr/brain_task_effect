from __future__ import absolute_import, division, print_function

import copy
import itertools
import logging
import numpy as np

from tabulate import tabulate

from ..data.preprocessing import normalize
from ..methods.metrics import (Accuracy2v2_SensorTimepoint, Accuracy2v2_4Outputs_SensorTimepoint,
                               DistPerTimePoint, MetricAggregator)
from ..util.container_ops import pretty_print, slice_data

__author__ = 'Otilia Stretcu'


def normalize_train_test(train_inputs, train_outputs, test_inputs, test_outputs,
                         normalize_inputs, normalize_outputs):
    """Normalize data."""
    def _normalize(train_d, test_d):
        train_d_normalized, mean_train, std_train = normalize(
            np.asarray(train_d), axis=0, return_offset=True)
        test_d_normalized = normalize(
            np.asarray(test_d), axis=0, offset=mean_train, scale=std_train)
        return train_d_normalized, test_d_normalized

    if normalize_inputs:
        print('Normalizing inputs...')
        train_inputs, test_inputs = _normalize(train_inputs, test_inputs)
    if normalize_outputs:
        print('Normalizing outputs...')
        train_outputs, test_outputs = _normalize(train_outputs, test_outputs)
    return train_inputs, train_outputs, test_inputs, test_outputs


class FoldResults(object):
    """Container for storing the results for each fold."""
    def __init__(self, model, params, train_results, test_results, train_predictions,
                 test_predictions, fold):
        self.model = model
        self.params = params
        self.train_results = train_results
        self.test_results = test_results
        self.train_predictions=train_predictions
        self.test_predictions = test_predictions
        self.fold = fold

    def _summarize_metric(self, results, metric_name):
        if metric_name not in results:
            return '-'
        if isinstance(results[metric_name], (list, tuple, np.ndarray)):
            return '[long array]'
        return results[metric_name]

    def summarize(self, suffix=''):
        """Print a table with the train and test results for one fold.

        Since the train and test metrics can be different, we will print a table with the union
        of all metrics. We will print a `-` for the missing metrics.
        """
        if self.train_results:
            all_metrics = sorted(list(set(
                self.train_results.keys()).union(set(self.test_results.keys()))))
            train_all_results = \
                ['Train'] + \
                [self._summarize_metric(self.train_results, metric) for metric in all_metrics]
        else:
            all_metrics = sorted(list(set(self.test_results.keys())))
        test_all_results = \
            ['Test'] + \
            [self._summarize_metric(self.test_results, metric) for metric in all_metrics]
        tabular_data = [train_all_results, test_all_results] if self.train_results else [test_all_results]
        table = tabulate(tabular_data=tabular_data, headers=[''] + all_metrics)
        table_width = table.find('\n')
        print('\n' + '-' * table_width)
        print('Fold %d ' % self.fold + suffix)
        print('-' * table_width)
        print(table)
        print('-' * table_width + '\n')


class Trainer(object):
    """A class that trains a model using cross-validation on the provided dataset.

    An experiment includes cross-validation based evaluation, and potentially also
    cross-validation-based parameter tuning if multiple values are passed for the same parameter.
    """
    def __init__(self, model, model_params, metrics, train_data,
                 cross_val_test=None,
                 cross_val_param_valid=None,
                 validation_metric=None,
                 validation_metric_higher_better=True,
                 max_folds_param_valid=np.inf,
                 normalize_inputs=False,
                 normalize_outputs=False,
                 postproc_predictions=(),
                 metric_aggregator_train=None,
                 metric_aggregator_test=None,
                 seed=None,
                 compute_train_metrics=False,
                 num_unique_train_words_to_keep=None,
                 seed_remove_words=123,
                 **kwargs):
        """Creates a Trainer object.

        Args:
            model: A model class (not instantiated) which can be obe of the models defined in
                src/methods/models.py.
            model_params (dict or list(dict)): One of two options:
                (i) Dictionary containing all the constructor arguments for the provided learner.
                    For each argument, there should be a corresponding key with its name in the
                    dictionary. For each such key, the corresponding value should be a list of
                    potential values for that argument. The optimal set of values will be
                    determined by the runner via cross-validation.
                (ii) List containing one dictionary per constructor arguments configuration.
                    The keys in these dictionaries are the constructor argument names and the
                    values are their corresponding values.
            metrics: A list of Metrics, as defined in methods/metrics.py
            train_data: A tuple containing (train_inputs, train_outputs, sample_groups), where
                train_inputs is a numpy array of shape (num_samples, num_input_features)
                train_outputs is a numpy array of shape (num_samples, num_output_features)
                sample_groups is a list of indices representing some identifier of the sample.
            cross_val_test: A cross-validation data iterator, defined in src/methods/iterators.py
            cross_val_param_valid: A cross-validation data iterator, defined in
                src/methods/iterators.py
            validation_metric: A Metric used for validation.
            validation_metric_higher_better: A boolean specifying whether for the chosen
                validation_metric, a higher value is better.
            max_folds_param_valid: An integer representing the maximum number of folds to test when
                choosing the best hyper-parameters in the inner cross-validation loop.
            normalize_inputs: Boolean specifying whether the inputs should be normalized in each
                fold before passing them to the model.
            normalize_outputs: Boolean specifying whether the targets should be normalized in each
                fold before computing the loss.
            postproc_predictions: A list of functions, which are applied to the model predictions as
                a post-processing step, before computing the metrics.
            metric_aggregator_train: A MetricAggregator that keeps track of the results on the
                training data.
            metric_aggregator_test: A MetricAggregator that keeps track of the results on the
                test data.
            seed: A seed used for the random number generators.
            compute_train_metrics: Boolean specifying if we want to compute all metrics on the train
                data as well (which is more time-consuming) or skip this step.
            num_unique_train_words_to_keep:  Keep only num_unique_train_words_to_keep words for
                training. Randomly pick their indices. This is intended for training and testing
                the model with less data.
            seed_remove_words: A seed used when randomly picking the num_unique_train_words_to_keep.
            **kwargs:
        """
        self.model_class = model

        # Get every combination of parameters and create a learner config.
        self.model_params = [
            dict(kwargs) for kwargs in itertools.product(
                *[[(name, value) for value in values]
                  for name, values in model_params.items()])]

        self.cross_val_param_val = cross_val_param_valid
        self.cross_val_test = cross_val_test
        self.validation_metric = validation_metric
        self.validation_metric_higher_better = validation_metric_higher_better
        self.max_folds_param_valid = max_folds_param_valid
        self.train_data = train_data
        self.metrics_test = metrics
        self.metrics_val = [self.validation_metric]
        self.seed = seed
        self.postproc_predictions = postproc_predictions
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.compute_train_metrics = compute_train_metrics
        self.num_unique_train_words_to_keep = num_unique_train_words_to_keep
        self.seed_remove_words = seed_remove_words

        self.metric_aggregator_train = metric_aggregator_train \
            if metric_aggregator_train else MetricAggregator(self.metrics_test, name='TRAIN')
        self.metric_aggregator_test = metric_aggregator_test \
            if metric_aggregator_test else MetricAggregator(self.metrics_test, name='TEST')

    def train_cross_val(self, callbacks, print_results=False):
        """Train using cross-validation.

        Args:
            callbacks: A list of callback functions to run on the results and predictions of each
                fold.
            print_results: Boolean specifying whether to print intermediate results to the console.

        Returns:
            A list of callback results, containing the output of each callback function.
        """
        inputs = self.train_data[0]
        outputs = self.train_data[1]
        group_ids = np.asarray(self.train_data[2])

        # Since the test data is not provided, we do cross-validation to split into train/test.
        logging.info('Running cross-validation...')
        cross_val = self.cross_val_test(inputs=inputs, targets=outputs, groups=group_ids)
        callback_results = []
        fold = 0

        if self.num_unique_train_words_to_keep:
            # Keep only num_unique_train_words_to_keep words for training.
            # Randomly pick their indices.
            rng = np.random.RandomState(self.seed_remove_words)
            word_ids = [g_id // 100 for g_id in group_ids]
            words_to_keep = set(
                rng.choice(word_ids, size=self.num_unique_train_words_to_keep, replace=False))

        for train_indices, test_indices in cross_val:
            print('-------------------------------------------------------------')
            print('                OUTER FOLD %d                                ' % fold)
            train_inputs = slice_data(inputs, train_indices)
            train_outputs = slice_data(outputs, train_indices)
            test_inputs = slice_data(inputs, test_indices)
            test_outputs = slice_data(outputs, test_indices)
            train_groups = slice_data(group_ids, train_indices)
            test_groups = slice_data(group_ids, test_indices)

            if self.num_unique_train_words_to_keep:
                # Keep only the words with ids in words_to_keep.
                keep_indices = np.asarray(
                    [i for i, g in enumerate(train_groups) if g // 100 in words_to_keep])
                train_inputs = train_inputs[keep_indices]
                train_outputs = train_outputs[keep_indices]
                train_groups = train_groups[keep_indices]

            # Normalize data, making sure the test data is not used to compute the mean and std.
            if self.normalize_inputs or self.normalize_outputs:
                train_inputs, train_outputs, test_inputs, test_outputs = normalize_train_test(
                    train_inputs, train_outputs, test_inputs, test_outputs,
                    self.normalize_inputs, self.normalize_outputs)

            # Train a model.
            fold_results = self.train_fold(
                train_inputs, train_outputs, test_inputs, test_outputs, train_groups, test_groups,
                self.model_params, fold, self.metrics_test)

            # Save and print results.
            if print_results:
                print('-------------------------------------------------------------')
                print('               RESULTS FOR OUTER FOLD %d                     ' % fold)
                print('-------------------------------------------------------------')
                fold_results.summarize()
            if fold_results.train_results:
                self.metric_aggregator_train.record_values(fold_results.train_results)
            self.metric_aggregator_test.record_values(fold_results.test_results)

            # Run potential callbacks.
            results = []
            for callback in callbacks:
                result = callback(
                    fold_results.model, train_inputs, train_outputs, train_groups,
                    test_inputs, test_outputs, test_groups, fold, fold_results.train_results,
                    fold_results.test_results, fold_results.train_predictions,
                    fold_results.test_predictions)
                results.append(result)
            callback_results.append(results)

            fold += 1

        # Print and save the results of the best models.
        if print_results:
            print('Results summary over all folds choosing the best learner per fold.')
            if self.compute_train_metrics:
                self.metric_aggregator_train.log_summary()
            self.metric_aggregator_test.log_summary()

        return callback_results

    def train_fold(self, train_inputs, train_outputs, test_inputs, test_outputs,
                   train_groups, test_groups, model_params, fold, metrics):
        """Trains a single cross-validation fold.

        Args:
            train_inputs: A numpy array of shape (num_samples_train, num_input_features)
                representing the training sample inputs.
            train_outputs: A numpy array of shape (num_samples_train, num_output_features)
                representing the training sample outputs.
            test_inputs: A numpy array of shape (num_samples_test, num_input_features) representing
                the test sample inputs.
            test_outputs: A numpy array of shape (num_samples_test, num_output_features)
                representing the test sample outputs.
            train_groups: A numpy array of shape (num_samples_train,) representing
                the train sample group indices.
            test_groups: A numpy array of shape (num_samples_test,) representing
                the test sample group indices.
            model_params: A dictionary containing the hyperparameters of the model to be trained.
            fold: An integer representing the fold number.
            metrics: A list of Metrics to evaluate on the test data.

        Returns:
            A FoldResults object.

        """
        # Validate parameters and pick the best learner configuration.
        model_params = self.pick_best_params(
            train_inputs, train_outputs, train_groups, model_params)

        # Train fold using the best parameters.
        logging.info('Training model...')
        model = self.model_class(**model_params)
        model.train(train_inputs, train_outputs)

        logging.info('Evaluating model...')
        if self.compute_train_metrics:
            train_predictions, train_results = self.evaluate(
                model, train_inputs, train_outputs, train_groups, metrics)
        else:
            train_predictions, train_results = None, None
        test_predictions, test_results = self.evaluate(
            model, test_inputs, test_outputs, test_groups, metrics)

        fold_results = FoldResults(
            model=model,
            params=model_params,
            train_results=train_results,
            test_results=test_results,
            train_predictions=train_predictions,
            test_predictions=test_predictions,
            fold=fold)

        return fold_results

    def pick_best_params(self, inputs, outputs, groups, model_params):
        """If we have multiple learner configurations, we do cross-validation to pick the best one.

        Args:
            inputs: A numpy array of shape (num_samples_train, num_input_features)
                representing the training sample inputs.
            outputs: A numpy array of shape (num_samples_train, num_output_features)
                representing the training sample outputs.
            groups: A numpy array of shape (num_samples_train,) representing
                the train sample group indices.
            model_params: A dictionary containing the possible model hyper-parameters.

        Returns:
            A dictionary containing only the best hyper-parameter configuration.
        """
        # If we have a single learner configuration, that is the best model.
        if len(model_params) == 1:
            return model_params[0]

        logging.info('Picking the best model parameters...')

        # Do multiple train/test splits with cross validation to determine the best params.
        result_for_params = []
        for params in model_params:
            cross_val = self.cross_val_param_val(inputs=inputs, targets=outputs, groups=groups)
            logging.info('Running experiment with params: %s.' % pretty_print(params))
            results = []

            # Create results aggregators.
            val_agg_test = MetricAggregator(
                [self.validation_metric],
                name='Cross-validation for parameter selection. Results on TEST data for params: %s'
                     % pretty_print(params))

            fold = 0
            for train_indices, test_indices in cross_val:
                fold_results = self.train_validation_fold(
                    inputs, outputs, groups, train_indices, test_indices, fold, [params],
                    summarize_results=False)
                results.append(fold_results)
                val_agg_test.record_values(fold_results.test_results)
                fold += 1
                logging.info('Number of param validation folds processed: %d', fold)
                if fold >= self.max_folds_param_valid:
                    break

            val_agg_test.log_summary()
            result_for_current_params = val_agg_test.metric_mean(self.validation_metric)
            result_for_params.append(result_for_current_params)

        # Pick the best learner configuration.
        best_index = np.argmax(result_for_params) if self.validation_metric_higher_better else \
                     np.argmin(result_for_params)
        logging.info('Best model has params {%s} and performance: %s.' %
                     (pretty_print(model_params[best_index]),
                      pretty_print(result_for_params[best_index])))
        return model_params[best_index]

    def train_validation_fold(self, inputs, outputs, groups, train_indices, test_indices, fold,
                              model_params, summarize_results=False):
        train_inputs = slice_data(inputs, train_indices)
        train_outputs = slice_data(outputs, train_indices)
        test_inputs = slice_data(inputs, test_indices)
        test_outputs = slice_data(outputs, test_indices)
        train_groups = slice_data(groups, train_indices)
        test_groups = slice_data(groups, test_indices)

        logging.info('Processing learner validation fold %d.' % fold)
        fold_results = self.train_fold(
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            train_groups=train_groups,
            test_groups=test_groups,
            model_params=model_params,
            fold=fold,
            metrics=self.metrics_val)
        if summarize_results:
            fold_results.summarize(suffix=' PARAM VALIDATION')
        return fold_results

    def evaluate(self, model, inputs, targets, groups, metrics):
        predictions = model.predict(inputs)
        predictions = self._apply_prediction_postprocess(predictions, groups)
        results = {
            metric.get_name(): metric(predictions, targets, None, groups)
            for metric in metrics}
        return predictions, results

    def _apply_prediction_postprocess(self, predictions, groups=None):
        result = predictions
        for op in self.postproc_predictions:
            result = op(result) if groups is None else op(result, groups)
        return result


class TrainerMultiRegularization(Trainer):
    """A trainer that picks a different regularization hyper-parameter for each output."""

    def __init__(self, model, model_params, metrics, train_data,
                 cross_val_test=None,
                 cross_val_param_valid=None,
                 validation_metric=None,
                 validation_metric_higher_better=True,
                 max_folds_param_valid=np.inf,
                 normalize_inputs=False,
                 normalize_outputs=False,
                 fold_preproc_ops=(),
                 postproc_predictions=(),
                 metric_aggregator_train=None,
                 metric_aggregator_test=None,
                 seed=None,
                 compute_train_metrics=False,
                 param_name_reg='alpha',
                 best_params_save_path='outputs/best_params.txt',
                 num_unique_train_words_to_keep=0):
        super(TrainerMultiRegularization, self).__init__(
            model, model_params, metrics, train_data,
            cross_val_test=cross_val_test,
            cross_val_param_valid=cross_val_param_valid,
            validation_metric=validation_metric,
            validation_metric_higher_better=validation_metric_higher_better,
            max_folds_param_valid=max_folds_param_valid,
            normalize_inputs=normalize_inputs,
            normalize_outputs=normalize_outputs,
            fold_preproc_ops=fold_preproc_ops,
            postproc_predictions=postproc_predictions,
            metric_aggregator_train=metric_aggregator_train,
            metric_aggregator_test=metric_aggregator_test,
            seed=seed,
            compute_train_metrics=compute_train_metrics,
            num_unique_train_words_to_keep=num_unique_train_words_to_keep)

        self.param_name_reg = param_name_reg
        self.best_params_save_path = best_params_save_path

    def pick_best_params(self, inputs, outputs, groups, model_params):
        """If we have multiple learner configurations, we do cross-validation to pick the best one.

        Args:
            inputs: A numpy array of shape (num_samples_train, num_input_features)
                representing the training sample inputs.
            outputs: A numpy array of shape (num_samples_train, num_output_features)
                representing the training sample outputs.
            groups: A numpy array of shape (num_samples_train,) representing
                the train sample group indices.
            model_params: A dictionary containing the possible model hyper-parameters.

        Returns:
            A dictionary containing only the best hyper-parameter configuration.
        """
        # If we have a single learner configuration, that is the best model.
        if len(model_params) == 1:
            return model_params[0]

        logging.info('Picking the best model parameters...')

        assert self.param_name_reg in model_params[0]
        scalar_reg_param = []

        # Do multiple train/test splits with cross validation to determine the best params.
        result_for_params = []
        for params in model_params:
            # Convert regularizaton into a matrix, with one value per output.
            scalar_reg_param.append(params[self.param_name_reg])
            params = copy.deepcopy(params)
            params[self.param_name_reg] = params[self.param_name_reg] * np.ones_like(outputs[0])

            cross_val = self.cross_val_param_val(inputs=inputs, targets=outputs, groups=groups)
            logging.info('Running experiment with params: %s.' % pretty_print(params))
            results = []

            # Create results aggregators.
            val_agg_test = MetricAggregator(
                self.metrics_val,
                name='Cross-validation for parameter selection. Results on TEST data for params: %s'
                     % pretty_print(params))

            fold = 0
            for train_indices, test_indices in cross_val:
                fold_results = self.train_validation_fold(
                    inputs, outputs, groups, train_indices, test_indices, fold, [params],
                    summarize_results=False)
                results.append(fold_results)
                val_agg_test.record_values(fold_results.test_results)
                fold += 1
                logging.info('Number of param validation folds processed: %d', fold)
                if fold >= self.max_folds_param_valid:
                    break

            val_agg_test.log_summary()

            # Temporary hack to aggregate distance per time.
            result_for_current_params = val_agg_test.get_metric(self.validation_metric.name)
            if issubclass(self.validation_metric.__class__, DistPerTimePoint):
                if self.validation_metric.average_folds:
                    result_for_current_params = np.stack(result_for_current_params)
                else:
                    result_for_current_params = np.concatenate(result_for_current_params)
                result_for_current_params = np.mean(result_for_current_params, axis=0)
                result_for_current_params = result_for_current_params.flatten()
            elif isinstance(self.validation_metric,
                            (Accuracy2v2_4Outputs_SensorTimepoint, Accuracy2v2_SensorTimepoint)):
                result_for_current_params = np.mean(np.stack(result_for_current_params), axis=0)
                result_for_current_params = result_for_current_params.flatten()

            result_for_params.append(result_for_current_params)

        # Pick the best learner configuration.
        if isinstance(result_for_params[0], np.ndarray):
            #############################################
            # This part is hardcoded for selecting a different alpha for every sensor-timepoint.
            # Select a different parameter for each element.
            logging.info('Picking the best regularization param per sensor-timepoint...')
            best_index = np.argmax(result_for_params, axis=0) if self.validation_metric_higher_better else \
                np.argmin(result_for_params, axis=0)
            best_params = copy.deepcopy(model_params[0])
            scalar_reg_param = np.asarray(scalar_reg_param)
            best_params[self.param_name_reg] = scalar_reg_param[best_index]
            with open(self.best_params_save_path, 'a') as file:
                np.savetxt(file, best_params[self.param_name_reg], delimiter=',', newline=',')
                file.write('\n')
            print('Saved best parameters at: ', self.best_params_save_path)
            ##############################################
        else:
            best_index = np.argmax(result_for_params) if self.validation_metric_higher_better else \
                np.argmin(result_for_params)
            best_params = model_params[best_index]

        logging.info('Best model has params: {%s}.' % pretty_print(best_params))
        return best_params

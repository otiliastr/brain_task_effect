from __future__ import absolute_import, division, print_function

import abc
import copy
import itertools

import numpy as np
import scipy
import six

from collections import OrderedDict

from ..util.container_ops import get_num_dims

__all__ = ['L2Distance', 'L2DistanceWithMeta', 'MetricAggregator']
__author__ = 'Otilia Stretcu'


class Metric:
    def __init__(self, name):
        self.name = name

    def __call__(self, predictions, targets, *args):
        return self.evaluate(predictions, targets, *args)

    def __str__(self):
        return self.name

    @abc.abstractmethod
    def evaluate(self, predictions, targets, *args):
        pass

    def get_name(self):
        return self.name


class L2Distance(Metric):
    def __init__(self, dist_metric='euclidean', avg_dist_per_feature=False,
                 name='l2_dist'):
        super(L2Distance, self).__init__(name=name)
        self.dist_metric = dist_metric
        self.avg_dist_per_feature = avg_dist_per_feature

    def __str__(self):
        return self.name

    def evaluate(self, predictions, targets, *args):
        def _l2_norms(xs):
                return np.sqrt(np.sum(np.square(xs), axis=1))
            
        assert predictions.shape == targets.shape
        if get_num_dims(predictions) == 1:
            predictions = predictions[..., None]
        if get_num_dims(targets) == 1:
            targets = targets[..., None]
        if self.dist_metric == 'euclidean':
            # Euclidean distance.
            dist = _l2_norms(np.subtract(targets, predictions))
            if self.avg_dist_per_feature:
                return np.mean(dist) / len(targets[0])
            return np.mean(dist)
        elif self.dist_metric == 'cosine':
            # Cosine distance.
            similarities = np.divide(
                np.sum(np.multiply(predictions, targets), axis=1),
                np.multiply(_l2_norms(predictions), _l2_norms(targets)))
            if self.avg_dist_per_feature:
                return (1 - np.mean(similarities)) / len(targets[0])
            return 1 - np.mean(similarities)
        raise ValueError('Unsupported distance metric: %s.' % self.dist_metric)


# Metric Wrappers
class L2DistanceWithMeta(L2Distance):
    def __init__(self, dist_metric='euclidean', avg_dist_per_feature=False,
                 name='l2_dist'):
        super(L2DistanceWithMeta, self).__init__(
            dist_metric=dist_metric, avg_dist_per_feature=avg_dist_per_feature, name=name)

    def evaluate(self, predictions, targets, *args):
        predictions = np.asarray([p[0] for p in predictions])
        return super(L2DistanceWithMeta, self).evaluate(predictions, targets, *args)


# Metric aggregators.
class MetricAggregator(object):
    """Aggregates metrics (e.g., over multiple folds, iterations, etc.)."""
    def __init__(self, metrics, name=None):
        self.name = name
        self.metric_values = OrderedDict([(m.get_name(), []) for m in metrics])

    def record_value(self, metric, value):
        if metric in self.metric_values:
            self.metric_values[metric].append(value)
        else:
            self.metric_values[metric] = [value]

    def record_values(self, values):
        for k, v in six.iteritems(values):
            if isinstance(v, list):
                if k in self.metric_values:
                    self.metric_values[k].extend(v)
                else:
                    self.metric_values[k] = copy.deepcopy(v)
            else:
                self.record_value(k, v)

    def mean(self):
        return OrderedDict([(m, np.asscalar(np.mean(v)))
                            for m, v in six.iteritems(self.metric_values)])

    def metric_mean(self, metric):
        if isinstance(metric, str):
            metric_name = metric
        elif isinstance(metric, Metric):
            metric_name = metric.get_name()
        else:
            raise ValueError('Unsupported data type %s for metric'
                             % metric.__class__)
        assert metric_name in self.metric_values.keys(), \
            'Metric %s is not part of this metric aggregator.' % metric_name
        return np.asscalar(np.mean(self.metric_values[metric_name]))

    def std(self):
        return OrderedDict([(m, np.asscalar(np.std(v)))
                            for m, v in six.iteritems(self.metric_values)])

    def metric_std(self, metric):
        if isinstance(metric, str):
            metric_name = metric
        elif isinstance(metric, Metric):
            metric_name = metric.get_name()
        else:
            raise ValueError('Unsupported data type %s for metric'
                             % metric.__class__)
        assert metric_name in self.metric_values.keys(), \
            'Metric %s is not part of this metric aggregator.' % metric_name
        return np.asscalar(np.std(self.metric_values[metric_name]))

    def get_metrics(self):
        return list(self.metric_values.keys())

    def get_num_recordings(self):
        def _get_num_values(v):
            if isinstance(v, (list, tuple)):
                return len(v)
            if isinstance(v, np.array):
                return v.shape[0]
            raise ValueError('Unsupported data type %s.' % v.__class__)

        return OrderedDict([(m, _get_num_values(v))
                            for m, v in six.iteritems(self.metric_values)])

    def get_metric(self, metric):
        return copy.deepcopy(self.metric_values[metric])

    def get_metric_values(self):
        return copy.deepcopy(self.metric_values)

    def has_any_recordings(self):
        counts = self.get_num_recordings()
        return any([v > 0 for m, v in six.iteritems(counts)])

    def compute_summary(self):
        summary = OrderedDict()
        for metric, values in six.iteritems(self.metric_values):
            # For metrics such as distance per timepoint we might have multiple samples per fold,
            # so we have to concatenate before averaging.

            if metric == '2v2_any_q1_q2_sensortimepoint':
                summary[metric] = tuple([np.nanmean(values, axis=0), np.nanstd(values, axis=0)])
            else:
                if isinstance(values, (list, tuple, np.ndarray)) and isinstance(values[0], (list, tuple, np.ndarray)):
                    values = [np.float32(val) for val in values]
                    values = np.concatenate(values)

                summary[metric] = tuple(
                    [np.asscalar(np.nanmean(values)), np.asscalar(np.nanstd(values))])
        return summary

    def compute_summary_per_features(self):
        summary = OrderedDict()
        for metric, values in six.iteritems(self.metric_values):
            if len(values) > 0 and not np.isscalar(values[0]) and len(values[0]) > 0:
                # For metrics such as distance per timepoint we might have multiple samples per fold,
                # so we have to concatenate before averaging.
                values = np.concatenate(values)
                summary[metric] = tuple([np.nanmean(values, axis=0), np.nanstd(values, axis=0)])
            else:
                summary[metric] = tuple([np.asscalar(np.nanmean(values)), np.asscalar(np.nanstd(values))])
        return summary

    def log_summary(self):
        summaries = self.compute_summary()
        if self.name is None:
            print('=' * 8 + ' Metrics Summary ' + '=' * 8)
        else:
            print('=' * 8 + ' Metrics Summary for ' + self.name + ' ' + '=' * 8)
        print('| %11s | %4s | %4s |' % ('Name', 'Mean', 'Std'))
        for metric, summary in six.iteritems(summaries):
            if metric != '2v2_any_q1_q2_sensortimepoint':
                print('| %11s | %1.2f ± %1.2f |' % (metric, summary[0], summary[1]))
            else:
                print('| %11s | Saved to file |' % (metric))
        print('=' * 33)

    @staticmethod
    def combine_aggregators(metric_aggregators, name=None):
        combined_aggregator = MetricAggregator([], name=name)
        for agg in metric_aggregators:
            combined_aggregator.record_values(agg.get_metric_values())
        return combined_aggregator

    @staticmethod
    def aggregate_summaries(metric_aggregators):
        summaries = OrderedDict()
        for metric_aggregator in metric_aggregators:
            summary = metric_aggregator.compute_summary()
            for metric, s in six.iteritems(summary):
                if metric in summaries:
                    summaries[metric].append(s)
                else:
                    summaries[metric] = [s]
        result = OrderedDict()
        print('=' * 8 + ' Metrics Summary ' + '=' * 8)
        for metric, summary in six.iteritems(summaries):
            metric_mean_of_means = np.mean([s[0] for s in summary])
            metric_mean_of_std = np.mean([s[1] for s in summary])
            print('| %11s | %1.2f ± %1.2f |' %
                  (metric, metric_mean_of_means, metric_mean_of_std))
            result[metric] = (metric_mean_of_means, metric_mean_of_std)
        return result

    @staticmethod
    def aggregate_summaries_per_feature(metric_aggregators):
        summaries = OrderedDict()
        for metric_aggregator in metric_aggregators:
            summary = metric_aggregator.compute_summary_per_features()
            for metric, s in six.iteritems(summary):
                if metric in summaries:
                    summaries[metric].append(s)
                else:
                    summaries[metric] = [s]
        result = OrderedDict()
        print('=' * 8 + ' Metrics Summary Per Feature ' + '=' * 8)
        for metric, summary in six.iteritems(summaries):
            metric_mean_of_means = np.mean(np.array([s[0] for s in summary]), axis=0)
            metric_mean_of_std = np.mean(np.array([s[1] for s in summary]), axis=0)
            result[metric] = (metric_mean_of_means, metric_mean_of_std)
            if np.isscalar(metric_mean_of_means):
                print('| %11s | %1.2f ± %1.2f |' % (metric, metric_mean_of_means, metric_mean_of_std))
            else:
                output = '| %11s ' % metric
                # result[metric] = [(m,s)
                #     for m, s in zip(metric_mean_of_means, metric_mean_of_std)]
                for feat in range(len(metric_mean_of_means)):
                    output += '| %1.2f ± %1.2f ' % \
                              (metric_mean_of_means[feat], metric_mean_of_std[feat])
                print(output)
        return result


class DistPerTimePoint(Metric):
    """Distance per time-point.
    This metric assumes you're predicting brain activity in as a flattened array of length
    sensors*time. It will reshape it and compute the distance at every time points between the
    predicted [sensor1, sensor2...,sensorK] and the target values for [sensor1, sensor2...,sensorK].
    """
    def __init__(self, num_sensors, dist_metric='euclidean', name='dist_per_time',
                 average_folds=False):
        super(DistPerTimePoint, self).__init__(name=name)
        self.dist_metric = dist_metric
        self.num_sensors = num_sensors
        self.average_folds = average_folds

        assert dist_metric == 'euclidean', 'Only euclidean currently supported'

    def __str__(self):
        return self.name

    def evaluate(self, predictions, targets, *args):
        def _l2_norms(xs):
            return np.sqrt(np.sum(np.square(xs), axis=1))

        assert predictions.shape == targets.shape

        num_samples = predictions.shape[0]
        predictions = predictions.reshape(num_samples, self.num_sensors, -1)
        targets = targets.reshape(num_samples, self.num_sensors, -1)

        dists = np.square(predictions - targets)

        if self.average_folds:
            dists = np.mean(dists, axis=0)

        return dists


class DistPerTimePointWithMeta(DistPerTimePoint):
    """Wrapper for DistPerTimePoint when meta-data is provided."""
    def __init__(self, num_sensors, dist_metric='euclidean', name='dist_per_time',
                 average_folds=False):
        super(DistPerTimePointWithMeta, self).__init__(
            num_sensors, dist_metric=dist_metric, name=name, average_folds=average_folds)

    def __str__(self):
        return self.name

    def evaluate(self, predictions, targets, *args):
        predictions = np.asarray([p[0] for p in predictions])
        return super(DistPerTimePointWithMeta, self).evaluate(predictions, targets)


class Accuracy2v2(Metric):
    """2 versus 2 accuracy (described in our paper).
    This metric assumes the we leave out exactly 2 words with all their repetitions.
    """
    def __init__(self, dist_metric='euclidean', use_median=False, name='2v2', tolerance=1e-16,
                 compare_same_question=False, allow_same_question=True, compare_same_word=False,
                 allow_same_word=True, majority_vote=False, seed=None):
        """
        Applies to the case where the samples we made predictions on belong to exactly 2 groups. Here we assume that
        the two groups correspond to 2 different words, each with multiple repetitions (for different questions).

        Then it picks random pairs of samples, one from each group, and does the 2v2 test: it checks whether the
        predicted output for sample 1 is closer to sample 1's true embedding, than to sample 2's true embedding.
        It returns:
         1.0 if
            dist(predicted_sample_1, embedding(target_label_1)) +
            dist(predicted_sample_2, embedding(target_label_2)) <
            dist(predicted_sample_1, embedding(target_label_2)) +
            dist(predicted_sample_2, embedding(target_label_1))
        0.5 if the two predictions are equal, so it can't decide.
        0.0, otherwise

        Args:
            embeddings (nd.array):          Numpy 2D array, where each row represents
                                            the embedding for the item with the cooresponding row number.
            dist_metric (str):              Distance metric to use when calculating the
                                            ranking. Can be 'euclidean' or 'cosine'.
            embedded_labels (bool):         If True, the targets that are provided in
                                            the evaluate funciton are already embedded. If false, they are
                                            indexes that need to be mapped to the corresponding embeddings
                                            using the embeddings dictionary.
            use_median (bool):              Whether to compute the mean or median rank.
            name (str):                     Name to use for this metric.
            tolerance (Float):              Tolerance when to consider two predictions to be
                                            equal. In this case we match randomly the predictions with the
                                            targets.
            max_pairs (Int):                Maximum number of comparisons to consider. When comparing (w0, qi) and
                                            (w1, qj), it will only consider at most max_pairs combinations (qi, qj).
            compare_same_question (Bool):   Whether to force qi == qj when comparing (w0, qi) and (w1, qj).
            allow_same_question (Bool):     Whether to allow qi == qj when comparing (w0, qi) and (w1, qj).
            seed (int or RandomState):      Seed to use by the random number generator used to select the pairs.
        """
        super(Accuracy2v2, self).__init__(name=name)
        if dist_metric == 'euclidean':
            self.dist = scipy.spatial.distance.euclidean
        elif dist_metric == 'cosine':
            self.dist = scipy.spatial.distance.cosine
        else:
            raise ValueError('Unsupported distance metric.')
        self.use_median = use_median
        self.tol = tolerance
        if compare_same_question:
            assert allow_same_question, \
                "If you compare the same question, you need to allow the same question."
        self.compare_same_question = compare_same_question
        self.allow_same_question = allow_same_question
        self.compare_same_word = compare_same_word
        self.allow_same_word = allow_same_word
        self.majority_vote = majority_vote
        self.random_state = np.random.RandomState(seed)

    def __str__(self):
        return self.name

    def __call__(self, predictions, targets, is_multioutput=False, *args):
        return self.evaluate(predictions, targets, *args)

    def _compare_predictions(self, w0_pred, w0_targ, w1_pred, w1_targ):
        score_correct_matching = self.dist(w0_pred, w0_targ) + self.dist(w1_pred, w1_targ)
        score_wrong_matching = self.dist(w0_pred, w1_targ) + self.dist(w1_pred, w0_targ)
        if score_correct_matching < score_wrong_matching:
            return 1.0
        elif self.dist(w0_pred, w1_pred) < self.tol:
            # If the predictions are equal, we have 50% chance to match it correctly.
            return 0.5
        return 0.0

    def evaluate(self, predictions, targets, *args):
        """
        Evaluate the 2v2 accuracy.
        Args:
            predictions(list of tuples): A list of elements (sample_prediction, sample_group_indexes).
                                            sample_prediction is an array like targets, of shape
                                                              (num_samples, num_output_features).
                                            sample_group_indexes is an int encoded as word_id * 100 + question_id

            targets(array-like):         Array of target results, of shape (num_samples, num_output_features).

        Returns: a float between [0.0, 1.0] representing the 2v2 accuracy score.
        """
        num_samples = len(targets)
        word_ids = [p[1] // 100 for p in predictions]
        question_ids = [p[1] % 100 for p in predictions]

        # Extract the brain data prediction from the prediction tuple.
        predictions = np.asarray([p[0] for p in predictions])
        if len(predictions.shape) > 2:
            predictions = predictions.reshape(predictions.shape[:2])
        if len(targets.shape) > 2:
            targets = targets.reshape(targets.shape[:2])

        # For all combinations of samples, compute their 2v2 acc.
        acc_sum = 0.0
        count = 0
        all_combinations = itertools.combinations(range(num_samples), 2)
        for i, j in all_combinations:
            if self.compare_same_question and question_ids[i] != question_ids[j]:
                continue
            if not self.allow_same_question and question_ids[i] == question_ids[j]:
                continue
            if self.compare_same_word and word_ids[i] != word_ids[j]:
                continue
            if not self.allow_same_word and word_ids[i] == word_ids[j]:
                continue
            score = self._compare_predictions(predictions[i], targets[i], predictions[j], targets[j])
            acc_sum += score
            count += 1

        if count == 0:
            return np.nan
        return acc_sum / count


class Accuracy2v2_SensorTimepoint(Accuracy2v2):
    def __init__(self, dist_metric='euclidean', use_median=False, name='2v2_sensortimepoint',
                 tolerance=1e-16, compare_same_question=False, allow_same_question=True,
                 compare_same_word=False, allow_same_word=True, majority_vote=False, seed=None,
                 sensor_groups=None):
        super(Accuracy2v2_SensorTimepoint, self).__init__(
            dist_metric=dist_metric,
            use_median=use_median,
            name=name,
            tolerance=tolerance,
            compare_same_question=compare_same_question,
            allow_same_question=allow_same_question,
            compare_same_word=compare_same_word,
            allow_same_word=allow_same_word,
            majority_vote=majority_vote,
            seed=seed)
        self.sensor_groups = sensor_groups

    def _compare_predictions(self, w0_pred, w0_targ, w1_pred, w1_targ):
        w0_pred = np.reshape(w0_pred, [306, -1])
        w1_pred = np.reshape(w1_pred, [306, -1])
        w0_targ = np.reshape(w0_targ, [306, -1])
        w1_targ = np.reshape(w1_targ, [306, -1])

        n_groups = self.sensor_groups.shape[0]
        group_accs = []

        for ig in range(n_groups):
            tmp_w0_pred = w0_pred[self.sensor_groups[ig]==1]
            tmp_w1_pred = w1_pred[self.sensor_groups[ig]==1]
            tmp_w0_targ = w0_targ[self.sensor_groups[ig]==1]
            tmp_w1_targ = w1_targ[self.sensor_groups[ig]==1]

            dist_correct = np.sum((tmp_w0_pred-tmp_w0_targ)**2, 0) + \
                           np.sum((tmp_w1_pred-tmp_w1_targ)**2, 0)
            dist_incorrect = np.sum((tmp_w0_pred-tmp_w1_targ)**2, 0) + \
                             np.sum((tmp_w1_pred-tmp_w0_targ)**2, 0)

            group_accs.append((dist_correct < dist_incorrect)*1.0 +
                              (dist_correct == dist_incorrect)*0.5)

        return np.asarray(group_accs)


class Accuracy2v2_4Outputs(Metric):
    """2 vs 2 accuracy"""
    def __init__(self, dist_metric='euclidean', use_median=False,
                 name='2v2_w_groups', tolerance=1e-16, max_pairs=None, compare_same_question=False,
                 allow_same_question=True, compare_same_word=False, allow_same_word=True,
                 majority_vote=False, seed=None):
        """
        Applies to the case where the samples we made predictions on belong to exactly 2 groups. Here we assume that
        the two groups correspond to 2 different words, each with multiple repetitions (for different questions).

        Then it picks random pairs of samples, one from each group, and does the 2v2 test: it checks whether the
        predicted output for sample 1 is closer to sample 1's true embedding, than to sample 2's true embedding.
        It returns:
         1.0 if
            dist(predicted_sample_1, embedding(target_label_1)) +
            dist(predicted_sample_2, embedding(target_label_2)) <
            dist(predicted_sample_1, embedding(target_label_2)) +
            dist(predicted_sample_2, embedding(target_label_1))
        0.5 if the two predictions are equal, so it can't decide.
        0.0, otherwise

        Args:
            dist_metric (str):              Distance metric to use when calculating the
                                            ranking. Can be 'euclidean' or 'cosine'.
            use_median (bool):              Whether to compute the mean or median rank.
            name (str):                     Name to use for this metric.
            tolerance (Float):              Tolerance when to consider two predictions to be
                                            equal. In this case we match randomly the predictions with the
                                            targets.
            max_pairs (Int):                Maximum number of comparisons to consider. When comparing (w0, qi) and
                                            (w1, qj), it will only consider at most max_pairs combinations (qi, qj).
            compare_same_question (Bool):   Whether to force qi == qj when comparing (w0, qi) and (w1, qj).
            allow_same_question (Bool):     Whether to allow qi == qj when comparing (w0, qi) and (w1, qj).
            seed (int or RandomState):      Seed to use by the random number generator used to select the pairs.
        """
        super(Accuracy2v2_4Outputs, self).__init__(name=name)
        if dist_metric == 'euclidean':
            self.dist = scipy.spatial.distance.euclidean
        elif dist_metric == 'cosine':
            self.dist = scipy.spatial.distance.cosine
        else:
            raise ValueError('Unsupported distance metric.')
        self.use_median = use_median
        self.tol = tolerance
        self.max_pairs = max_pairs
        if compare_same_question:
            assert allow_same_question, "If you compare the same question, you need to allow the same question."
        self.compare_same_question = compare_same_question
        self.allow_same_question = allow_same_question
        if compare_same_word:
            assert allow_same_word, "If you compare the same word, you need to allow the same word."
            assert not compare_same_question, "If you compare the same word, it cannot be with the same question."
        self.compare_same_word = compare_same_word
        self.allow_same_word = allow_same_word
        self.majority_vote = majority_vote
        self.random_state = np.random.RandomState(seed)

    def __str__(self):
        return self.name

    def __call__(self, predictions, targets, is_multioutput=False, *args):
        return self.evaluate(predictions, targets, *args)

    def evaluate(self, predictions, targets, *args):
        """
        Evaluate the 2v2 accuracy.
        Args:
            predictions(list of tuples): A list of elements (sample_prediction, sample_group_indexes).
                                            sample_prediction is an array like targets, of shape
                                                              (num_samples, num_output_features).
                                            sample_group_indexes is an int encoded as word_id * 100 + question_id

            targets(array-like):         Array of target results, of shape (num_samples, num_output_features).

        Returns: a float between [0.0, 1.0] representing the 2v2 accuracy score.
        """
        def _compare_predictions(w0_pred_targets, w1_pred_targets):
            w0_pred, w0_targ = w0_pred_targets
            w1_pred, w1_targ = w1_pred_targets
            score_correct_matching = self.dist(w0_pred, w0_targ) + self.dist(w1_pred, w1_targ)
            score_wrong_matching = self.dist(w0_pred, w1_targ) + self.dist(w1_pred, w0_targ)
            if score_correct_matching < score_wrong_matching:
                return 1.0
            elif self.dist(w0_pred, w1_pred) < self.tol:
                # If the predictions are equal, we have 50% chance to match it correctly.
                return 0.5
            return 0.0

        word_ids = [p[1] // 100 for p in predictions]
        question_ids = [p[1] % 100 for p in predictions]
        unique_word_ids = list(set(word_ids))
        unique_q_ids = list(set(question_ids))
        if len(unique_word_ids) != 2:
            # This could happen on the training set, but shouldn't happen on the test set.
            # warnings.warn("The samples need to come from exactly 2 words.")
            return np.nan

        predictions = np.asarray([p[0] for p in predictions])
        if len(predictions.shape) > 2:
            predictions = predictions.reshape(predictions.shape[:2])
        if len(targets.shape) > 2:
            targets = targets.reshape(targets.shape[:2])

        # Split the samples in two groups.
        w0_pred_targets = {q: (p, t)
                           for q, w, p, t in zip(question_ids, word_ids, predictions, targets)
                           if unique_word_ids[0] == w}
        w1_pred_targets = {q: (p, t)
                           for q, w, p, t in zip(question_ids, word_ids, predictions, targets)
                           if unique_word_ids[1] == w}
        assert len(w0_pred_targets) == len(w1_pred_targets), "The two questions need to have the same number of samples."

        if not self.name.endswith('sensortimepoint'):
            print('#' * 10)
            print(' ' * 5, self.name)
        # Now the predictions of each question are ordered by word id.
        scores = []
        if self.compare_same_question:
            # We try to match the two words to their brain activity, for the same question.
            for q in unique_q_ids:
                score = _compare_predictions(w0_pred_targets[q], w1_pred_targets[q])
                scores.append(score)
                if not self.name.endswith('sensortimepoint'):
                    print('Word %d question id %d' % (unique_word_ids[0], q),
                          'Word %d question id %d' % (unique_word_ids[1], q),
                          ' score ', score)
        elif self.compare_same_word:
            # We try to match the same word to their brain activity, for the different questions.
            q0 = unique_q_ids[0]
            q1 = unique_q_ids[1]
            i = 0
            for w_pred_targets in (w0_pred_targets, w1_pred_targets):
                score = _compare_predictions(w_pred_targets[q0], w_pred_targets[q1])
                scores.append(score)
                if not self.name.endswith('sensortimepoint'):
                    print('Word %d question id %d' % (unique_word_ids[i], q0),
                          'Word %d question id %d' % (unique_word_ids[i], q1),
                          ' score ', score)
                i += 1
        else:
            # Pick max_pairs pairs of (w0_sample, w1_sample).
            if self.allow_same_question:
                # We can potentially pick samples that belong to same question.
                # Take all combinations of questions.
                question_pairs_to_compare = [(i, j) for i in unique_q_ids for j in unique_q_ids]
            else:
                # We cannot pick samples that belong to same question.
                question_pairs_to_compare = [(i, j) for i in unique_q_ids for j in unique_q_ids if i!=j]

            for q0, q1 in question_pairs_to_compare:
                score = _compare_predictions(w0_pred_targets[q0], w1_pred_targets[q1])
                scores.append(score)
                if not self.name.endswith('sensortimepoint'):
                    print('Word %d question id %d' % (unique_word_ids[0], q0),
                          'Word %d question id %d' % (unique_word_ids[1], q1),
                          ' score ', score)
        if not self.name.endswith('sensortimepoint'):
            print('#' * 10)
        if len(scores) == 0:
            return np.nan
        return np.mean(scores)


class Accuracy2v2_4Outputs_SensorTimepoint(Metric):
    def __init__(self, dist_metric='euclidean', use_median=False,
                 name='2v2_w_groups_sensortimepoint', tolerance=1e-16, max_pairs=None,
                 compare_same_question=False, allow_same_question=True, majority_vote=False,
                 compare_same_word=False, allow_same_word=True, seed=None, sensor_groups=None):
        """
        Applies to the case where the samples we made predictions on belong to exactly 2 groups. Here we assume that
        the two groups correspond to 2 different words, each with multiple repetitions (for different questions).
        Then it picks random pairs of samples, one from each group, and does the 2v2 test: it checks whether the
        predicted output for a group of sensors for sample 1 is closer to the same sensor group in sample 1's true embedding, than to the
        same sample group in sample 2's true embedding.
        returns a 2d Numpy array with one accuracy per sensor group and timepoint. Each accuracy in the array is:
         1.0 if
            dist(predicted_sample_1, embedding(target_label_1)) +
            dist(predicted_sample_2, embedding(target_label_2)) <
            dist(predicted_sample_1, embedding(target_label_2)) +
            dist(predicted_sample_2, embedding(target_label_1))
        0.5 if the two predictions are equal, so it can't decide.
        0.0, otherwise
        Args:
            embeddings (nd.array):          Numpy 2D array, where each row represents
                                            the embedding for the item with the cooresponding row number.
            dist_metric (str):              Distance metric to use when calculating the
                                            ranking. Can be 'euclidean' or 'cosine'.
            embedded_labels (bool):         If True, the targets that are provided in
                                            the evaluate funciton are already embedded. If false, they are
                                            indexes that need to be mapped to the corresponding embeddings
                                            using the embeddings dictionary.
            use_median (bool):              Whether to compute the mean or median rank.
            name (str):                     Name to use for this metric.
            tolerance (Float):              Tolerance when to consider two predictions to be
                                            equal. In this case we match randomly the predictions with the
                                            targets.
            max_pairs (Int):                Maximum number of comparisons to consider. When comparing (w0, qi) and
                                            (w1, qj), it will only consider at most max_pairs combinations (qi, qj).
            compare_same_question (Bool):   Whether to force qi == qj when comparing (w0, qi) and (w1, qj).
            allow_same_question (Bool):     Whether to allow qi == qj when comparing (w0, qi) and (w1, qj).
            seed (int or RandomState):      Seed to use by the random number generator used to select the pairs.
            sensor_groups (nd.array):       Numpy 2D array, where each row represents the neighborhood of sensors
                                            to use in calculting the 2v2 accuracy for each of the 102 sensor positions.
        """
        super(Accuracy2v2_4Outputs_SensorTimepoint, self).__init__(name=name)
        if dist_metric == 'euclidean':
            self.dist = scipy.spatial.distance.euclidean
        elif dist_metric == 'cosine':
            self.dist = scipy.spatial.distance.cosine
        else:
            raise ValueError('Unsupported distance metric.')
        self.use_median = use_median
        self.tol = tolerance
        self.max_pairs = max_pairs
        if compare_same_question:
            assert allow_same_question, "If you compare the same question, you need to allow the same question."
        self.compare_same_question = compare_same_question
        self.allow_same_question = allow_same_question
        if compare_same_word:
            assert allow_same_word, "If you compare the same word, you need to allow the same word."
            assert not compare_same_question, "If you compare the same word, it cannot be with the same question."
        self.compare_same_word = compare_same_word
        self.allow_same_word = allow_same_word
        self.majority_vote = majority_vote
        self.random_state = np.random.RandomState(seed)
        self.sensor_groups = sensor_groups

    def __str__(self):
        return self.name

    def __call__(self, predictions, targets, is_multioutput=False, *args):
        return self.evaluate(predictions, targets, *args)

    def evaluate(self, predictions, targets, *args):
        """
        Evaluate the 2v2 accuracy.
        Args:
            predictions(list of tuples): A list of elements (sample_prediction, sample_group_indexes).
                                            sample_prediction is an array like targets, of shape
                                                              (num_samples, num_output_features).
                                            sample_group_indexes is an int encoded as word_id * 100 + question_id
            targets(array-like):         Array of target results, of shape (num_samples, num_output_features).
        Returns: an 2d array of dimensions number of sensor groups by number of timepoints. Values are floats between [0.0, 1.0]
        representing the 2v2 accuracy score per sensor-timepoint.
        """
        def _compare_predictions(w0_pred_targets, w1_pred_targets):
            w0_pred = np.reshape(w0_pred_targets[0], [306, -1])
            w1_pred = np.reshape(w1_pred_targets[0], [306, -1])
            w0_targ = np.reshape(w0_pred_targets[1], [306, -1])
            w1_targ = np.reshape(w1_pred_targets[1], [306, -1])


            n_groups = self.sensor_groups.shape[0]
            group_accs = []

            for ig in range(n_groups):
                tmp_w0_pred = w0_pred[self.sensor_groups[ig]==1]
                tmp_w1_pred = w1_pred[self.sensor_groups[ig]==1]
                tmp_w0_targ = w0_targ[self.sensor_groups[ig]==1]
                tmp_w1_targ = w1_targ[self.sensor_groups[ig]==1]

                dist_correct = np.sum((tmp_w0_pred-tmp_w0_targ)**2, 0)+np.sum((tmp_w1_pred-tmp_w1_targ)**2, 0)
                dist_incorrect = np.sum((tmp_w0_pred-tmp_w1_targ)**2, 0)+np.sum((tmp_w1_pred-tmp_w0_targ)**2, 0)

                group_accs.append((dist_correct < dist_incorrect)*1.0 + (dist_correct == dist_incorrect)*0.5)

            return np.asarray(group_accs)

        word_ids = [p[1] // 100 for p in predictions]
        question_ids = [p[1] % 100 for p in predictions]
        unique_word_ids = list(set(word_ids))
        unique_q_ids = list(set(question_ids))
        if len(unique_word_ids) != 2:
            # This could happen on the training set, but shouldn't happen on the test set.
            # warnings.warn("The samples need to come from exactly 2 words.")
            return np.nan

        predictions = np.asarray([p[0] for p in predictions])
        if len(predictions.shape) > 2:
            predictions = predictions.reshape(predictions.shape[:2])
        if len(targets.shape) > 2:
            targets = targets.reshape(targets.shape[:2])

        # Split the samples in two groups.
        w0_pred_targets = {q: (p, t)
                           for q, w, p, t in zip(question_ids, word_ids, predictions, targets)
                           if unique_word_ids[0] == w}
        w1_pred_targets = {q: (p, t)
                           for q, w, p, t in zip(question_ids, word_ids, predictions, targets)
                           if unique_word_ids[1] == w}
        assert len(w0_pred_targets) == len(w1_pred_targets), "The two questions need to have the same number of samples."

        if not self.name.endswith('sensortimepoint'):
            print('#' * 10)
            print(' ' * 5, self.name)
        # Now the predictions of each question are ordered by word id.
        scores = []
        if self.compare_same_question:
            # We try to match the two words to their brain activity, for the same question.
            for q in unique_q_ids:
                score = _compare_predictions(w0_pred_targets[q], w1_pred_targets[q])
                scores.append(score)
                if not self.name.endswith('sensortimepoint'):
                    print('Word %d question id %d' % (unique_word_ids[0], q),
                          'Word %d question id %d' % (unique_word_ids[1], q),
                          ' score ', score)
        elif self.compare_same_word:
            # We try to match the same word to their brain activity, for the different questions.
            q0 = unique_q_ids[0]
            q1 = unique_q_ids[1]
            i = 0
            for w_pred_targets in (w0_pred_targets, w1_pred_targets):
                score = _compare_predictions(w_pred_targets[q0], w_pred_targets[q1])
                scores.append(score)
                if not self.name.endswith('sensortimepoint'):
                    print('Word %d question id %d' % (unique_word_ids[i], q0),
                          'Word %d question id %d' % (unique_word_ids[i], q1),
                          ' score ', score)
                i += 1
        else:
            # Pick max_pairs pairs of (w0_sample, w1_sample).
            if self.allow_same_question:
                # We can potentially pick samples that belong to same question.
                # Take all combinations of questions.
                question_pairs_to_compare = [(i, j) for i in unique_q_ids for j in unique_q_ids]
            else:
                # We cannot pick samples that belong to same question.
                question_pairs_to_compare = [(i, j) for i in unique_q_ids for j in unique_q_ids if i!=j]

            for q0, q1 in question_pairs_to_compare:
                score = _compare_predictions(w0_pred_targets[q0], w1_pred_targets[q1])
                scores.append(score)
                if not self.name.endswith('sensortimepoint'):
                    print('Word %d question id %d' % (unique_word_ids[0], q0),
                          'Word %d question id %d' % (unique_word_ids[1], q1),
                          ' score ', score)
        if not self.name.endswith('sensortimepoint'):
            print('#' * 10)
        if len(scores) == 0:
            return np.nan
        return np.mean(scores, 0)

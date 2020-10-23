from __future__ import absolute_import

import itertools
import logging
import numpy as np
import random

from sklearn.model_selection import LeavePOut, LeavePGroupsOut

from ..util.statistics import sample_reservoir
from ..util.container_ops import slice_data, get_dim_size

__author__ = 'Otilia Stretcu'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def batch_iterator(inputs, targets=None, batch_size=None, shuffle=False,
                   allow_smaller_batch=False):
    if not isinstance(inputs, (np.ndarray, list)):
        raise TypeError('Unsupported data type %s encountered.' % type(inputs))
    if not isinstance(targets, (np.ndarray, list)):
        raise TypeError('Unsupported data type %s encountered.' % type(targets))
    num_samples = get_dim_size(inputs, 0)
    if batch_size is None:
        batch_size = num_samples
    if batch_size > num_samples:
        allow_smaller_batch = True
    while True:
        indexes = np.arange(0, num_samples)
        if shuffle:
            np.random.shuffle(indexes)
        shuffled_inputs = slice_data(inputs, indexes)
        shuffled_targets = slice_data(targets, indexes)
        for start_index in range(0, num_samples, batch_size):
            if allow_smaller_batch:
                end_index = min(start_index + batch_size, num_samples)
            else:
                end_index = start_index + batch_size
                if end_index > num_samples:
                    break
            batch_inputs = slice_data(shuffled_inputs,
                                      range(start_index, end_index))
            batch_targets = slice_data(shuffled_targets,
                                       range(start_index, end_index))
            yield batch_inputs, batch_targets


def leave_p_out_indices(p, inputs, targets=None, max_num_folds=None, shuffle=True,
                        random_state=None, print_indices=True, **kwargs):
    """
    Leave p out cross validation.
    Args:
        p: An integer representing the number of samples to leave out for test.
        inputs: A numpy ndarray containing the input data, of shape (samples, input_features).
        targets: A numpy ndarray containing the input data, of shape (samples, output_features).
        max_num_folds: An integer (or None) representing the max number of folds to return. It
            will not take all combinations of num_samples choose p, but at most max_num_folds
            such combinations.
        shuffle: A boolean. If True, we shuffle the samples.
        random_state: An integere representing the seed to use for the random number generator.

    Returns:
        An iterator over (train_data, test_data) batches.
    """
    num_samples = len(inputs)
    assert p < num_samples, "Not enough samples to leave %d out" % p

    # Set the seed for the order shuffling in each batch.
    if shuffle and random_state is not None:
        random.seed(random_state)

    cv = LeavePOut(p)
    batches = cv.split(inputs, y=targets)
    if max_num_folds is not None:
        batches = sample_reservoir(batches, max_num_folds, random_state=random_state)
        if shuffle:
            random.shuffle(batches)

    for train_indices, test_indices in batches:
        if print_indices:
            logger.info('Train indices: %s', str(train_indices))
            logger.info('Train indices: %s', str(test_indices))
            if 'groups' in kwargs:
                word_ids = kwargs['groups'] // 100
                q_ids = kwargs['groups'] % 100
                logger.info('Train word ids: %s', str(word_ids[train_indices]))
                logger.info('Test word ids: %s', str(word_ids[test_indices]))
                logger.info('Train question ids: %s', str(q_ids[train_indices]))
                logger.info('Test question ids: %s', str(q_ids[test_indices]))
        yield train_indices, test_indices


def leave_p_groups_out_indices(p, inputs, groups, targets=None, max_num_folds=None, shuffle=True,
                               random_state=None, print_indices=False, group_id_fn=None):
    """Leave p groups out cross validation.

    It leaves out all elements from p groups. Same as `leave_p_groups_out` but it returns the
    split indices, instead of the split data.

    Note that if max_num_folds is None, then it will consider all possible combinations of p groups.
    But if max_num_folds is provided, it will sample with replacement max_num_folds
    combinations of p groups, so it is possible to have repetitions of the same
    p grpups (although unlikely if max_num_folds << total number of combinations).

    Args:
        p: An integer representing the number of sample groups to leave out for test.
        inputs: A numpy ndarray containing the input data, of shape (num_samples, input_features).
        groups: A list of length (num_samples,) containing the group assignments for each sample.
            The returned split will have p samples that are all in the same group.
        targets: A numpy ndarray containing the input data, of shape (num_samples, output_features).
        max_num_folds: An int (or None) representing the max number of folds to return. It
            will not take all combinations of num groups choose p, but at most max_num_folds such
            combinations.
        shuffle: A boolean. If True, we shuffle the samples.
        random_state: An integer representing the seed to use for the random number generator.

    Returns:
        An iterator over (train_indices, test_indices) batches.
    """
    def _random_combinations(batches):
        batches = sample_reservoir(batches, max_num_folds, random_state=random_state)
        if shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch[0], batch[1]

    num_samples = len(inputs)
    assert p < num_samples, "Not enough samples to leave %d out" % p

    # Set the seed for the order shuffling in each batch.
    if shuffle and random_state is not None:
        random.seed(random_state)

    # Apply some transformation to the groups, if necessary.
    groups_original = groups
    if group_id_fn:
        groups = group_id_fn(groups)

    batches = LeavePGroupsOut(p).split(inputs, y=targets, groups=groups)
    if max_num_folds is not None:
        batches = _random_combinations(batches)

    for train_indices, test_indices in batches:
        if print_indices:
            logger.info('Train indices: %s' % str(train_indices))
            logger.info('Train indices: %s' % str(test_indices))
            word_ids = groups_original // 100
            q_ids = groups_original % 100
            logger.info('Train word ids: %s' % str(word_ids[train_indices]))
            logger.info('Test word ids: %s' % str(word_ids[test_indices]))
            logger.info('Train question ids: %s' % str(q_ids[train_indices]))
            logger.info('Test question ids: %s' % str(q_ids[test_indices]))
        yield train_indices, test_indices


def leave_out_2_words_all_questions_indices(groups, shuffle=True,  random_state=None,
                                            print_indices=True, max_num_folds=np.inf, **kwargs):

    word_ids = [g_id // 100 for g_id in groups]
    question_ids = [g_id % 100 for g_id in groups]

    num_samples = len(groups)
    assert len(word_ids) == num_samples, "We don't have word ids for all samples."
    assert len(question_ids) == num_samples, "We don't have question ids for all samples."

    # Set the seed for the order shuffling in each batch.
    if shuffle and random_state is not None:
        random.seed(random_state)

    # Obtain all combinations of words and questions, from which we select 2 out for test.
    unique_word_ids = list(set(word_ids))
    unique_question_ids = list(set(question_ids))
    num_words = len(unique_word_ids)
    num_questions = len(unique_question_ids)

    word_combinations = [(unique_word_ids[w1], unique_word_ids[w2])
                         for w1 in range(num_words-1) for w2 in range(w1+1, num_words)]
    # Here we don't allow q1==q2 because that case is taken care of in the 2v2 metric.
    question_combinations = [(unique_question_ids[q1], unique_question_ids[q2])
                             for q1 in range(num_questions) for q2 in range(q1+1, num_questions)]

    # Get all combinations of (w1, w2, q1, q2).
    combos = list(itertools.product(word_combinations, question_combinations))
    random.shuffle(combos)
    combos = combos[:max_num_folds]

    g_id_to_pos = {g_id: pos for pos, g_id in enumerate(groups)}

    for combo in combos:
        w1, w2 = combo[0]
        q1, q2 = combo[1]
        g1 = w1 * 100 + q1
        g2 = w2 * 100 + q2

        # Find the location of (w1, q1) and (w2, q2) which will be used as the two test samples.
        test_indices = [g_id_to_pos[g1], g_id_to_pos[g2]]

        g1 = w1 * 100 + q2
        g2 = w2 * 100 + q1
        if g1 in g_id_to_pos and g2 in g_id_to_pos:
            test_indices.extend([g_id_to_pos[g1], g_id_to_pos[g2]])

        # Select for training all samples that do not contain w1 nor w2 nor qq1 nor q2.
        train_indices = [i for i, g_id in enumerate(groups)
                         if g_id // 100 not in (w1, w2) and g_id % 100 not in (q1, q2)]

        if print_indices:
            logger.info('Train indices: %s', str(train_indices))
            logger.info('Train indices: %s', str(test_indices))
        yield train_indices, test_indices

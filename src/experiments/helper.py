import abc
import copy
import functools
import numpy as np
import os
import pickle
import tensorflow as tf

from ..data.load_20questions import get_augmented_feature_weights
from ..methods.iterators import (leave_p_groups_out_indices, leave_p_out_indices,
    leave_out_2_words_all_questions_indices)
from ..methods.models import regression
from ..methods.models.ridge_with_learned_attention import RidgeWithLearnedAttention
from ..methods.metrics import (Accuracy2v2, Accuracy2v2_SensorTimepoint, Accuracy2v2_4Outputs,
    Accuracy2v2_4Outputs_SensorTimepoint, L2DistanceWithMeta, DistPerTimePointWithMeta)
from ..util.plotting import plot_heatmap, plot_time_signal, plot_time_signal_all_channels

__author__ = 'Otilia Stretcu'


def fill_data_parameters(params):
    """Helper function that fills in the missing parameters with default values, or with values
    that can be inferred from the variable `params`.

    Args:
        params(dict): A Parameters object in the format found in the config files.

    Returns:
         A dictionary of parameters that extends `params` with more parameters.
    """
    # These data paths are created according to our data storage format, and should be adjusted
    # accordingly.
    params_data = params.data
    params_data['subj_data_fname_data_array'] = os.path.join(
        params_data['data_dir'],
        params_data['subject_id'],
        params_data['subject_id'] + '_' + params_data['proc_slug'] + '_data_array.npz')

    params_data['subj_data_fname_response_array'] = os.path.join(
        params_data['data_dir'],
        params_data['subject_id'],
        params_data['subject_id'] + '_' + params_data['proc_slug'] + '_response_array.npz')

    params_data['subj_data_fname_response_time_array'] = os.path.join(
        params_data['data_dir'],
        params_data['subject_id'],
        params_data['subject_id'] + '_' +
        params_data['proc_slug'] + '_response_time_array.npz')

    params_data['subj_data_fname_question_array'] = os.path.join(
        params_data['data_dir'],
        params_data['subject_id'],
        params_data['subject_id'] + '_' +
        params_data['proc_slug'] + '_question_array.npz')

    params_data['subj_data_fname_all'] = os.path.join(
        params_data['data_dir'],
        params_data['subject_id'],
        params_data['subject_id'] + '_' +
        params_data['proc_slug'] + '.npz')

    params['latest_data_path'] = os.path.join(params['output']['path_outputs'], 'latest_data.pkl')

    return params


def fill_learner_params(params, inputs, targets, num_question_features=None,
                        num_word_features=None):
    """Helper function that fills in the missing learning parameters with default values, or with
    values that can be inferred from the variable `params`.

    Args:
        params: A Parameters object.
        inputs: A numpy array of shape (num_samples, num_input_features) containing the model
            inputs.
        targets: A numpy array of shape (num_samples, num_output_features) containing the model
            targets.
        num_question_features: Integer representing the number of features in the question
            representation.
        num_word_features: Integer representing the number of features in the word representation.
    """
    params_learn = params.learning
    if params_learn['model'] == RidgeWithLearnedAttention:
        params_learn['model_params']['num_outputs'] = [targets.shape[-1]]
        params_learn['model_params']['num_word_features'] = [num_word_features]
        params_learn['model_params']['num_question_features'] = [num_question_features]


def create_inputs_outputs(word_trials, params_data, group_by_word=False, group_by_question=False):
    """
    Creates the numpy arrays with the inputs and outputs for a machine learning model. It also
    assigns a group id to each sample, which can be used to split the samples into train/test
    based on groups.
    
    Args:
        word_trials(list): List of WordTrial objects.
        params_data(dict): A Parameters object containing parameters for data loading. It is used
            to decide what goes into the model input, and what goes in the model output.
        group_by_word(bool): Flag that indicates whether the word id should be part of the group id
            of the sample.
        group_by_question(bool): Flag that indicates whether the question id should be part of the
            group id of the sample.

    Returns: A tuple (inputs, outputs, groups) where
        inputs  is a numpy array of shape (num_samples, num_input_features)
        outputs is a numpy array of shape (num_samples, num_output_features)
        groups  is a list of shape (num_samples,) containing a group id for each sample. The group id is:
                    word id,                        if group_by_word == True and group_by_question == False
                    question id,                    if group_by_word == False and group_by_question == True
                    word id * 100 + question id,    if group_by_word == True and group_by_question == True
                    0,                              if group_by_word == False and group_by_question == False
        
    """
    use_augmented_embed = params_data.word_embedding_type in \
                          ['augmented_MTurk', 'augmented_MTurk_no_experiment_questions']
    if use_augmented_embed:
        # Augment the current word features with question similarity weights.
        exclude_exp_qs = params_data.word_embedding_type == 'augmented_MTurk_no_experiment_questions'
        with open(params_data['semantic_feature_path'], 'rb') as fin:
            semantic_features = np.load(fin)
            augmented_feature_weights = get_augmented_feature_weights(
                semantic_features, exclude_exp_qs=exclude_exp_qs, dist='cosine')
    
    inputs = []
    outputs = []
    groups = []
    for trial in word_trials:
        # Create the inputs.
        sample_input = []
        if params_data.input_question_features:
            sample_input.append(trial.question_trial.question.features)
        if params_data.input_question_brain:
            sample_input.append(trial.question_trial.brain.flatten())
        if params_data.input_word_features:
            if use_augmented_embed:
                trial_feature_weights = augmented_feature_weights[trial.question_trial.question.text]
                reweighted_features = np.multiply(trial_feature_weights, trial.word.features)
                sample_input.append(reweighted_features)
            else:
                sample_input.append(trial.word.features)
        if params_data.input_word_brain:
            sample_input.append(trial.brain.flatten())
        # Flatten all inputs for this sample to get a 1D sample.
        sample_input = np.concatenate(sample_input)
        inputs.append(sample_input)

        # Create the outputs.
        sample_output = []
        if params_data.output_question_features:
            sample_output.append(trial.question_trial.question.features)
        if params_data.output_question_brain:
            sample_output.append(trial.question_trial.brain.flatten())
        if params_data.output_word_features:
            sample_output.append(trial.word.features)
        if params_data.output_word_brain:
            sample_output.append(trial.brain.flatten())
        # Flatten all outputs for this sample to get a 1D sample.
        sample_output = np.concatenate(sample_output)
        outputs.append(sample_output)
        
        # Assign a group index to this sample.
        group = 0
        if group_by_word:
            group = trial.word.id * 100
        if group_by_question:
            group = group + trial.question_trial.question.id
        groups.append(group)

    # Return the inputs and outputs as numpy arrays.
    return np.asarray(inputs), np.asarray(outputs), groups


def create_iterators(params_learn, groups):
    """Creates the cross-validation data iterators.

    This takes into account if we are considering zero-shot setting for the words, questions or
    both, as specified by the experimental configuration in `params_learn`.

    Args:
        params_learn: A Parameters object containing the experimental setting.
        groups: A list of length num_samples, which specifies a group index for each sample. These
            can be used to extract the word id and question id.

    """
    word_ids = [w_q // 100 for w_q in groups]
    question_ids = [w_q % 100 for w_q in groups]
    num_words = len(set(word_ids))
    num_questions = len(set(question_ids))
    max_test_folds = params_learn.max_folds_test
    max_val_folds = params_learn.max_folds_param_valid
    if params_learn.zero_shot_words and params_learn.zero_shot_questions:
        cross_val = functools.partial(leave_out_2_words_all_questions_indices,
            num_words=num_words, num_questions=num_questions, max_num_folds=max_test_folds,
            shuffle=True, random_state=params_learn.seed, print_indices=False)
        cross_val_param_valid = functools.partial(leave_out_2_words_all_questions_indices,
            num_words=num_words, num_questions=num_questions, max_num_folds=max_val_folds,
            shuffle=True, random_state=params_learn.seed, print_indices=False)
    elif params_learn.zero_shot_words and not params_learn.zero_shot_questions:
        # To leave out 2 words with all repetitions, we use leave_p_groups_out_indices, and set
        # the group id to be the word it.
        group_id_fn = lambda group_id: group_id // 100
        cross_val = functools.partial(leave_p_groups_out_indices,
            p=2,  group_id_fn=group_id_fn, max_num_folds=max_test_folds, shuffle=True,
            random_state=params_learn.seed, print_indices=params_learn.seed)
        cross_val_param_valid = functools.partial(leave_p_groups_out_indices,
            p=2,  group_id_fn=group_id_fn, max_num_folds=max_test_folds, shuffle=True,
            random_state=params_learn.seed, print_indices=False)
    elif not params_learn.zero_shot_words and params_learn.zero_shot_questions:
        # To leave out 2 words with all repetitions, we use leave_p_groups_out_indices, and set
        # the group id to be the question it.
        group_id_fn = lambda group_id: group_id % 100
        cross_val = functools.partial(leave_p_groups_out_indices,
            p=2, group_id_fn=group_id_fn, max_num_folds=max_test_folds, shuffle=True,
            random_state=params_learn.seed, print_indices=params_learn.seed)
        cross_val_param_valid = functools.partial(leave_p_groups_out_indices,
            p=2, group_id_fn=group_id_fn, max_num_folds=max_test_folds, shuffle=True,
            random_state=params_learn.seed, print_indices=False)
    else:
        # Neither questions nor words are zero shot. We simply leave out 2 random examples.
        cross_val = functools.partial(leave_p_out_indices,
            p=2, max_num_folds=max_test_folds, shuffle=True,
            random_state=params_learn.seed, print_indices=params_learn.seed)
        cross_val_param_valid = functools.partial(leave_p_out_indices,
            p=2, max_num_folds=max_test_folds, shuffle=True,
            random_state=params_learn.seed, print_indices=False)
    return cross_val, cross_val_param_valid


def create_test_metrics(params):
    params_learn = params['learning']
    metrics = [L2DistanceWithMeta()]
    if params_learn.zero_shot_words and params_learn.zero_shot_questions:
        metrics.extend([
            Accuracy2v2_4Outputs(
                name='2v2_q1!=q2',
                dist_metric=params_learn['dist_metric'],
                max_pairs=params_learn['max_pairs'],
                compare_same_question=False,
                allow_same_question=False,
                majority_vote=params_learn['majority_vote'],
                seed=params_learn['seed']),
            Accuracy2v2_4Outputs(
                name='2v2_q1==q2',
                dist_metric=params_learn['dist_metric'],
                max_pairs=params_learn['max_pairs'],
                compare_same_question=True,
                allow_same_question=True,
                majority_vote=params_learn['majority_vote'],
                seed=params_learn['seed']),
            Accuracy2v2_4Outputs(
                name='2v2_any_q1_q2',
                dist_metric=params_learn['dist_metric'],
                max_pairs=params_learn['max_pairs'],
                compare_same_question=False,
                allow_same_question=True,
                majority_vote=params_learn['majority_vote'],
                seed=params_learn['seed']),
            Accuracy2v2_4Outputs(
                name='2v2_w1==w2',
                dist_metric=params_learn['dist_metric'],
                max_pairs=params_learn['max_pairs'],
                compare_same_word=True,
                allow_same_word=True,
                majority_vote=params_learn['majority_vote'],
                seed=params_learn['seed']),
            Accuracy2v2_4Outputs_SensorTimepoint(
                name='2v2_any_q1_q2_sensortimepoint',
                dist_metric=params_learn['dist_metric'],
                max_pairs=params_learn['max_pairs'],
                compare_same_question=False,
                allow_same_question=True,
                majority_vote=params_learn['majority_vote'],
                seed=params_learn['seed'],
                sensor_groups=params_learn['sensor_groups']),
            Accuracy2v2_4Outputs_SensorTimepoint(
                name='2v2_w1==w2_sensortimepoint',
                dist_metric=params_learn['dist_metric'],
                max_pairs=params_learn['max_pairs'],
                compare_same_word=True,
                allow_same_word=True,
                majority_vote=params_learn['majority_vote'],
                seed=params_learn['seed'],
                sensor_groups=params_learn['sensor_groups']),
        ])
    else:
        metrics.extend([
            Accuracy2v2(
                name='2v2_q1!=q2',
                dist_metric=params_learn['dist_metric'],
                compare_same_question=False,
                allow_same_question=False,
                majority_vote=params_learn['majority_vote'],
                seed=params_learn['seed']),
            Accuracy2v2(
                name='2v2_q1==q2',
                dist_metric=params_learn['dist_metric'],
                compare_same_question=True,
                allow_same_question=True,
                majority_vote=params_learn['majority_vote'],
                seed=params_learn['seed']),
            Accuracy2v2(
                name='2v2_any_q1_q2',
                dist_metric=params_learn['dist_metric'],
                compare_same_question=False,
                allow_same_question=True,
                majority_vote=params_learn['majority_vote'],
                seed=params_learn['seed']),
            Accuracy2v2(
                name='2v2_w1==w2',
                dist_metric=params_learn['dist_metric'],
                compare_same_word=True,
                allow_same_word=True,
                majority_vote=params_learn['majority_vote'],
                seed=params_learn['seed']),
            Accuracy2v2_SensorTimepoint(
                name='2v2_any_q1_q2_sensortimepoint',
                dist_metric=params_learn['dist_metric'],
                compare_same_question=False,
                allow_same_question=True,
                majority_vote=params_learn['majority_vote'],
                seed=params_learn['seed'],
                sensor_groups=params_learn['sensor_groups'])
        ])

    if params.output.plot_prediction_error_per_time:
        merics_dist_per_time = params.output.dist_metric_plot_per_time_pt
        if not isinstance(merics_dist_per_time, (list, tuple)):
            merics_dist_per_time = [merics_dist_per_time]
        for m in merics_dist_per_time:
            if params.data.space_downsample:
                num_sensors = 162
            else:
                num_sensors = 306
            metrics.append(DistPerTimePointWithMeta(
                name='dist_per_time_%s' % m,
                dist_metric=m,
                num_sensors=num_sensors))
    return metrics


def create_val_metric(params, use_2v2=False):
    num_sensors = 162 if params.data.space_downsample else 306
    if use_2v2:
        if params.learning.differentRegPerOutput:
            if params.learning.zero_shot_words and params.learning.zero_shot_questions:
                acc_class = Accuracy2v2_4Outputs_SensorTimepoint
            else:
                acc_class = Accuracy2v2_SensorTimepoint
            val_metric = acc_class(
                name='2v2_any_q1_q2_sensortimepoint',
                dist_metric=params.learning['dist_metric'],
                compare_same_question=False,
                allow_same_question=True,
                majority_vote=params.learning['majority_vote'],
                seed=params.learning['seed'],
                sensor_groups=np.eye(num_sensors))
        else:
            if params.learning.zero_shot_words and params.learning.zero_shot_questions:
                acc_class = Accuracy2v2_4Outputs
            else:
                acc_class = Accuracy2v2_SensorTimepoint
            val_metric = acc_class(
                name='2v2_any_q1_q2',
                dist_metric=params.learning['dist_metric'],
                compare_same_question=False,
                allow_same_question=True,
                majority_vote=params.learning['majority_vote'],
                seed=params.learning['seed'])
        val_metric_higher_better = True
    else:
        if params.learning.differentRegPerOutput:
            val_metric = DistPerTimePointWithMeta(
                name='dist_per_time_euclid',
                dist_metric='euclidean',
                num_sensors=num_sensors,
                average_folds=True)  # Set this to True only if the cross val iterator + val metric
                                     # produces the same number of results per fold.
        else:
            val_metric = L2DistanceWithMeta(avg_dist_per_feature=False)
        val_metric_higher_better = False
    return val_metric, val_metric_higher_better


def create_results_string(params):
    """Creates a string used to describe the experiment."""
    name = params.data.subject_id
    name += '-' + get_hypothesis(params)
    name += '-' + params['data']['word_embedding_type']

    # Data preprocessing.
    if params['data']['space_downsample']:
        name += '-spaceDown'
    if params['data']['normalize_inputs']:
        name += '-normInp'
    if params['data']['normalize_outputs']:
        name += '-normOut'
    if params['data']['brain_scaling']:
        name += '-scale_' + str(params['data']['brain_scaling'])
    if params['data']['avg_time_window_length']:
        name += '-avgTime_' + str(params['data']['avg_time_window_length'])
    if params['data']['num_unique_train_words_to_keep']:
        name += '-keep%dWords-seedRemove_%d' % \
                (params['data']['num_unique_train_words_to_keep'], params['data']['seed_remove_words'])

    # Zero-shotness.
    if params.learning.zero_shot_words:
        name += '-zeroShotWords'
    if params.learning.zero_shot_questions:
        name += '-zeroShotQuest'

    name += '-' + params.output.output_dir_suffix

    return name

# Callbacks - functions that can be applied within the Runner to the model of every fold.
def callback_save_fold_results(model, train_inputs, train_outputs, train_groups, test_inputs,
                               test_outputs, test_groups, fold, train_results, test_results,
                               train_predictions=None, test_predictions=None, *other_args):
    return train_results, test_results


def callback_save_fold_predictions(model, train_inputs, train_outputs, train_groups,  test_inputs,
                               test_outputs, test_groups, fold, train_results, test_results,
                               train_predictions, test_predictions, *other_args):
    return test_predictions, test_outputs


def callback_save_attention(model, train_data, test_data, fold, train_results, test_results, *other_args):
    return model.get_attention()


def callback_save_weights(model, train_data, test_data, fold, train_results, test_results, *other_args):
    return model.parameters()


# Plotting.
def plot_prediction_per_time(params, callback_results, callback_idx_dist_per_time, suffix,
                             outputs_folder):
    callback_results = [callback_fold[callback_idx_dist_per_time]
                        for callback_fold in callback_results]
    merics_dist_per_time = params.output.dist_metric_plot_per_time_pt
    if not isinstance(merics_dist_per_time, (list, tuple)):
        merics_dist_per_time = [merics_dist_per_time]

    for m in merics_dist_per_time:
        metric_name = 'dist_per_time_%s' % m
        # Extract the test results for this metric for each fold.
        test_error = [results_fold[1][metric_name] for results_fold in callback_results]
        # Since each fold had multiple test samples, concatenate them all before averaging.
        test_error = np.concatenate(test_error, axis=0)
        # Average te error over all test samples from all folds.
        test_error = np.mean(test_error, axis=0)

        path = os.path.join(outputs_folder, 'prediction_error_%s_per_time%s.txt' % (m, suffix))
        np.savetxt(path, test_error, delimiter=',')
        print('Saved distance per time point at: ', path)

        # Plot.
        save_a_plot_per_channel = False
        if save_a_plot_per_channel:
            num_channels = test_error.shape[0]
            for channel in range(num_channels):
                filename = "test_%s_loss_per_time%s-chan_%d.png" % (m, suffix, channel)
                plot_time_signal(
                    test_error[channel],
                    output_path=outputs_folder,
                    output_filename=filename,
                    title='Error per time point')
                print('Saved figure at: ', filename)
        filename = "test_%s_loss_per_time%s.png" % (m, suffix)
        plot_time_signal_all_channels(
            test_error,
            output_path=outputs_folder,
            output_filename=filename,
            title='Test %s error per time %s' % (m, suffix))
        # Save a plot that averages the channels.
        filename = "test_%s_loss_per_time%s-avg_chan.png" % (m, suffix)
        plot_time_signal(
            np.mean(test_error, axis=0),
            output_path=outputs_folder,
            output_filename=filename,
            title='Error per time point averaged across sensors',
            figsize=(15, 8))
        print('Saved figure at: ', filename)


def load_best_params(path):
    """Loads the best model parameters from file in case we have done the parameter validation
    previously.

    Args:
        path: Path to a text file containing the best model parameters per fold.

    Returns:
        An array of the same shape as the number of outputs, containing a different regularization
    alpha per output.
    """
    fold_results = []
    with open(path, 'r') as file:
        lines = file.readlines()
        fold_result = [[float(i) for i in line.strip('\n,').split(',')] for line in lines]
        fold_results.append(fold_result)
    fold_results = np.concatenate(fold_results)
    fold_results = np.median(fold_results, axis=0)
    return fold_results


def get_reg_params_name(params_learn):
    """Returns the name of the regularization parameters that we can tune."""
    if params_learn['model'] == RidgeWithLearnedAttention:
        return 'reg_weights'
    elif params_learn['model'] in [regression.RidgeRegression, regression.LassoRegression]:
        return 'alpha'
    else:
        raise ValueError('Need to add here the regularization parameters of model: %s' %
                         str(params_learn['model']))


def get_hypothesis(params):
    if params.learning.model == RidgeWithLearnedAttention:
        return 'H4_2-questEmb_' + params.data.question_embedding_type
    elif not params.data.input_question_features and params.data.input_word_features:
        return 'H1'
    elif params.data.input_question_features and not params.data.input_word_features:
        return 'H2-questEmb_' + params.data.question_embedding_type
    elif params.data.input_question_features and params.data.input_word_features:
        if params.data.word_embedding_type in ['augmented_MTurk',
                                               'augmented_MTurk_no_experiment_questions']:
            return 'H4_1-questEmb_' + params.data.question_embedding_type
        return 'H3-questEmb_' + params.data.question_embedding_type
    raise ValueError('Should not get here.')


def get_serializable_params(params):
    def _serialize(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = _serialize(v)
        elif isinstance(obj, np.ndarray):
            obj = str(obj)
        elif type(obj) == abc.ABCMeta:
            obj = str(obj)
        elif obj in [tf.train.AdamOptimizer, tf.train.MomentumOptimizer]:
            return str(obj)
        elif isinstance(obj, list):
            obj = [_serialize(elem) for elem in obj]
        return obj
    params = copy.deepcopy(params)
    return _serialize(params)


def save_results_and_plot(params, callback_results, outputs_folder, callback_idx_dist_per_time,
                          callback_idx_save_attention, callback_idx_predictions,
                          callback_idx_save_weights):
    """Saves the results stored in the callbacks and make plots.

    Args:
        params: A Parameters object containig the parameters of the experiment.
        callback_results: A list of callback results.
        outputs_folder: A string representing the path where to save the results.
        callback_idx_dist_per_time: A integer representing the position of the results regarding
            distance per timepoint in the callback_results list, or None.
        callback_idx_save_attention: A integer representing the position of the attention weights
            in the callback_results list, or None.
        callback_idx_predictions: A integer representing the position of the predictions in the
            callback_results list, or None.
        callback_idx_save_weights: A integer representing the position of the learned model weights
            in the callback_results list, or None.
    """
    suffix = '-' + params.data.subject_id + '-' + get_hypothesis(params)

    # Save results per sensor-timepoint to file.
    sensortimepoint = [results_fold[0][1]['2v2_any_q1_q2_sensortimepoint']
                       for results_fold in callback_results]
    path = os.path.join(outputs_folder, 'sensortimepoint%s.npy' % suffix)
    np.save(path, [np.mean(sensortimepoint, axis=0), np.std(sensortimepoint, axis=0)])
    print('Sensor-timepoint accuracies save at: ', path)

    if params.output.plot_prediction_error_per_time:
        plot_prediction_per_time(params, callback_results, callback_idx_dist_per_time, suffix,
                                 outputs_folder)

    # Save learned attention params.
    if params.output.save_attention and params.learning.model in [RidgeWithLearnedAttention]:
        qs_folds = [callbacks_fold[callback_idx_save_attention] for callbacks_fold in callback_results]
        attention = np.mean(np.stack(qs_folds), axis=0)
        path = os.path.join(outputs_folder, 'attention%s.txt' % suffix)
        np.savetxt(path, attention, newline='\n')
        print('Saved attention matrix at: ', path)
        plot_heatmap(attention,
                     output_path=params.output.path_outputs,
                     output_filename='attention.png',
                     row_names="Questions",
                     col_names="Word Features")

    # Save predictions.
    callback_predictions = [callback_fold[callback_idx_predictions]
                            for callback_fold in callback_results]
    predictions = [np.vstack(callback_predictions[i][0]) for i in range(len(callback_predictions))]
    targets = [np.vstack(callback_predictions[i][1]) for i in range(len(callback_predictions))]
    path = os.path.join(outputs_folder, 'predictions_and_test_data%s.npy' % suffix)
    np.save(path, {'predictions': predictions, 'test': targets})

    # Save learned weights.
    if params.output.save_weights:
        weights = [callbacks_fold[callback_idx_save_weights] for callbacks_fold in callback_results]
        if isinstance(weights[0], (list, tuple)):
            # There is more than one weight per fold.
            num_weights = len(weights[0])
            num_folds = len(weights)
            weights = [np.mean(np.stack([weights[f][i] for f in range(num_folds)]), axis=0)
                       for i in range(num_weights)]
            path = os.path.join(outputs_folder, 'weights.pickle')
            pickle.dump(weights, open(path, "wb"))
        else:
            weights = np.mean(np.stack(weights), axis=0)
            path = os.path.join(outputs_folder, 'weights.txt')
            np.savetxt(path, weights, newline='\n')
    print('Saved attention matrix at: ', path)

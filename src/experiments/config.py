import tensorflow as tf

from ..methods.models import regression
from ..methods.models.ridge_with_learned_attention import RidgeWithLearnedAttention
from ..util.containers import Parameters

__author__ = 'Otilia Stretcu'


def get_params():
    return Parameters({
        # Data loading and preprocessing parameters.
        'data': Parameters({
            # Data loading.
            'load_from_last_checkpoint': False,   # If True, it loads data from a latest preprocessed file.
                                                  # It will not take into account the configurations below.
            'data_dir': get_data_path(),
            'semantic_feature_path': get_semantic_feature_path(),
            'subject_id': 'G',
            'proc_slug': '',                      # An optional suffix appended to the brain data file name
                                                  # (e.g. in our case this contains information about the
                                                  # preprocessing that was done).
            'num_words_question_representation': 60,

            # Parameters for data preprocessing.
            'space_downsample': False,            # Whether to average over all sensors within each of 8 lobes
            'time_window_lower_bound': 0,         # Trial start time relative to stimulus onset, in ms.
            'time_window_length': 800,            # Time window to consider in ms.
            'word_embedding_type': 'MTurk_no_experiment_questions', # Type of question embedding to use: can be MTurk,
                                                  # MTurk, MTurk_no_experiment_questions,
                                                  # augmented_MTurk, augmented_MTurk_no_experiment_questions,
                                                  # word2vec, random, BERT_emb
            'question_embedding_type': 'MTurk',   # Type of question embedding to use: can be MTurk, one-hot, BERT_pooled, BERT_CLS
            'question_time_window_lower_bound': 500, # Question data start time relative to question onset, in ms.
            'question_time_window_length': 500,   # Question data duration to consider, in ms.
            'average_repetitions': False,         # If True, it will average stimulus repetitions.
            'normalize_brain_to_baseline': False, # Normalize all brain data to a baseline period [-250, 0] before question.
            'normalize_inputs': True,             # If True, we zscore the input features over the fold training data
                                                  # and we normalize the test inputs relative to the train mean and std.
            'normalize_outputs': True,            # If True, we zscore the output features over the fold training data
                                                  # and we normalize the test outputs relative to the train mean and std.
            'brain_scaling': None,                # Multiply each entry in the brain data with this coefficient.
                                                  # Set to None for no scaling.
            'avg_time_window_length': 25,         # Averge over the time dimension, using windows of this number of ms.
            'num_unique_train_words_to_keep': None, # Number of words (with all repetitions) to keep in the experiment. If None, all are kept.
            'seed_remove_words': 123,             # Seed when picking which words to keep.

            # Select what are the inputs and outputs to the model.
            'input_question_features':  True,   # If True, the question semantic features are added to the input. Set to True for H2, H3, H4.
            'input_question_brain':     False,  # If True, the question brain activity response is added to the input.
            'input_word_features':      True,   # If True, the word semantic features are added to the input. Set to True for H1, H3, H4.
            'input_word_brain':         False,  # If True, the word brain activity response is added to the input.
            'output_question_features': False,  # If True, the question semantic features are added to the targets.
            'output_question_brain':    False,  # If True, the question brain activity is added to the targets.
            'output_word_features':     False,  # If True, the word semantic features are added to the targets.
            'output_word_brain':        True,   # If True, the word brain activity is added to the targets.

            # Paths to other data to load.
            'sensor_location_path': get_sensor_location_path()
        }),
        # Parameters for plotting and printing.
        'output': Parameters({
            'path_outputs': get_output_path(),
            'output_dir_suffix': '',            # A suffix to add at the end of the results dir.
            'print_fold_results': True,
            'plot_prediction_error_per_time': True,
            'dist_metric_plot_per_time_pt': 'euclidean',  # Could be 'cosine', 'euclidean'.
            'save_attention': True,
            'save_weights': True,
        }),

        # Learning.
        'learning': Parameters({
            # Seed for the random number generators.
            'seed': 1234,
            'differentRegPerOutput': False,
            'load_best_params': False,
            'path_best_params': None,   # If load_best_params is True, the params will be loaded
                                        # from this path. If None, we compute the path automatically
                                        # for the current experiment configuration.
            'max_folds_test': 100,
            'max_folds_param_valid': 20,
            'use_2v2_val': False,       # Whether to use 2v2 as validation metric, or l2 loss.

            # Select below if you want to leave out data zero-shot in words, or questions or both.
            'zero_shot_words': True,
            'zero_shot_questions': True,

            # Select parameters for the Accuracy2v2WordQuestionGroups metric.
            'max_pairs': None,
            'majority_vote': False,
            'dist_metric': 'euclidean',
            'sensor_groups': get_sensor_groups(num_neighbors=27),  # If set to less than 3,
                                                                   # evaluates accuracy based on all
                                                                   # sensortimepoints together.

            # Select one of the regression models below. More models available in methods/models.
            # The model parameters are provided as dictionaries mapping from parameter name to
            # a list of possible values. If more than one value is provided, the training pipeline
            # will perform cross-validation to pick the best hyper-parameter configuration.
            'model': regression.RidgeRegression,
            'model_params': {'alpha': [1e5]},
            #'model': regression.LassoRegression,
            #'model_params': {'alpha': [1]},
            # 'model': RidgeWithLearnedAttention,
            # 'model_params': {
            #     'reg_weights': [1.0],
            #     'reg_question': [1.0],
            #     'use_sigmoid': [True],
            #     'use_softmax': [False],         # Set either use_softmax or use_sigmoid to True, or neither.
            #     'optimizer': [tf.train.AdamOptimizer],
            #     'max_iter': [20000],
            #     'batch_size': [512],
            #     'display_step': [100],
            #     'shuffle_batch': [True],
            #     'rel_loss_chg_tol': [1e-8],
            #     'abs_loss_chg_tol': [1e-5],
            #     'learning_rate': [1e-3],
            #     'max_steps_no_improvement': [1000],
            #     'checkpoint_step': [500]},
        })
    })


def get_sensor_groups(num_neighbors=1):
    """Groups nearby MEG sensors and returns the sensor groups."""
    from sklearn.metrics.pairwise import euclidean_distances
    import csv
    import numpy as np

    if num_neighbors is None or num_neighbors < 3:
        return np.ones([1, 306])

    path = get_sensor_location_path()
    with open(path, 'r') as f:
        locs = csv.reader(f,delimiter=',')
        loc306 = np.asarray([
            [float(w1[0].split(' ')[1]), float(w1[0].split(' ')[2])]
            for w1 in locs])

    loc102 = loc306[::3]
    dists = euclidean_distances(loc102, loc306)
    neighbors = np.argsort(dists,axis = 1)
    neighbors = neighbors[:, :num_neighbors]
    sensor_groups = np.zeros((102, 306))
    for i in range(102):
        sensor_groups[i, neighbors[i]] = 1
    return sensor_groups


def get_sensor_location_path():
    """Add here the path to the file containing the MEG sensor locations."""
    return './data/meg_sensor_locations.txt'


def get_semantic_feature_path():
    """Add here the path to the semantic features file.
     This is a file that contains the stimuli and question representations."""
    return 'data/MTurk_semantic_features.npz'


def get_data_path():
    """Add here the path to the folder containing brain data."""
    return ''


def get_output_path():
    """Add here the path where you want to save results."""
    return './outputs'

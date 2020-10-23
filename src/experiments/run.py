"""Main run script.

This script can predict from any input modality to any output modality.
The exact inputs and outputs should be set in the configuration file by setting the
following fields:
        'input_question_features':  True,   # If True, the question semantic features are added to the input.
        'input_question_brain':     False,  # If True, the question brain activity response is added to the input.
        'input_word_features':      False,  # If True, the word semantic features are added to the input.
        'input_word_brain':         True,   # If True, the word brain activity response is added to the input.
        'output_question_features': False,  # If True, the question semantic features are added to the targets.
        'output_question_brain':    False,  # If True, the question brain activity is added to the targets.
        'output_word_features':     True,   # If True, the word semantic features are added to the targets.
        'output_word_brain':        False,  # If True, the word brain activity is added to the targets.

To run a specific hypothesis from our paper, you should set:
    - H1:
        input_word_features = True
        input_question_features = False
        model = RidgeRegression
    - H2:
        input_word_features = False
        input_question_features = True
        model = RidgeRegression
    - H3:
        input_word_features = True
        input_question_features = True
        model = RidgeRegression
    - H4.1:
        input_word_features = True
        input_question_features = False
        word_embedding_type = augmented_MTurk_no_experiment_questions
        model = RidgeRegression
    - H4.2:
        input_word_features = True
        input_question_features = True
        model = RidgeWithLearnedAttention

Further parameters can be found in config.py.
"""
import json
import pickle
import logging
import os
import time
import tensorflow as tf

from ..data.load_20questions import load_all_data
from ..experiments.config import get_params
from ..experiments import helper
from ..methods.models.ridge_with_learned_attention import RidgeWithLearnedAttention
from ..methods.models.base import TfModel
from ..methods.trainer import Trainer, TrainerMultiRegularization
from .helper import create_results_string, load_best_params, \
    get_reg_params_name, create_iterators, get_serializable_params, \
    create_test_metrics, create_val_metric, save_results_and_plot

# TensorFlow configuration.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load experiment configuration and hyper-parameters.
params = get_params()

# Create results output folder.
outputs_folder = os.path.join(params.output.path_outputs, create_results_string(params))
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)

# Save config file in the same folder.
config_path = os.path.join(outputs_folder, 'config.json')
with open(config_path, 'w') as file:
    json.dump(get_serializable_params(params), file, indent=4)
    logging.info('Saved config file to: %s', config_path)

params = helper.fill_data_parameters(params)

# Create a directory where to store checkpoints for TensorFlow models.
if issubclass(params.learning.model, TfModel):
    checkpoints_dir = os.path.join(outputs_folder, 'checkpoints')
    params.learning.model_params['checkpoints_dir'] = [checkpoints_dir]

# Data loading.
start_time = time.time()
if params.data.load_from_last_checkpoint:
    # Since data preprocessing takes some time, we might want load the last preprocessed data.
    print('Loading data from last checkpoint...')
    word_id_to_class, question_id_to_class, word_trials, question_id_to_trial = \
        pickle.load(open(params.latest_data_path, "rb"))
else:
    # Load data and preprocess from scratch.
    print('Loading data...')
    word_id_to_class, question_id_to_class, word_trials, question_id_to_trial = load_all_data(
        semantic_feature_path=params.data.semantic_feature_path,
        subject_id=params.data.subject_id,
        subj_data_fname_data_array=params.data.subj_data_fname_data_array,
        subj_data_fname_question_array=params.data.subj_data_fname_question_array,
        space_downsample=params.data.space_downsample,
        time_window_lower_bound=params.data.time_window_lower_bound,
        time_window_length=params.data.time_window_length,
        word_embedding_type=params.data.word_embedding_type,
        question_embedding_type=params.data.question_embedding_type,
        num_words_question_representation=params.data.num_words_question_representation,
        question_time_window_lower_bound=params.data.question_time_window_lower_bound,
        question_time_window_length=params.data.question_time_window_length,
        normalize_to_baseline=params.data.normalize_brain_to_baseline,
        brain_scaling=params.data.brain_scaling,
        avg_time_window_length=params.data.avg_time_window_length)

    print('Saving preprocessed data to checkpoint file %s...' % params.latest_data_path)
    pickle.dump(
        (word_id_to_class, question_id_to_class, word_trials, question_id_to_trial),
        open(params.latest_data_path, "wb"))

print('Loading data done. Took %.2f seconds.' % (time.time() - start_time))


# Preparing model inputs and outputs.
print('Preparing model inputs and outputs...')
start_time = time.time()
inputs, outputs, groups = helper.create_inputs_outputs(
    word_trials, params.data, group_by_word=True, group_by_question=True)
print('inputs shape: ', inputs.shape)
print('outputs shape: ', outputs.shape)
print('Preparing model inputs and outputs done. Took %.2f seconds.' % (time.time() - start_time))


# Prepare training.
print('Preparing models...')
start_time = time.time()

# Create the cross validation data iterators.
cross_val, cross_val_param_valid = create_iterators(params.learning, groups)

# Create a postprocessing op that appends the groups to the predictions, so we know in 2v2 which
# word and question each sample is coming from.
postproc_op = lambda preds, groups: [(p, g) for p, g in zip(preds, groups)]

# Select what metrics you want to compute.
metrics = create_test_metrics(params)
val_metric, val_metric_higher_better = create_val_metric(
    params, use_2v2=params.learning.use_2v2_val)

# Fill in some parameters needed for training specific models.
num_question_features = len(word_trials[0].question_trial.question.features)
num_word_features = len(word_trials[0].word.features)
helper.fill_learner_params(params, inputs, outputs,
                           num_question_features=num_question_features,
                           num_word_features=num_word_features)

# Load model parameters from file, if applicable.
best_params_path = os.path.join(outputs_folder, 'best_reg_params.txt')
reg_params_name = get_reg_params_name(params.learning)
if params.learning.load_best_params:
    if params.learning.path_best_params:
        best_params_path = params.learning.path_best_params
    params.learning.model_params[reg_params_name] = [load_best_params(best_params_path)]

# Create a trainer that applies splits the data into train and test using cross-validation.
trainer_class = TrainerMultiRegularization if params.learning.differentRegPerOutput else Trainer
trainer = trainer_class(
    params.learning.model,
    params.learning.model_params,
    metrics=metrics,
    train_data=(inputs, outputs, groups),
    postproc_predictions=[postproc_op],
    cross_val_test=cross_val,
    cross_val_param_valid=cross_val_param_valid,
    validation_metric=val_metric,
    validation_metric_higher_better=val_metric_higher_better,
    max_folds_param_valid=params.learning.max_folds_param_valid,
    normalize_inputs=params.data.normalize_inputs,
    normalize_outputs=params.data.normalize_outputs,
    seed=params.learning.seed,
    param_name_reg=reg_params_name,
    best_params_save_path=best_params_path,
    num_unique_train_words_to_keep=params.data.num_unique_train_words_to_keep,
    seed_remove_words=params.data.seed_remove_words)
print('Preparing models done. Took %.2f seconds.' % (time.time() - start_time))

# Create some callbacks that save some results from each fold needed for plotting or visualization.
callbacks = []
if params.output.plot_prediction_error_per_time:
    callback_idx_dist_per_time = len(callbacks)
    callbacks.append(helper.callback_save_fold_results)
else:
    callback_idx_dist_per_time = None
if params.output.save_attention and params.learning.model == RidgeWithLearnedAttention:
    callback_idx_save_attention = len(callbacks)
    callbacks.append(helper.callback_save_attention)
else:
    callback_idx_save_attention = None
if params.output.save_weights:
    callback_idx_save_weights = len(callbacks)
    callbacks.append(helper.callback_save_weights)
else:
    callback_idx_save_weights = None
callback_idx_predictions = len(callbacks)
callbacks.append(helper.callback_save_fold_predictions)
num_callbacks = len(callbacks)


# Train and Test.
print('Training and testing...')
start_time = time.time()
callback_results = trainer.train_cross_val(
    callbacks=callbacks,
    print_results=params.output.print_fold_results)
print('Training and testing done. Took %.2f seconds.' % (time.time() - start_time))


# Plots and saving results.
save_results_and_plot(params, callback_results, outputs_folder, callback_idx_dist_per_time,
    callback_idx_save_attention, callback_idx_predictions, callback_idx_save_weights)


print('Done. Results can be found in the folder: ', outputs_folder)

import csv
import numpy as np
import time

from collections import OrderedDict

from .containers import Question, Word, WordTrial, QuestionTrial


def load_20questions_question_array(fname,
                                    time_window_lower_bound=None,
                                    time_window_length=None,
                                    baseline_time_window_lower_bound=-250,
                                    baseline_time_window_length=250):
    """Loads question-related data.

    Args:
        fname: Path to npz file containing the data.
        time_window_lower_bound: Select brain data for the question starting with this time point.
        time_window_length: Select a window of length time_window_length for the question brain data.
        baseline_time_window_lower_bound: The baseline brain activity starts at this time point.
        baseline_time_window_length: elect a window of this length for the baseline brain data.

    Returns:
        times_from_onset_ms: Times from stimulus onset.
        question_array: An array with shape (num_questions, num_channels, num_timepoints), where
            the questions are in order of inv_question_order_dict, and the times can be found in
            times_from_stim_onset.
        question_order_dict: Dictionary mapping from question string to question id.
        inv_question_order_dict: Dictionary mapping from question id to question string.
        baseline_mean:  An array of shape num_questions x num_channels representing the average
            brain activity over the baseline period.
        baseline_std: An array of shape num_questions x num_channels representing the standard
            deviation of the brain activity over the baseline period.
    """
    with open(fname, "rb") as fin:
        loaded_dict = np.load(fin, encoding='latin1', allow_pickle=True)
        
    times_from_onset_ms = loaded_dict['times_from_onset_ms']
    question_order_dict = loaded_dict['question_order_dict']
    inv_question_order_dict = loaded_dict['inv_question_order_dict']
    question_array = loaded_dict['question_array']
    
    if time_window_lower_bound is not None and time_window_length is not None:
        # Select only the requested time window for the question.
        time_window_start_ind = np.searchsorted(times_from_onset_ms, time_window_lower_bound, side='left')
        print('Including question data from time window: %.2f - %.2f ms' %
              (time_window_lower_bound, time_window_lower_bound+time_window_length))
        question_array = question_array[:, :, time_window_start_ind:time_window_start_ind+time_window_length]

        # Select only the requested time window for the baseline.
        time_window_start_ind = np.searchsorted(times_from_onset_ms, baseline_time_window_lower_bound, side='left')
        time_window_end_ind = time_window_start_ind + baseline_time_window_length
        print('Including baseline data from time window %.2f - %.2f ms relative to question onset.' %
              (baseline_time_window_lower_bound, baseline_time_window_lower_bound+baseline_time_window_length))
        baseline_array = loaded_dict['question_array'][:, :, time_window_start_ind: time_window_end_ind]
        # Average over the time dimension.
        baseline_mean = np.mean(baseline_array, axis=-1)
        baseline_std = np.std(baseline_array, axis=-1)
        assert (baseline_mean.shape == baseline_array.shape[:2])
    else:
        baseline_mean, baseline_std = None, None

    return times_from_onset_ms, question_array, question_order_dict, inv_question_order_dict, \
           baseline_mean, baseline_std


def load_20questions_data_array(fname):
    """Loads 20 questions brain data.

    The data is found in a .npy file, which contains a dictionary. See the "Returns" section for
    the different entries in the dictionary which are returned as is.

    Args:
        fname: Path to brain data array.

    Returns:
        times_from_onset_ms: Times from stimulus onset.
        data_array: An array with size num_stims x num_questions x num_channels x num_timepoints,
            where the stims are in order of inv_stimulus_order_dict, the questions are in order of
            inv_question_order_dict, and the times can be found in times_from_stim_onset.
        stimulus_order_dict: An array containing the sampling time of each of the time points in the
            brain data, relative to stimulus onset.
        inv_stimulus_order_dict: Dictionary mapping from id to stimulus string.
        question_order_dict: Dictionary mapping from question string to question id.
        inv_question_order_dict: Dictionary mapping from question id to question string.
    """
    with open(fname, "rb") as fin:
        loaded_dict = np.load(fin, allow_pickle=True)
    
    times_from_onset_ms = loaded_dict['times_from_onset_ms']
    stimulus_order_dict = loaded_dict['stimulus_order_dict']
    inv_stimulus_order_dict = loaded_dict['inv_stimulus_order_dict']
    question_order_dict = loaded_dict['question_order_dict']
    inv_question_order_dict = loaded_dict['inv_question_order_dict']
    data_array = loaded_dict['data_array']

    return times_from_onset_ms, data_array, stimulus_order_dict, inv_stimulus_order_dict, \
           question_order_dict, inv_question_order_dict


def load_20questions_data(fname):
    """Loads the 20-questions data.

    Args:
        fname: Path to npz file containing the data.

    Returns:
        times_from_onset_ms: Experimental times where the data comes from (-250ms to 1750ms)
            measured from onset of question or stimulus.
        data_dict: A dictionary where keys correspond to the question text, values are dictionaries
            with keys ['question_data' and stimulus_text], values data corresponding to the key
            [nchannels, ntime_points].
    """
    with open(fname, "rb") as fin:
        loaded_dict = np.load(fin, allow_pickle=True)
    times_from_onset_ms = loaded_dict.item()['times_from_onset_ms']
    data_dict = loaded_dict.item()['data_dict']
    return times_from_onset_ms, data_dict


def sort_questions_by_theme():
    """Returns a dictionary mapping a question text to an index, and the inverse dictionary."""
    question_order_dict = {}
    inv_question_order_dict = {}
    sorted_questions_by_theme = [  
        'Can you hold it?',
        'Can you hold it in one hand?',
        'Can you pick it up?',

        'Is it bigger than a loaf of bread?',
        'Is it bigger than a microwave oven?',
        'Is it bigger than a car?',

        'Can it keep you dry?',
        'Could you fit inside it?',
        'Does it have at least one hole?',
        'Is it hollow?',
        'Is part of it made of glass?',
        'Is it made of metal?',

        'Is it manufactured?',
        'Is it manmade?',

        'Is it alive?',
        'Was it ever alive?',
        'Does it grow?',
        'Does it have feelings?',
        'Does it live in groups?',

        'Is it hard to catch?']
    for q, question in enumerate(sorted_questions_by_theme):
        question_order_dict[question] = q
        inv_question_order_dict[q] = question
    return question_order_dict, inv_question_order_dict


def get_all_stimulus_text(semantic_features):
    """Returns all stimuli texts."""
    return semantic_features['stimuli'][0:60]


def get_all_questions_text():
    """Return all question texts."""
    questions_dict, _ = sort_questions_by_theme()
    return np.array([question for question in questions_dict.keys()])


def get_feature_label(semantic_features, feat_ind):
    """Returns specific semantic feature label."""
    return semantic_features['features'][feat_ind]


def get_feat_ind(semantic_features, feat):
    """Returns the index of a specific word feature / question."""
    for i, f in enumerate(semantic_features['features']):
        if f == feat:
            return i
    return None
  
  
def get_channel_lobe_inds():
    channel_lobe_inds = OrderedDict()
    lobe_names = ['R_frontal', 'L_frontal',
                  'R_temporal', 'L_temporal',
                  'R_parietal', 'L_parietal',
                  'R_occipital','L_occipital']
    lobe_inds = [
        [84,  85,  86,  90,  91,  92,  93,  94,  95, 96,  97,  98,  99, 100, 101, 102, 103, 104,
         105, 106, 107, 108, 109, 110, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
         150, 151, 152],
        [3,  4,  5, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 48, 49, 50, 51, 52, 53, 54,
         55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 69, 70, 71, 87, 88, 89],
        [138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 153, 154, 155, 156, 157, 158,
         159, 160, 161, 270, 271, 272, 273, 274, 275, 294, 295, 296, 297, 298, 299, 300, 301, 302,
         303, 304, 305],
        [0,   1,   2,   6,   7,   8,   9,  10,  11, 12,  13,  14,  15, 16,  17,  18,  19,  20,
         21,  22,  23, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
         177, 178, 179],
        [75,  76,  77,  78,  79,  80, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
         123, 124, 125, 225, 226, 227, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257,
         279, 280, 281],
        [36,  37,  38,  39,  40,  41,  42,  43,  44, 45,  46,  47,  66, 67,  68,  72,  73,  74,
         81,  82,  83, 180, 181, 182, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
         222, 223, 224],
        [228, 229, 230, 237, 238, 239, 240, 241, 242, 258, 259, 260, 261, 262, 263, 264, 265, 266,
         267, 268, 269, 276, 277, 278, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293],
        [183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 210, 211, 212,
         213, 214, 215, 216, 217, 218, 219, 220, 221, 231, 232, 233, 234, 235, 236, 243, 244, 245]]
    
    for l, lobe in enumerate(lobe_inds):
        for k in range(3):
            sensors = [j for i, j in enumerate(lobe) if i % 3 == k]
            for m in range(0, len(sensors), 2):
                channel_lobe_inds[lobe_names[l]+'_'+str(k)+'_'+str(m)] = sensors[m:m+2]

    return channel_lobe_inds


def get_stimulus_representation(semantic_features, stimulus_text, embedding_type='MTurk'):
    """Returns representation for stimulus_text as vector of features.

    Args:
        semantic_features: An npz object containing the semantic features.
        stimulus_text: A string representing the stimulus of interest.
        embedding_type: A string representing the word embedding type.

    Returns:
        A 1-D numpy array containing the stimulus embedding.
    """
    if embedding_type in ['MTurk', 'augmented_MTurk', 'MTurk_learnt_attention',
                          'MTurk_learnt_attention_no_exp_questions']:
        stimulus_ind = np.where(get_all_stimulus_text(semantic_features) == stimulus_text)[0][0]
        return semantic_features['vectors'][stimulus_ind, :]
    elif (embedding_type == 'MTurk_no_experiment_questions' or
          embedding_type == 'augmented_MTurk_no_experiment_questions'):
        # Find the experiment question indices in the stimuli vectors.
        question_order_dict, inv_question_order_dict = sort_questions_by_theme()
        num_questions = len(question_order_dict)
        experiment_question_inds = []
        for question in range(0, num_questions):
            question_text = inv_question_order_dict[question]
            question_ind = get_feat_ind(semantic_features, question_text.upper())
            experiment_question_inds.append(question_ind)
        to_keep_inds = []
        for ind in range(0, 229):
            if ind not in experiment_question_inds:
                to_keep_inds.append(int(ind))
        to_keep_inds = np.array(to_keep_inds)
        
        stimulus_ind = np.where(get_all_stimulus_text(semantic_features) == stimulus_text)[0][0]
        return semantic_features['vectors'][stimulus_ind, to_keep_inds]
    elif embedding_type == 'word2vec':
        with open('data/word2vec_dict.npz', 'rb') as fin:
            word2vec_representations = np.load(fin, allow_pickle=True)
        return word2vec_representations.item()[stimulus_text]
    elif embedding_type == 'random':
        import numpy.random
        return numpy.random.rand(229)
    elif embedding_type == 'BERT_emb':
        embeddings = np.load('data/stim_rep_BERT_emb.npy')
        stim_ind = np.where(semantic_features['stimuli'][:60] == stimulus_text)[0][0]
        return embeddings[stim_ind, :]
    else:
        raise NameError('The provided embedding_type argument not recognized: only MTurk, '
            'augmented_MTurk, word2vec, random, MTurk_learnt_attention, '
            'MTurk_learnt_attention_no_exp_questions, MTurk_no_experiment_questions, BERT_emb '
            'are supported.')


def get_question_representation(semantic_features, question_text, num_words=60):
    """Returns a vector representation for a provided question as vector of MTurk responses.

    Each question representation is a vector of size num_words representing the answer to the
    given question for num_words words.  These are the first num_words words following the
    first 60 words (because the first 60 are also the stimuli words).

    Args:
        semantic_features: Loaded npz object.
        question_text: A string representing the question text.
        num_words: Number of words to include in the question representation.

    Returns:
        Returns a vector of length num_words containing the question representation.
    """
    # Leave out the first 60 words from the question representation.
    word_inds = np.array(range(60, min(60+num_words, 1000)))
    capitalized_question_text = question_text.upper()     
    question_ind = np.where(semantic_features['features'] == capitalized_question_text)[0][0]
    return semantic_features['vectors'][word_inds, question_ind]


def construct_feature_to_feature_rdm(semantic_features, dist='cosine'):
    """Creates a similarity matrix between all word features (which also correspond to questions)."""
    num_feats = len(semantic_features['features'])
    feature_RDM = np.zeros([num_feats, num_feats])

    sem_mat = [semantic_features['vectors'][60:120,feat_ind] for feat_ind in range(0, num_feats)]
    
    if dist == 'dotproduct':
        for f, feat in enumerate(sem_mat):
            for f2, feat2 in enumerate(sem_mat):
                feature_RDM[f, f2] = np.dot(feat, feat2)
    else:
        import scipy.spatial.distance as spd
        feature_RDM = 1 - spd.squareform(spd.pdist(np.array(sem_mat), dist))
    
    return feature_RDM


def get_augmented_feature_weights(semantic_features, exclude_exp_qs=True, dist='cosine'):
    """Returns dict of weights that the word features need to be augmented by for a given question
    keys of dict are the question texts, values are the weights weights correspond to similairties
    between word feature/question.

    Args:
        semantic_features:
        exclude_exp_qs: Boolean specifying whether to exclude the experiment questions.
        dist: Distance metric to use. Defaults to cosine.

    Returns:
        augmented_feature_weights
    """
    from scipy.special import softmax
    
    feature_RDM = construct_feature_to_feature_rdm(semantic_features, dist=dist)
    question_order_dict, inv_question_order_dict = sort_questions_by_theme()
    num_questions = len(question_order_dict)
    augmented_feature_weights = dict()
    
    question_inds = [get_feat_ind(semantic_features, inv_question_order_dict[question].upper()) for question in range(0,num_questions)]
    non_question_inds = [ind for ind in range(feature_RDM.shape[1]) if ind not in question_inds]
    
    for question in range(0,num_questions):
        question_text = inv_question_order_dict[question]
        question_ind = get_feat_ind(semantic_features, question_text.upper())
        
        if exclude_exp_qs:
            augmented_feature_weights[question_text] = softmax(feature_RDM[question_ind,:][non_question_inds])
        else:
            augmented_feature_weights[question_text] = softmax(feature_RDM[question_ind,:])
    return augmented_feature_weights


def load_all_data(semantic_feature_path, subject_id, subj_data_fname_data_array,
                  subj_data_fname_question_array, space_downsample, time_window_lower_bound,
                  time_window_length, word_embedding_type='MTurk', question_embedding_type='MTurk',
                  num_words_question_representation=60, question_time_window_lower_bound=None,
                  question_time_window_length=None, normalize_to_baseline=False, brain_scaling=None,
                  avg_time_window_length=None):
    """
    Loads the stimuli information, question information and brain data.
    Args:
        semantic_feature_path: Path to the semantic features .npz file.
        subject_id: Id of the subject for which to load the data.
        subj_data_fname_data_array: Path to the subject data .npz file that contains data arrays.
        subj_data_fname_all: Path to the subject data .npz file.
        space_downsample: Whether to average over sensors within each predefined lobe.
        time_window_lower_bound: We select trial data starting at the time point
            `time_window_lower_bound` relative to the trial onset.
        time_window_length: We select trial data for a duration of `time_window_length` ms.
        embedding_type: Type of representation we use for the stimuli (can be 'MTurk','word2vec',
            'augmented_MTurk', 'MTurk_no_experiment_questions',
            'augmented_MTurk_no_experiment_questions', 'random','BERT_emb').
        num_words_question_representation: We represent a question semantically by the scores the
            first `num_words_question_representation` words get as answers to this question.
        question_time_window_lower_bound: Question data start time relative to question onset, in ms.
        question_time_window_length: Question data duration to consider, in ms.
        avg_time_window_length: Integer specifying to average data over the time dimension in
            windows of `avg_time_window_length` elements.
    Returns: A triple (stim_id_to_class, question_id_to_class, trials), where
        stim_id_to_class     is a dictionary mapping from word id to a Word object.
        question_id_to_class is a dictionary mapping from question id to a Question object.
        trials               is a list of WordTrial objects.
    """

    # Load semantic features in an npz file.
    print('Loading semantic features...')
    start_time = time.time()
    semantic_features = np.load(open(semantic_feature_path, "rb"))
    print('Loading semantic features done. Took %.2f seconds.' % (time.time() - start_time))

    # Get all possible stimulus texts, in an array of strings of length 60.
    print('Loading stimuli text and semantic features...')
    start_time = time.time()
    all_stimulus_text = get_all_stimulus_text(semantic_features)
    stim_id_to_class = OrderedDict()
    for stim_ind, current_stimulus_text in enumerate(all_stimulus_text):
        # Get semantic features for the current stimulus, and create the corresponding Word object.
        feat = get_stimulus_representation(
            semantic_features, current_stimulus_text, word_embedding_type)
        stim_id_to_class[stim_ind] = Word(
            id=stim_ind,
            text=current_stimulus_text,
            semantic_features=feat)
    print('Loading stimuli done. Took %.2f seconds.' % (time.time() - start_time))

    # Load all data for the current subject.
    print('Loading data for subject %s..' % subject_id)
    start_time = time.time()
    times_from_onset_ms, data_array, stimulus_order_dict, inv_stimulus_order_dict, \
    question_order_dict, inv_question_order_dict = load_20questions_data_array(subj_data_fname_data_array)
    
    if space_downsample:
        print('Downsampling data in space..')
        channel_inds_dict = get_channel_lobe_inds()
        ds_data_array = []
        for lobe_inds in channel_inds_dict.values():
            ds_data_array.append(np.mean(data_array[:, :, lobe_inds, :], axis=2))
        data_array = np.stack(ds_data_array, axis=2)
        print('Done downsampling data in space. New dimensions:{}'.format(data_array.shape))

    # Load question brain data.
    _, question_array, question_order_dict, inv_question_order_dict, baseline_mean, baseline_std = \
        load_20questions_question_array(
            subj_data_fname_question_array,
            time_window_lower_bound=question_time_window_lower_bound,
            time_window_length=question_time_window_length)
    if brain_scaling is not None:
        question_array = question_array * brain_scaling
        data_array = data_array * brain_scaling

    # Potentially normalize the question brain activity to baseline.
    if normalize_to_baseline:
        print('Normalizing all data to baseline before question...')
        baseline_std[baseline_std == 0.0] = 1.0
        question_array = (question_array - baseline_mean[:, :, None]) / baseline_std[:, :, None]
        data_array = (data_array - baseline_mean[:, :, None]) / baseline_std[:, :, None]

    question_id_to_data = OrderedDict([(q_id, question_array[i])
        for q_id, i in enumerate(inv_question_order_dict.keys())])
    print('Loading data done. Took %.2f seconds.' % (time.time() - start_time))

    # Get question representations using the 60 experiment nouns.
    print('Loading question representations...')
    question_id_to_class = OrderedDict()
    question_id_to_trial = OrderedDict()
    start_time = time.time()
    num_questions = len(inv_question_order_dict)
    for question_ind in range(0, num_questions):
        current_question_text = inv_question_order_dict[question_ind]

        # Load semantic features for a question.
        if question_embedding_type == 'MTurk':
            current_question_representation = get_question_representation(
            semantic_features, current_question_text, num_words=num_words_question_representation)
        elif question_embedding_type == 'one-hot':
            current_question_representation = np.zeros(60)
            current_question_representation[question_ind] = 1
        elif question_embedding_type == 'BERT_pooled':
            pooled = np.load('data/quest_rep_BERT_pooled.npy')
            current_question_representation = pooled[question_ind,:]
        elif question_embedding_type == 'BERT_CLS':
            pooled = np.load('data/quest_rep_BERT_CLS_last_layer.npy')
            current_question_representation = pooled[question_ind, :]
        else:
            raise NameError('The specified question_embedding_type argument not recognized: '
                            'only MTurk and one-hot supported.')
            
        question_id_to_class[question_ind] = Question(
            id=question_ind,
            text=current_question_text,
            semantic_features=current_question_representation)

        # Load brain activity for a question.
        question_id_to_trial[question_ind] = QuestionTrial(
            question=question_id_to_class[question_ind],
            question_brain_activity=question_id_to_data[question_ind])
    print('Loading question done. Took %.2f seconds.' % (time.time() - start_time))

    # Get trial brain activity data.
    start_time = time.time()
    time_window_start_ind = np.searchsorted(times_from_onset_ms, time_window_lower_bound, side='left')
    print('Including stimuli data from time window: %.2f - %.2f ms' %
          (time_window_lower_bound, time_window_lower_bound+time_window_length))
    trials = []
    for stim_ind, current_stimulus_text in enumerate(all_stimulus_text):
        # Load all trials for this stimulus in an array of shape (num_questions, num_sensors, time)
        current_stimulus_data = data_array[stim_ind, :, :, time_window_start_ind:time_window_start_ind+time_window_length]

        # Average data in windows over the time dimension.
        if avg_time_window_length:
            current_stimulus_data = average_over_time(current_stimulus_data, avg_time_window_length)

        for question_id in range(current_stimulus_data.shape[0]):
            trials.append(WordTrial(
                word=stim_id_to_class[stim_ind],
                question_trial=question_id_to_trial[question_id],
                brain_activity=current_stimulus_data[question_id]))

    print('Preparing trials done. Took %.2f seconds.' % (time.time() - start_time))

    return stim_id_to_class, question_id_to_class, trials, question_id_to_trial


def average_over_time(data, window_length):
    """Average brain data over time windows of provided length."""
    num_questions, num_channels, num_time = data.shape
    assert num_time > window_length
    num_time_new = num_time // window_length
    averaged_data = np.zeros((num_questions, num_channels, num_time_new))
    for i in range(num_time_new):
        averaged_data[:, :, i] = np.mean(
            data[:, :, window_length*i:window_length*(i+1)], axis=-1)
    return averaged_data


def get_sensor_locations(path, num_sensors=306):
    """Load the locations of the MEG sensors."""
    assert num_sensors in [102, 306]
    with open(path, 'r') as f:
        locs = csv.reader(f, delimiter=',')
        loc306 = np.array([
            [float(w1[0].split(' ')[1]), float(w1[0].split(' ')[2])]
            for w1 in locs])
        if num_sensors == 306:
            return loc306
        loc102 = loc306[::3]
        return loc102

# Data Description

We thank Gustavo Sudre for collecting and sharing the MEG dataset, and 
Dean Pomerleau for the human-judgment Mechanical Turk dataset!
With their permission we provide the Mechanical Turk data. Additionally,
we provide the BERT and word2vec representations for the stimuli and questions
used in our experiments. Further details below.


### Mechanical Turk responses
The Mechanical Turk responses for all combinations of 1000 words and 
218 questions can be found at `MTurk_semantic_features.npz`. There are additional 
11 perceptual features for each word that were added by the experimenters.
These can be loaded as follows:
```python
import numpy
semantic_feature_path = 'MTurk_semantic_features.npz'
semantic_features = np.load(open(semantic_feature_path, 'rb'))
```
The variable `semantic_features` has a dictionary structure containing the 
following keys:
- `'stimuli'`: A numpy array of strings containing the text of the 1000 words shown during the Mechanical Turk experiment.
- `'features'`: A numpy array of strings containing the text of the 218 questions asked during the Mechanical Turk experiment, and the names of the 11 additional perceptual features.
- `'vectors'`: A numpy array of shape (1000, 229), where each element (i,j) represents the answer to question j about word i. These elements are integers from {1, 2, 3, 4, 5} representing the degree to which question j applies to word i.

### Word2vec semantic features
We also provide the word2vec vector representations for the stimuli used in our
experiment. These can be found at `word2vec_dict.npz`. This can be loaded
as follows:
```python
import numpy
with open('word2vec_dict.npz', 'rb') as fin:
    word2vec_representations = np.load(fin, allow_pickle=True).item()
```
The variable `word2vec_representations` is a dictionary mapping from the text
of a word (e.g. 'carrot') to its 300-dimensional word2vec vector representation.


### BERT semantic features
We additionally provide the BERT vector representations for the stimuli and questions used in our
experiment. These can be found at `BERT_dict.npy`. This can be loaded
as follows:
```python
import numpy
with open('BERT_dict.npy', 'rb') as fin:
    BERT_representations = np.load(fin, allow_pickle=True).item()
```
The variable `BERT_representation` is a dictionary of dictionaries with the following keys: 
- `'questions_text'`: the text that corresponds to each question
- `'stimuli_text'`: the text that corresponds to each stimulus
- `'questions_BERT_pooled`: the BERT representations from the pooled output (described in Appendix D) in the same order as in `'questions_text'`
- `'questions_BERT_CLS'`: the BERT representations from the CLS token at the last layer (described in Appendix D) in the same order as in `'questions_text'` 
- `'stimuli_BERT'`: the token-level word embeddings from BERT (described in Appendix D) in the same order as in `'stimuli_text'`


## MEG sensor locations
We provide the locations of the MEG sensors in `meg_sensor_locations.txt`.
This file contains information about each of the 306 sensors, one per line, as 
follows:
```
sensor_number x y sensor_id
```
where x and y are the coordinates of the 2D projection of the sensor locations.

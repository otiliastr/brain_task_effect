# Data Description

We thank Gustavo Sudre for collecting and sharing the MEG dataset, and 
Dean Pomerleau for the human-judgment Mechanical Turk dataset!
With their permission we provide the Mechanical Turk data. Additionally,
we provide the BERT and word2vec representations for the stimuli and questions
used in our experiments. Further details below.


### Mechanical Turk responses
The Mechanical Turk responses for all combinations of 1000 questions and 
229 stimuli can be found at `MTurk_semantic_features.npz`. 
These can be loaded as follows:
```python
import numpy
semantic_feature_path = 'MTurk_semantic_features.npz'
semantic_features = np.load(open(semantic_feature_path, 'rb'))
```
The variable `semantic_features` has a dictionary structure containing the 
following fields:
```python
semantic_features['stimuli']: A numpy array of strings containing the text of 
    the 1000 words shown during the Mechanical Turk experiment.
semantic_features['features']: A numpy array of strings containing the text of 
    the 229 questions asked during the Mechanical Turk experiment.
semantic_features['vectors']: A numpy array of shape (1000, 229), where each
    element (i,j) represents the answer to question j about word i. These 
    elements are integers from {1, 2, 3, 4, 5} representing the degree
    to which question j applies to word i.
```

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


## MEG sensor locations
We provide the locations of the MEG sensors in `meg_sensor_locations.txt`.
This file contains information about each of the 306 sensors, one per line, as 
follows:
```
sensor_number x y sensor_id
```
where x and y are the coordinates of the 2D projection of the sensor locations.

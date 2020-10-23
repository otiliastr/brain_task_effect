__author__ = 'Otilia Stretcu'


class Word:
    """Stores semantic information for a word. """
    def __init__(self, id, text, semantic_features):
        """Creates an object containing semantic information about a single word.
        Args:
            id: An integer uniquely identifying the word.
            text: A string representing the stimulus text (e.g. "dog").
            semantic_features: A 1-D numpy array representing the embedding of the current word.
        """
        self.id = id
        self.text = text
        self.features = semantic_features

    def copy(self, id=None, text=None, semantic_features=None):
        id = self.id if id is None else id
        text = self.text if text is None else text
        features = self.features if semantic_features is None else semantic_features
        return Word(id, text, features)


class Question:
    """Stores semantic information for a question. """
    def __init__(self, id, text, semantic_features):
        """Creates an object containing semantic information about a single question.

        Args:
            id: An integer uniquely identifying the question.
            text: A string representing the stimulus text (e.g. "Can you hold it?").
            semantic_features: A 1-D numpy array representing the embedding of the current question.
        """
        self.id = id
        self.text = text
        self.features = semantic_features


class WordTrial:
    """Stores brain activity information for a single word stimulus. """
    def __init__(self, word, question_trial, brain_activity):
        """Creates an object containing semantic information about a word trial.
         This is a combination of word-question shown to the experiment participants,
         as well as the recorded brain activity for one participant.

        Args:
            word: A Word object.
            question_trial: A QuestionTrial object.
            brain_activity: A numpy array of shape (num_sensors, num_time_steps) containing the
                recorded brain activity for one participant when during the provided question trial.
        """
        self.word = word
        self.question_trial = question_trial
        self.brain = brain_activity

    def copy(self, word=None, question_trial=None, brain_activity=None):
        word = self.word if word is None else word
        question_trial = self.question_trial if question_trial is None else question_trial
        brain = self.brain if brain_activity is None else brain_activity
        return WordTrial(word, question_trial, brain)


class QuestionTrial:
    """Stores brain activity information for a question."""
    def __init__(self, question, question_brain_activity):
        """Creates an object containing brain activity information for a question.

        Args:
            question: A Question object.
            question_brain_activity: A numpy array of shape (num_sensors, num_time_steps) containing
                the recorded brain activity of a participant when shown this `question`.
        """
        self.question = question
        self.brain = question_brain_activity


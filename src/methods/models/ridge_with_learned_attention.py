"""Implementation of Hypothesis 4.2 which learns the attention weights."""
import logging
import pickle
import tensorflow as tf

from .base import TfModel


__author__ = 'Otilia Stretcu'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RidgeWithLearnedAttention(TfModel):
    """Implementation of Hypothesis 4.2 which learns the attention weights."""
    def __init__(self, num_word_features, num_question_features, num_outputs,
                 use_sigmoid=False,
                 use_softmax=False,
                 reg_weights=0.00000,
                 reg_question=0.00000,
                 optimizer=tf.train.AdamOptimizer,
                 learning_rate=1e-3,
                 max_iter=10000,
                 max_steps_no_improvement=1000,
                 batch_size=32,
                 abs_loss_chg_tol=1e-10,
                 rel_loss_chg_tol=1e-3,
                 loss_chg_iter_below_tol=5,
                 display_step=1,
                 shuffle_batch=True,
                 checkpoints_dir=None,
                 checkpoint_step=1000,
                 path_initial_values=None):
        assert not (use_softmax and use_sigmoid), "Choose either sigmoid or softmax, or neither."
        optimization_opts = {
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'max_steps_no_improvement': max_steps_no_improvement,
            'batch_size': batch_size,
            'abs_loss_chg_tol': abs_loss_chg_tol,
            'rel_loss_chg_tol': rel_loss_chg_tol,
            'loss_chg_iter_below_tol': loss_chg_iter_below_tol,
            'display_step': display_step}

        model_params = {
            'num_outputs': num_outputs,
            'num_question_features': num_question_features,
            'num_word_features': num_word_features,
            'use_sigmoid': use_sigmoid,
            'use_softmax': use_softmax,
            'path_initial_values': path_initial_values}

        super(RidgeWithLearnedAttention, self).__init__(
            model_class=RidgeQuestionSemanticsKeras,
            model_params=model_params,
            optimizer=optimizer,
            optimization_opts=optimization_opts,
            shuffle_batch=shuffle_batch,
            checkpoints_dir=checkpoints_dir,
            checkpoint_step=checkpoint_step)

        self.num_outputs = num_outputs
        self.num_word_features = num_word_features
        self.num_question_features = num_question_features
        self.reg_weights = reg_weights
        self.reg_question = reg_question

        self.loss_fn = tf.compat.v1.keras.losses.MeanSquaredError(reduction='none')

    def predict(self, data_inputs, training=False):
        data_inputs = self.process_inputs_targets(data_inputs)
        return self.model(data_inputs).numpy()

    def loss(self, predictions, targets):
        return tf.reduce_sum(self.loss_fn(predictions, targets)) + \
               tf.nn.l2_loss(self.reg_weights * self.model.layer.W) + \
               tf.nn.l2_loss(self.reg_question * self.model.layer.W_att)

    def parameters(self):
        return self.model.layer.W_att.numpy(), self.model.layer.bias_att.numpy(), \
               self.model.layer.W.numpy(), self.model.layer.bias.numpy()

    def get_attention(self):
        return self.model.layer.W_att.numpy()

    def __str__(self):
        return 'ridge_with_question_%d_%d_%f_%f' % \
               (self.num_word_features, self.num_outputs, self.reg_weights, self.reg_question)

    @staticmethod
    def class_name():
        return 'ridge_with_question'


class RidgeQuestionSemanticsKeras(tf.keras.Model):
    """Implementation of Hypothesis 4.2 which learns the attention weights."""
    def __init__(self, num_outputs, num_question_features, num_word_features, use_sigmoid,
                 use_softmax, name='RidgeQuestionKeras', path_initial_values=None):
        super(RidgeQuestionSemanticsKeras, self).__init__(name=name)

        # Create layers.
        self.layer = RidgeQuestionSemanticsBlock(
            num_question_features=num_question_features,
            num_word_features=num_word_features,
            num_outputs=num_outputs,
            use_sigmoid=use_sigmoid,
            use_softmax=use_softmax,
            path_initial_values=path_initial_values)

    def call(self, inputs, training=None, mask=None):
        return self.layer(inputs, training=training)


class RidgeQuestionSemanticsBlock(tf.keras.layers.Layer):

    def __init__(self, num_question_features, num_word_features, num_outputs, use_sigmoid,
                 use_softmax, path_initial_values=None):
        super(RidgeQuestionSemanticsBlock, self).__init__()
        self.num_question_features = num_question_features
        self.num_word_features = num_word_features
        self.num_outputs = num_outputs
        self.use_sigmoid = use_sigmoid
        self.use_softmax = use_softmax
        self.path_initial_values = path_initial_values
        assert not (use_softmax and use_sigmoid)

    def build(self, input_shape):
        if self.path_initial_values:
            with open(self.path_initial_values, 'rb') as file:
                print('Loading inital attention value from: ', self.path_initial_values)
                initial_weight = pickle.load(file)

            def my_init(shape, dtype=None):
                return tf.convert_to_tensor(initial_weight, dtype=dtype)

            self.W_att = self.add_weight(
                name='W_att',
                shape=(self.num_question_features, self.num_word_features),
                initializer=my_init,
                dtype=tf.float32,
                use_resource=True,
                trainable=True)
        else:
            self.W_att = self.add_weight(
                name='W_att',
                shape=(self.num_question_features, self.num_word_features),
                initializer='random_normal',
                dtype=tf.float32,
                use_resource=True,
                trainable=True)
        self.bias_att = self.add_weight(
            shape=(self.num_word_features,),
            name='bias_att',
            initializer='zeros',
            dtype=tf.float32,
            use_resource=True,
            trainable=True)
        self.W = self.add_weight(
            name='W',
            shape=(self.num_word_features, self.num_outputs),
            initializer='random_normal',
            dtype=tf.float32,
            use_resource=True,
            trainable=True)
        self.bias = self.add_weight(
            shape=(self.num_outputs,),
            name='bias',
            initializer='zeros',
            dtype=tf.float32,
            use_resource=True,
            trainable=True)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, tf.float32)
        word_features = inputs[:, :self.num_word_features]
        question_features = inputs[:, self.num_word_features:]

        # Select the rows in q that correspond to the indices in question_indices.
        attention = tf.nn.xw_plus_b(question_features, self.W_att, self.bias_att)
        if self.use_sigmoid:
            attention = tf.nn.sigmoid(attention)
        elif self.use_softmax:
            attention = tf.nn.softmax(attention)
        augmented_features = tf.multiply(attention, word_features)
        predictions = tf.nn.xw_plus_b(augmented_features, self.W, self.bias)

        return predictions

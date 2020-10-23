"""Contains the base classes extended by the models."""

import abc
import keras
import logging
import numpy as np
import os
import sys
import tensorflow as tf

from six import with_metaclass

from ..learning_rate_schedules import get_lr_schedule, FakeModel
from ..iterators import batch_iterator

__author__ = 'Otilia Stretcu'

logger = logging.getLogger(__name__)


class Model(with_metaclass(abc.ABCMeta, object)):
    """Base model class."""
    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def train(self, inputs, outputs):
        """
        Trains this learner.

        Args:
        inputs: A np.ndarray or tuple or list or dict, containing the input data.
        outputs: A np.ndarray or tuple or list or dict, containing the output data. For example,
            for a classification problem, this would correspond to the classes of the input data.
        """
        pass

    @abc.abstractmethod
    def predict(self, inputs):
        """
        Predicts the outputs for the provided input data.

        Args:
        inputs: A np.ndarray or tuple or list or dict containing the input data.

        Returns:
            A np.ndarray or tuple or list or dict containing the predicted output data.
        """
        pass


class SKLearnModel(with_metaclass(abc.ABCMeta, Model)):
    """Abstract class providing some common functionality among Scikit-Learn based learners."""
    def __init__(self, sk_learner):
        super(SKLearnModel, self).__init__()
        self._sk_learner = sk_learner

    @abc.abstractmethod
    def __str__(self):
        pass

    def train(self, inputs, outputs):
        """Trains this learner.

        Args:
            inputs (np.ndarray): Input data. The first dimension of the array
                corresponds to the data instances.
            outputs (np.ndarray): Output data. The first dimension of the array
                corresponds to the data instances.
        """
        self._sk_learner.fit(inputs, outputs)

    def predict(self, inputs):
        """Predicts the outputs for the provided input data.

        Args:
            inputs (np.ndarray): Input data. The first dimension of the array
                corresponds to the data samples.

        Returns:
            np.ndarray: Predicted output data, with matching first dimension
                with the provided input data array.
        """
        predictions = self._predict(inputs)
        if predictions.shape[1] == 2:
            return predictions[:, 1]
        return predictions

    def parameters(self):
        return self._sk_learner.coef_


class TfModel(Model):
    """Superclass for TensorFlow models. Contains code for training a model using TensorFlow."""
    def __init__(self, model_class, model_params, optimizer, optimization_opts=None,
                 logging_level=0, shuffle_batch=True, checkpoints_dir=None, checkpoint_step=500):
        self.model_class = model_class
        self.model_params = model_params
        self.optimizer = optimizer
        self.optimization_opts = optimization_opts if optimization_opts is not None else dict()
        self.logging_level = logging_level
        self.shuffle_batch = shuffle_batch
        self.model = None
        self.checkpoints_dir = checkpoints_dir
        self.checkpoint_step = checkpoint_step

    def train(self, data_inputs, data_targets, display_step=10):
        batch_size = self.optimization_opts.get('batch_size', None)
        max_iter = self.optimization_opts.get('max_iter', 10000)
        abs_loss_chg_tol = self.optimization_opts.get('abs_loss_chg_tol', 1e-10)
        rel_loss_chg_tol = self.optimization_opts.get('rel_loss_chg_tol', 1e-5)
        loss_chg_iter_below_tol = self.optimization_opts.get('loss_chg_iter_below_tol', 10)
        display_step = self.optimization_opts.get('display_step', display_step)
        learning_rate = self.optimization_opts.get('learning_rate', 1e-3)
        max_steps_no_improvement = self.optimization_opts.get('max_steps_no_improvement', 1000)
        clip_gradient_norm = self.optimization_opts.get('clip_gradient_norm', None)
        lr_decay_rate = self.optimization_opts.get('lr_decay_rate', 0.9)
        lr_decay_step = self.optimization_opts.get('lr_decay_step', None)
        lr_decay_per_iter = self.optimization_opts.get('lr_decay_per_iter', False)
        lr_decay_start_step = self.optimization_opts.get('lr_decay_start_step', None)
        lr_decay_end_step = self.optimization_opts.get('lr_decay_end_step', None)
        lr_warmup_init = self.optimization_opts.get('lr_warmup_init', 1e-9)
        lr_warmup_final = self.optimization_opts.get('lr_warmup_final', learning_rate)
        lr_warmup_start_step = self.optimization_opts.get('lr_warmup_start_step', 0)
        lr_warmup_end_step = self.optimization_opts.get('lr_warmup_end_step', None)
        lr_warmup_per_iter = self.optimization_opts.get('lr_warmup_per_iter', False)

        tf.keras.backend.clear_session()
        self.model = self.model_class(**self.model_params)
        if self.optimizer == tf.train.MomentumOptimizer:
            optimizer = self.optimizer(learning_rate=learning_rate, momentum=0.9)
        else:
            optimizer = self.optimizer(learning_rate=learning_rate)

        lr_schedules = get_lr_schedule(
            lr_decay_rate=lr_decay_rate,
            lr_decay_step=lr_decay_step,
            lr_decay_per_iter=lr_decay_per_iter,
            lr_decay_start_step=lr_decay_start_step,
            lr_decay_end_step=lr_decay_end_step,
            lr_warmup_init=lr_warmup_init,
            lr_warmup_final=lr_warmup_final,
            lr_warmup_start_step=lr_warmup_start_step,
            lr_warmup_end_step=lr_warmup_end_step,
            lr_warmup_per_iter=lr_warmup_per_iter)
        # Create a callback for updating the lr.
        callbacks = keras.callbacks.CallbackList(lr_schedules)
        callbacks.set_model(FakeModel(optimizer=optimizer))

        # Training loop.
        prev_loss = sys.float_info.max
        iter_below_tol = 0
        step = 0
        best_val = np.inf
        num_steps_no_improvement = 0
        if data_targets.dtype == np.float64:
            data_targets = data_targets.astype(np.float32)
        data_iterator = batch_iterator(data_inputs, data_targets, batch_size,
                                       self.shuffle_batch, allow_smaller_batch=False)
        num_samples = data_inputs.shape[0]
        num_batches_per_epoch = max(num_samples // batch_size, 1)
        checkpoint_saved = False
        last_saved_iter = -np.inf
        while True:
            # Evaluate callbacks.
            if step % num_batches_per_epoch == 0:
                callbacks.on_epoch_begin(step)
            callbacks.on_batch_begin(None, logs={'iter': step})

            # Extract the batch from the training data.
            batch_inputs, batch_targets = next(data_iterator)

            # Prepocess the inputs if need be.
            batch_inputs, batch_targets = self.process_inputs_targets(batch_inputs, batch_targets)

            # Compute loss and update gradients.
            with tf.GradientTape() as tape:
                pred_batch = self.model(batch_inputs, training=True)
                loss = self.loss(predictions=pred_batch, targets=batch_targets)

            grads = tape.gradient(loss, self.model.trainable_weights)
            if clip_gradient_norm:
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=clip_gradient_norm)
            optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            loss = loss.numpy()

            # Evaluate callbacks.
            if (step + 1) % num_batches_per_epoch == 0:
                callbacks.on_epoch_end(step)

            # Display logs per epoch step.
            step += 1
            if display_step and step % display_step == 0:
                lr = optimizer._lr if hasattr(optimizer, '_lr') else optimizer._learning_rate
                logger.info("Iteration: %4d  loss = %.9f, lr = %.8f" % (step, loss, lr))

            # Check if we have reached the desired loss tolerance.
            loss_diff = abs(prev_loss - loss)
            if loss_diff < abs_loss_chg_tol or abs(loss_diff / prev_loss) < rel_loss_chg_tol:
                iter_below_tol += 1
            else:
                iter_below_tol = 0

            if iter_below_tol >= loss_chg_iter_below_tol:
                if self.logging_level > 0:
                    logger.info('Loss value converged.')
                break
            if max_iter is not None and step >= max_iter:
                if self.logging_level > 0:
                    logger.info('Maximum number of iterations reached.')
                break
            # Check if the loss has not improved in the last num_steps_no_improvement steps.
            if loss < best_val:
                best_val = loss
                num_steps_no_improvement = 0
                # Save a checkpoint with the model at the best loss value.
                if step - last_saved_iter > self.checkpoint_step:
                    self.save_checkpoint()
                    checkpoint_saved = True
                    last_saved_iter = step
            else:
                num_steps_no_improvement += 1
            if num_steps_no_improvement > max_steps_no_improvement:
                logger.info('No loss improvement in the last %d steps.', num_steps_no_improvement)
                break

            prev_loss = loss

        if checkpoint_saved:
            self.load_checkpoint()
            loss = self.loss(predictions=pred_batch, targets=batch_targets)
            print('Restored loss: ', loss.numpy())

    def process_inputs_targets(self, batch_inputs, batch_targets=None):
        if batch_targets is None:
            return batch_inputs
        return batch_inputs, batch_targets

    def predict(self, data_inputs, training=False):
        return self.model(data_inputs, training=training)

    def loss(self, predictions, targets):
        return tf.nn.l2_loss(predictions, targets)

    @abc.abstractmethod
    def __str__(self):
        pass

    def save_checkpoint(self, filename='best_model.ckpt'):
        if self.checkpoints_dir:
            trainable_variables = {v.name: v for v in self.model.trainable_weights}
            checkpoint = tf.train.Checkpoint(**trainable_variables)
            path = os.path.join(self.checkpoints_dir, filename)
            checkpoint.save(file_prefix=path)
            logging.info('Saved checkpoint at: %s', path)

    def load_checkpoint(self, filename='best_model.ckpt'):
        trainable_variables = {v.name: v for v in self.model.trainable_weights}
        checkpoint = tf.train.Checkpoint(**trainable_variables)
        path = os.path.join(self.checkpoints_dir, filename)
        status = checkpoint.restore(tf.train.latest_checkpoint(self.checkpoints_dir))
        status.assert_consumed()
        status.assert_existing_objects_matched()
        logging.info('Loaded checkpoint from: %s', path)

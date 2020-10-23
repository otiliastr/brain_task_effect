import numpy as np
import tensorflow as tf


def get_lr_schedule(lr_decay_rate=None,
                    lr_decay_step=None,
                    lr_decay_per_iter=True,
                    lr_decay_start_step=0,
                    lr_decay_end_step=np.inf,
                    lr_warmup_init=1e-9,
                    lr_warmup_final=None,
                    lr_warmup_start_step=0,
                    lr_warmup_end_step=np.inf,
                    lr_warmup_per_iter=True):
    schedules = []

    if lr_warmup_final and lr_warmup_end_step is not None:
        lr_schedule = get_lr_warmup_schedule(
            lr_final=lr_warmup_final,
            schedule_end_step=lr_warmup_end_step,
            schedule_start_step=lr_warmup_start_step,
            lr_init=lr_warmup_init)
        lr_scheduler = LearningRateScheduler(
            lr_schedule,
            iteration_based=lr_warmup_per_iter,
            verbose=False)
        schedules.append(lr_scheduler)

    if lr_decay_rate is not None and lr_decay_step is not None:
        lr_schedule = get_lr_decay_schedule(
            lr_decay_step=lr_decay_step,
            lr_decay_rate=lr_decay_rate,
            schedule_start_step=lr_decay_start_step,
            schedule_end_step=lr_decay_end_step)
        lr_scheduler = LearningRateScheduler(
            lr_schedule,
            iteration_based=lr_decay_per_iter,
            verbose=False)
        schedules.append(lr_scheduler)

    return schedules


def get_lr_decay_schedule(lr_decay_step, lr_decay_rate=0.9, schedule_start_step=0,
                          schedule_end_step=np.inf):
    """Creates a learning rate decay schedule.

    Learning rate is reduced every 10 epochs to decay * original value.
    Arguments:
        lr_decay_step: An integer representing the number of steps after which to decay learning
            rate (i.e. the learning rate is decayed every `lr_decay_step` steps.
        lr_decay_rate: A float representing the factor with which to multiply the current learning
            rate. Default: 0.9.
        schedule_start_step: An integer representing the step (either iteration or epoch) after
            which to start applying the decay.
        schedule_end_step: An integer representing the step (either iteration or epoch) after
            which we stop applying the decay.

    Returns
        lr: A float representing the learning rate.
    """
    if schedule_end_step is None:
        schedule_end_step = np.inf
    if schedule_start_step is None:
        schedule_start_step = 0
    assert schedule_start_step <= schedule_end_step

    def lr_schedule(step, lr):
        if schedule_start_step <= step <= schedule_end_step and (step + 1) % lr_decay_step == 0:
            lr = lr * lr_decay_rate
        # print('Step ', step, '      lr=', lr)
        return lr

    return lr_schedule


def get_lr_warmup_schedule(lr_final, schedule_end_step, schedule_start_step=0, lr_init=1e-10):
    """Creates a learning rate warm-up schedule.

    The learning rate is increased linearly during the steps.

    Arguments:
        lr_final: A float representing the value of the learning rate at the end of the warmup.
        schedule_start_step: An integer representing the step (either iteration or epoch) after
            which to start applying the warmupi.
        schedule_end_step: An integer representing the step (either iteration or epoch) after
            which we stop applying the warmup, and the learning rate reaches its final value
            `lr_final`.
         lr_init: A float representing the value of the learning rate at the beginning of the
            warm-up.
    Returns
        lr: A float representing the learning rate.
    """
    total_steps = schedule_end_step - schedule_start_step
    total_lr_diff = lr_final - lr_init

    def lr_schedule(step, lr):
        if schedule_start_step <= step < schedule_end_step:
            lr = lr_init + total_lr_diff * step / total_steps
            # print('Step ', step, '      lr=', lr)

        return lr

    return lr_schedule


class LearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler.

    Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0, model=None, iteration_based=True):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.model = model
        self.iteration_based = iteration_based

    def _update_lr(self, step):
        lr = self.model.optimizer._lr if hasattr(self.model.optimizer, '_lr') else \
            self.model.optimizer._learning_rate
        try:  # new API
            lr = self.schedule(step, lr)
        except TypeError:  # old API for backward compatibility
            lr = self.schedule(step)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be float.')
        if hasattr(self.model.optimizer, '_lr'):
            self.model.optimizer._lr = lr
        else:
            self.model.optimizer._learning_rate = lr
        if self.verbose > 0:
            print('\nStep %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (step + 1, lr))

    def on_epoch_begin(self, epoch, logs=None, **kwargs):
        if not self.iteration_based:
            self._update_lr(epoch)

    def on_epoch_end(self, epoch, logs=None):
        if not self.iteration_based:
            logs = logs or {}
            logs['lr'] = self.model.optimizer._lr if hasattr(self.model.optimizer, '_lr') else \
                self.model.optimizer._learning_rate

    def on_batch_begin(self, batch, logs):
        if self.iteration_based:
            self._update_lr(logs['iter'])

    def on_batch_end(self, logs=None, **kwargs):
        if self.iteration_based:
            logs = logs or {}
            logs['lr'] = self.model.optimizer._lr if hasattr(self.model.optimizer, '_lr') else \
                self.model.optimizer._learning_rate


class FakeModel(object):
    def __init__(self, optimizer=None):
        self.optimizer = optimizer

"""Contains common routines used for training a model.

In practice, one should define a subclass inheriting from the class
`Train` by giving definitions to `get_data`, `decide_use_data`,
`compute`, `get_total_loss`, `get_summary_op` and `summary_log_info`.

Other methods may also be defined depending on different use cases.
See `TrainClassifyImagesCNN` for a training class that implements all
these methods and can be used directly.

** It seems that `tf.train.Supervisor` is deprecated and one should
use `tf.train.MonitoredSession` instead.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import time

import abc

import numpy as np
import tensorflow as tf

from nets_base.arg_scope import nets_arg_scope

slim = tf.contrib.slim


class TrainAbstract(object):
    """The interface/abstract class of the whole training framework.

    Note here we just put some methods that should be defined for
    training classes, for detailed implementation please refer to the
    class `Train` and its subclasses. The docstrings of subclasses
    sometimes also contain more information.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        """The main method that should be called when we train a model."""
        pass

    @abc.abstractmethod
    def train_step(self, sess, train_op, global_step, *args):
        """This is an auxiliary method that computes some `Tensors`
        (with `sess.run`) and prints logging information.
        Generally it's called at each step.

        Args:
            sess: The session used to compute the `Tensors`.
            train_op: A `Tensor` representing the training operation
                that should be called and returns the loss. This is the
                first element to be computed.
            global_step: A `Tensor` that gives the current gloabl step
                of the training process. This is the second element to
                be computed.
            *args: A list of supplementary `Tensors` to be runned.

        Returns:
            A list of values that are results of `sess.run` on given `Tensors`.
        """
        pass

    @abc.abstractmethod
    def get_data(self, tfrecord_dir, batch_size):
        """Read data from some directory and load them in batches.

        Args:
            tfrecord_dir: The directory where the tfrecords of the
                dataset are stored.
            batch_size: The number of elements contained in each batch.
        """
        pass

    @abc.abstractmethod
    def decide_used_data(self):
        """Decide the input data to be used for the main model.

        Normally while training the model we test also from time to time
        the performance of the model on a validation set to avoid
        overfitting. Training data and testing data are often read from
        different files and are represented by different tensors in the graph.

        Therefore to be consistent when doing the computations we should
        decide which data to use by using for example `tf.cond` with
        `self.is_training` as the first argument. `self.is_training` is a
        boolean `Tensor` that indicates whether we're in training or
        testing phrase.
        """
        pass

    @abc.abstractmethod
    def used_arg_scope(self, use_batch_norm, renorm):
        """The slim argument scope that is used for main computations.

        Args:
            use_batch_norm: Whether to do batch normalization or not.
            renorm: Whether to do batch renormalization or not. I've in
                fact never used it.

        Returns:
            An argument scope to be used for model computations.
        """
        pass

    @abc.abstractmethod
    def compute(self, **kwargs):
        """Compute necessary values of the model.

        The content of this function can vary a lot from case to case and
        we don't have some fixed arguments for this functions. For example,
        for classification normally it's used to compute the probabilities
        that an instance belongs to each class, and for auto-encoder it's
        used to compute the reconstruction of input.
        """
        pass

    @abc.abstractmethod
    def get_total_loss(self):
        """Get the loss of the model that needs to be minimized.

        Use `tf.losses` interface and one must call
        `tf.losses.get_total_loss` at the end otherwise the
        regularization loss is not included.

        Returns:
            The total loss `Tensor`.
        """
        pass

    @abc.abstractmethod
    def get_learning_rate(self):
        """Returns the learning rate used by the optimizer."""
        pass

    @abc.abstractmethod
    def get_optimizer(self):
        """Returns the optimizer used to minimize the loss."""
        pass

    @abc.abstractmethod
    def get_supervisor(self, log_dir, init_fn):
        """Get the supervisor used to monitor training.

        Args:
            log_dir: The directory to log event files and checkpoints.
            init_fn: The initialization function to be called before
                starting training. Ex, restoring parameters from a
                pre-trained model.

        Returns:
            A supervisor to monitor the training process.
        """
        pass

    @abc.abstractmethod
    def step_log_info(self, sess):
        """Things to be done at every step.

        Normally we call the `self.train_step` method and may print
        some extra logging information.

        Args:
            sess: The session to use for computations.
        """
        pass

    @abc.abstractmethod
    def summary_log_info(self, sess):
        """Things to be done for steps that store summaries.

        Normally we call the `self.train_step` method, print extra
        logging information and compute and save summaries to the
        event files.

        Args:
            sess: The session to use for computations.
        """
        pass

    @property
    def default_trainable_scopes(self):
        """Define default trainable scopes to be used (see `self.train`)."""
        return None

    def get_summary_op(self):
        """Returns the `Tensor` grouping summary operations for training."""
        pass

    def get_test_summary_op(self):
        """Returns the `Tensor` grouping summary operations for test"""
        pass

    def get_metric_op(self):
        """Returns the `Tensor` grouping metric operations"""
        pass

    def get_init_fn(self, checkpoint_dirs):
        """Get a function to warm-start the training."""
        return None

    def extra_initialization(self, sess):
        """Extra initialization to be done once the session is started.

        In some particular cases, initializations need also to be done
        after session and queue runners are started (for example to
        pre-compute some representations). Then `self.get_init_fn`
        may not be totally appropriate. Note that in this case one cannot
        use the default `ready_op` of `tf.train.Supervisor`.

        Args:
            sess: The session used to run initialization.
        """
        pass

    def test_log_info(self, sess, test_use_batch):
        """Print logging information and compute summaries for test.

        Args:
            sess: The session in which the computations are done.
            test_use_batch: Whether to use batch statistics or moving
                ones for batch normalization during test. Normally
                this should be `False` but if the probability distribution
                in the test domain is very different from the training
                domain and in this should better be turned to `True`.
                Note that unfortunately `slim.dropout` is also affected.
        """
        pass

    def final_log_info(self, sess):
        """Print logging information and compute summaries for the end.

        Args:
            sess: The session to use.
        """
        pass


class Train(TrainAbstract):
    """Implementation of the interface `TrainAbstract`.

    In practice, one should define a subclass inheriting from this class
    and implement `get_data`, `decide_use_data`, `compute`, `get_total_loss`,
    `get_summary_op` and `summary_log_info`.

    Other methods like `get_test_summary_op`, `test_log_info`, `get_init_fn`,
    `get_metric_op` and `extra_initialization` may also be implemented
    depending on different use cases.

    Of cource the methods already defined here like `step_log_info` and
    `get_learning_rate` can also be overrided.

    See `TrainClassifyImagesCNN` for a training class that implements all
    these methods and can be used directly.
    """

    # Parameters used by `self.get_learning_rate`.
    initial_learning_rate = 0.005
    lr_decay_steps = 100
    lr_decay_rate = 0.8

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def train(self,
              tfrecord_dir,
              checkpoint_dirs,
              log_dir,
              number_of_steps=None,
              number_of_epochs=1,
              batch_size=24,
              save_summaries_steps=5,
              save_model_steps=250,
              do_test=True,
              trainable_scopes=None,
              use_default_trainable_scopes=True,
              use_batch_norm=True,
              renorm=False,
              test_use_batch=False,
              **kwargs):
        """Train the model.

        Args:
            tfrecord_dir: The directory that contains the dataset tfreocrds
                (which can be generated by `convert_TFrecord` scripts).
            checkpoints_dir: The directorys containing checkpoints of
                models to be used if any; use `None` otherwise.
            log_dir: The directory to log event files and checkpoints.
            number_of_steps: number of steps to run the training process
                (one step = one batch), if `None` then `number_of_epochs`
                is used.
            number_of_epochs: Number of epochs to run through the whole
                dataset.
            batch_size: The batch size used to train and test (if so).
            save_summaries_steps: We save the summary every
                `save_summaries_steps`.
            save_model_steps: We save the model to a checkpoint every
                `save_model_steps`.
            do_test: If `True` we test the performance on the validation
                set every `save_summaries_steps` results are also
                shown on tensorboard.
            trainable_scopes: The layers to train, if left `None` then we
                may use the default trainable scopes depending on the
                value of `use_default_trainable_scopes`.
            use_default_trainable_scopes: When `trainable_scopes` is `None`,
                if this is `True` then we use the default trainable scopes
                defined in each class, and if this is `False` we train all
                the weights found in the model.
            use_batch_norm, renorm: Passed to `self.used_arg_scope` to decide
                whether to use batch normalization/renormalization.
            test_use_batch: Decide whether to use batch statistics or
                moving ones for batch normalization during tests.
            **kwargs: Arguments pass to the `self.compute`.
        """
        # Create the log directory if it doesn't exist
        if not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)

        if (checkpoint_dirs is not None and
                not isinstance(checkpoint_dirs, (list, tuple))):
            checkpoint_dirs = [checkpoint_dirs]

        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)

            # Read the data
            with tf.name_scope('Data_provider'):
                dataset = self.get_data(tfrecord_dir, batch_size)

            # Use `number_of_epochs` when `number_of_steps` is not given
            if number_of_steps is None:
                number_of_steps = int(np.ceil(
                    dataset.num_samples * number_of_epochs / batch_size))

            # Decide if we're training or not to use the right data
            self.training = tf.placeholder(tf.bool, shape=(), name='training')
            self.decide_used_data()

            # Decide if we use batch statstics or moving mean/variance for
            # batch normalization during the test (see `self.test_log_info`)
            self.batch_stat = tf.placeholder(
                tf.bool, shape=(), name='batch_stat')

            # Create the model, use the default arg scope to configure the
            # batch norm parameters
            with slim.arg_scope(self.used_arg_scope(use_batch_norm, renorm)):
                self.compute(**kwargs)

            # Specify the loss function
            # Create the global step for monitoring training
            # Specify the learning rate and optimizer
            total_loss = self.get_total_loss()
            self.global_step = tf.train.get_or_create_global_step()
            self.get_learning_rate()
            optimizer = self.get_optimizer()

            # Decide what variables need to be trained
            if trainable_scopes is None:
                if (not use_default_trainable_scopes or
                        self.default_trainable_scopes is None):
                    variables_to_train = tf.trainable_variables()
                else:
                    variables_to_train = self.get_variables_to_train(
                        self.default_trainable_scopes)
            else:
                variables_to_train = \
                    self.get_variables_to_train(trainable_scopes)
            print(variables_to_train)

            # Create the training operation
            self.train_op = slim.learning.create_train_op(
                total_loss, optimizer,
                variables_to_train=variables_to_train)

            # The metrics to predict (may be omitted)
            self.get_metric_op()

            # Create some summaries to visualize the training process
            self.get_summary_op()
            self.get_test_summary_op()

            # Define the initialization function used by the supervisor
            if checkpoint_dirs is None:
                init_fn = None
            else:
                init_fn = self.get_init_fn(checkpoint_dirs)

            # Define the supervisor
            self.sv = self.get_supervisor(log_dir, init_fn)

            with self.sv.managed_session() as sess:

                # Finalize the initialization if necessary
                self.extra_initialization(sess)

                # Run the training process
                for step in xrange(number_of_steps):

                    # Save summries from time to time
                    if (step+1) % save_summaries_steps == 0:
                        self.summary_log_info(sess)
                        if do_test:
                            self.test_log_info(sess, test_use_batch)
                    else:
                        self.step_log_info(sess)

                    # Save the model from time to time
                    if (step+1) % save_model_steps == 0:
                        self.sv.saver.save(
                            sess, self.sv.save_path,
                            global_step=self.sv.global_step)

                # Finish training and save model to checkpoint
                self.final_log_info(sess)
                self.sv.saver.save(
                    sess, self.sv.save_path, global_step=self.sv.global_step)

    def train_step(self, sess, train_op, global_step, *args):
        """Run `Tensors` and print logging information.

        Args:
            sess: The session used to compute the `Tensors`.
            train_op: A `Tensor` representing the training operation
                that should be called and returns the loss. This is the
                first element to be computed.
            global_step: A `Tensor` that gives the current gloabl step
                of the training process. This is the second element to
                be computed.
            *args: A list of supplementary `Tensors` to be runned.

        Returns:
            A list of values that are results of `sess.run` on given `Tensors`.
        """
        tensors_to_run = [train_op, global_step]
        tensors_to_run.extend(args)

        start_time = time.time()
        tensor_values = sess.run(
            tensors_to_run,
            feed_dict={self.training: True, self.batch_stat: True})
        time_elapsed = time.time() - start_time

        self.loss = tensor_values[0]
        global_step_count = tensor_values[1]

        tf.logging.info(
            'global step %s: loss: %.4f (%.2f sec/step)',
            global_step_count, self.loss, time_elapsed)
        return tensor_values

    def used_arg_scope(self, use_batch_norm, renorm):
        """The slim argument scope that is used for main computations.

        In my experiences normally it includes batch normalization, proper
        initialization and weight regularization. `self.batch_stat` is
        particularly used to indicate whether to use moving mean/variance
        or batch statistics for batch normalization. For training we must
        use batch statistics so it's always `True`. For testing this is
        often `False` but sometimes the probability distribution in the test
        domain can be very different from the training domain and in this
        case this should better be `True` to guarantee a sensible performance.
        However, dropouts are then also affected (in `slim.dropout` dropouts
        are applied when `self.batch_stat` is `True`).

        Args:
            use_batch_norm: Whether to do batch normalization or not.
            renorm: Whether to do batch renormalization or not. I've in
                fact never used it.

        Returns:
            An argument scope to be used for model computations.
        """
        return nets_arg_scope(
            is_training=self.batch_stat,
            use_batch_norm=use_batch_norm,
            renorm=renorm)

    def get_learning_rate(self):
        """Use an exponentially decaying learning rate"""
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=self.initial_learning_rate,
            global_step=self.global_step,
            decay_steps=self.lr_decay_steps,
            decay_rate=self.lr_decay_rate, staircase=True)
        return self.learning_rate

    def get_optimizer(self):
        """Define an Adam optimizer for learning."""
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def get_supervisor(self, log_dir, init_fn):
        """Get the supervisor used to monitor training.

        Args:
            log_dir: The directory to log event files and checkpoints.
            init_fn: The initialization function to be called before
                starting training. Ex, restoring parameters from a
                pre-trained model.

        Returns:
            A supervisor to monitor the training process.
        """
        return tf.train.Supervisor(
            logdir=log_dir, summary_op=None,
            init_fn=init_fn, save_model_secs=0)

    def step_log_info(self, sess):
        """Things to be done at every step.

        Here we simply call the `self.train_step` with adequate arguments.

        Args:
            sess: The session to use for computations.
        """
        if hasattr(self, 'metric_op'):
            self.loss = self.train_step(
                sess, self.train_op, self.sv.global_step, self.metric_op)[0]
        else:
            self.loss = self.train_step(
                sess, self.train_op, self.sv.global_step)[0]

    def final_log_info(self, sess):
        """Print logging information and compute summaries for the end.

        Args:
            sess: Not used here, just for interface consistency.
        """
        tf.logging.info('Finished training. Final Loss: %s', self.loss)
        tf.logging.info('Saving model to disk now.')

    @staticmethod
    def get_variables_to_train(scopes):
        """An auxiliary method to easily get variables we want to train.

        Args:
            scopes: A list of strings as scope names.

        Returns:
            All the variables of the model that are contained in some
            scope of `scopes`.
        """
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train

    @staticmethod
    def get_variables_to_restore(scopes=None, exclude=None):
        """An auxiliary method to easily get variables we want to restore.

        Args:
            scopes: A list of strings as scope names to include.
            exclude: A list of strings as scope names to exclude.

        Returns:
            All the variables of the model that are contained in some
            scope of `scopes` but not in any scope of `exclude`.
            If `scopes` is `None` we include all the variables by default
            and apply `exclude` after, and if `exclude` is `None` we
            don't exclude any variables in particular and use only `scopes`.
        """
        if scopes is not None:
            variables_to_restore = []
            for scope in scopes:
                variables = tf.get_collection(
                    tf.GraphKeys.MODEL_VARIABLES, scope)
                variables_to_restore.extend(variables)
        else:
            variables_to_restore = tf.model_variables()

        if exclude is not None:
            variables_to_restore_final = []
            for var in variables_to_restore:
                excluded = False
                for exclusion in exclude:
                    if var.op.name.startswith(exclusion):
                        excluded = True
                        break
                if not excluded:
                    variables_to_restore_final.append(var)
        else:
            variables_to_restore_final = variables_to_restore

        return variables_to_restore_final

    def get_batch_norm_summary(self):
        """Track moving mean&variance of the last layer on Tensorboard."""
        try:
            last_moving_mean = [
                v for v in tf.model_variables()
                if v.op.name.endswith('moving_mean')][0]
            last_moving_variance = [
                v for v in tf.model_variables()
                if v.op.name.endswith('moving_variance')][0]
            tf.summary.histogram('batch_norm/last_layer/moving_mean',
                                 last_moving_mean)
            tf.summary.histogram('batch_norm/last_layer/moving_variance',
                                 last_moving_variance)
        except IndexError:
            tf.logging.info('No moiving mean or variance')


# Only for convenience
def train(train_class,
          used_structure,
          tfrecord_dir,
          checkpoint_dirs,
          log_dir,
          number_of_steps=None,
          **kwargs):
    train_instance = train_class(used_structure)
    for key in kwargs.copy():
        if hasattr(train_instance, key):
            setattr(train_instance, key, kwargs[key])
            del kwargs[key]
    train_instance.train(
        tfrecord_dir, checkpoint_dirs, log_dir,
        number_of_steps=number_of_steps, **kwargs)

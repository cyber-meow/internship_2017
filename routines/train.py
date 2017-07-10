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

    __meta_class__ = abc.ABCMeta

    @property
    def default_trainable_scopes(self):
        return None

    @abc.abstractmethod
    def train(self, *args):
        pass

    @abc.abstractmethod
    def train_step(self, sess, train_op, global_step, *args):
        pass

    @abc.abstractmethod
    def get_data(self, tfrecord_dir, batch_size):
        pass

    @abc.abstractmethod
    def decide_used_data(self):
        pass

    @abc.abstractmethod
    def compute(self, *args):
        pass

    @abc.abstractmethod
    def get_total_loss(self):
        pass

    @abc.abstractmethod
    def get_learning_rate(self):
        pass

    @abc.abstractmethod
    def get_optimizer(self):
        pass

    @abc.abstractmethod
    def get_metric_op(self):
        pass

    def get_test_metrics(self):
        pass

    @abc.abstractmethod
    def get_summary_op(self):
        pass

    def get_test_summary_op(self):
        pass

    def extra_log_info(self):
        pass

    @abc.abstractmethod
    def normal_log_info(self, sess):
        pass

    def test_log_info(self, sess):
        pass

    @abc.abstractmethod
    def final_log_info(self, sess):
        pass


class Train(TrainAbstract):

    def __init__(self,
                 initial_learning_rate=0.005,
                 lr_decay_steps=100,
                 lr_decay_rate=0.8):
        self.initial_learning_rate = initial_learning_rate
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate

    def train(self,
              tfrecord_dir,
              checkpoint_dirs,
              log_dir,
              number_of_steps=None,
              number_of_epochs=5,
              batch_size=24,
              save_summaries_steps=5,
              do_test=True,
              trainable_scopes=None,
              **kwargs):
        """Fine tune a pre-trained model using customized dataset.

        Args:
            tfrecord_dir: The directory that contains the tfreocrd files
              (which can be generated by data/convert_TFrecord.py)
            checkpoints_dir: The directory containing the checkpoint of
              the model to use
            log_dir: The directory to log event files and checkpoints
            number_of_steps: number of steps to run the training process
              (one step = one batch), if is None then number_of_epochs is used
            number_of_epochs: Number of epochs to run through the whole dataset
            batch_size: The batch size used to train and test (if any)
              save_summaries_steps: We save the summary every
              save_summaries_steps
            do_test: If True the test is done every save_summaries_steps and
              is shown on tensorboard
            trainable_scopes: The layers to train, if left None then all
            **kwargs: Arguments pass to the main structure/function
        """
        if not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)

        if (checkpoint_dirs is not None and
                not isinstance(checkpoint_dirs, (list, tuple))):
            checkpoint_dirs = [checkpoint_dirs]

        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)

            with tf.name_scope('Data_provider'):
                dataset = self.get_data(tfrecord_dir, batch_size)

            if number_of_steps is None:
                number_of_steps = int(np.ceil(
                    dataset.num_samples * number_of_epochs / batch_size))

            # Decide if we're training or not
            self.training = tf.placeholder(tf.bool, shape=(), name='training')
            self.decide_used_data()

            # Create the model, use the default arg scope to configure the
            # batch norm parameters
            with slim.arg_scope(nets_arg_scope(is_training=self.training)):
                self.compute(**kwargs)

            # Specify the loss function
            # Create the global step for monitoring training
            # Specify the learning rate, optimizer and train op
            total_loss = self.get_total_loss()
            self.global_step = tf.train.get_or_create_global_step()
            self.get_learning_rate()
            optimizer = self.get_optimizer()

            if trainable_scopes is None:
                if self.default_trainable_scopes is None:
                    variables_to_train = tf.trainable_variables()
                else:
                    variables_to_train = self.get_variables_to_train(
                        self.default_trainable_scopes)
            else:
                variables_to_train = \
                    self.get_variables_to_train(trainable_scopes)

            self.train_op = slim.learning.create_train_op(
                total_loss, optimizer,
                variables_to_train=variables_to_train)

            # The metrics to predict
            self.get_metric_op()
            self.get_test_metrics()

            # Create some summaries to visualize the training process:
            self.get_summary_op()
            self.get_test_summary_op()

            # Define the supervisor
            self.sv = tf.train.Supervisor(
                logdir=log_dir, summary_op=None,
                init_fn=self.get_init_fn(checkpoint_dirs))

            with self.sv.managed_session() as sess:
                self.extra_log_info()
                for step in xrange(number_of_steps):
                    if (step+1) % save_summaries_steps == 0:
                        summaries = self.normal_log_info(sess)
                        self.sv.summary_computed(sess, summaries)
                        if do_test:
                            summaries_test = self.test_log_info(sess)
                            if summaries_test is not None:
                                self.sv.summary_computed(sess, summaries_test)
                    elif self.metric_op is not None:
                        self.loss = self.train_step(
                            sess, self.train_op, self.sv.global_step,
                            self.metric_op)[0]
                    else:
                        self.loss = self.train_step(
                            sess, self.train_op, self.sv.global_step)[0]
                self.final_log_info(sess)
                self.sv.saver.save(sess, self.sv.save_path,
                                   global_step=self.sv.global_step)

    def train_step(self, sess, train_op, global_step, *args):
        tensors_to_run = [train_op, global_step]
        tensors_to_run.extend(args)

        start_time = time.time()
        tensor_values = sess.run(tensors_to_run,
                                 feed_dict={self.training: True})
        time_elapsed = time.time() - start_time

        total_loss = tensor_values[0]
        global_step_count = tensor_values[1]

        tf.logging.info(
            'global step %s: loss: %.4f (%.2f sec/step)',
            global_step_count, total_loss, time_elapsed)
        return tensor_values

    @staticmethod
    def get_variables_to_train(scopes):
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train

    def get_learning_rate(self):
        # Exponentially decaying learning rate
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=self.initial_learning_rate,
            global_step=self.global_step,
            decay_steps=self.lr_decay_steps,
            decay_rate=self.lr_decay_rate, staircase=True)
        return self.learning_rate

    def get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def get_batch_norm_summary(self):
        # Track moving mean and moving varaince
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
            tf.info.logging('No moiving mean or variance')
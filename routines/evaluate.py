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


class EvaluateAbstract(object):

    __meta_class__ = abc.ABCMeta

    @abc.abstractmethod
    def used_arg_scope(self, use_batch_norm):
        pass

    @abc.abstractmethod
    def evaluate(self, *args):
        pass

    @abc.abstractmethod
    def get_data(self, split_name, tfrecord_dir, batch_size):
        pass

    @abc.abstractmethod
    def compute(self, *args):
        pass

    def compute_log_data(self):
        pass

    @abc.abstractmethod
    def init_model(self, sess, checkpoint_dirs):
        pass

    @abc.abstractmethod
    def step_log_info(self, sess):
        pass

    @abc.abstractmethod
    def last_step_log_info(self, sess, batch_size):
        pass


class Evaluate(EvaluateAbstract):

    def evaluate(self,
                 tfrecord_dir,
                 checkpoint_dirs,
                 log_dir=None,
                 number_of_steps=None,
                 batch_size=12,
                 split_name='validation',
                 use_batch_norm=True,
                 **kwargs):

        if log_dir is not None and not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)

        if not isinstance(checkpoint_dirs, (tuple, list)):
            checkpoint_dirs = [checkpoint_dirs]

        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)

            with tf.name_scope('Data_provider'):
                dataset = self.get_data(split_name, tfrecord_dir, batch_size)

            if number_of_steps is None:
                number_of_steps = int(np.ceil(dataset.num_samples/batch_size))

            with slim.arg_scope(self.used_arg_scope(use_batch_norm)):
                self.compute(**kwargs)

            self.compute_log_data()

            # Define global step to be show in tensorboard
            global_step = tf.train.get_or_create_global_step()
            self.global_step_op = tf.assign(global_step, global_step+1)

            # File writer for the tensorboard
            if log_dir is not None:
                self.fw = tf.summary.FileWriter(log_dir)

            with tf.Session() as sess:
                with slim.queues.QueueRunners(sess):
                    sess.run(tf.variables_initializer([global_step]))
                    self.init_model(sess, checkpoint_dirs)

                    for step in xrange(number_of_steps-1):
                        global_step_count, summaries = self.step_log_info(sess)
                        if summaries is not None and log_dir is not None:
                            self.fw.add_summary(
                                summaries, global_step=global_step_count)
                    global_step_count, summaries_last = \
                        self.last_step_log_info(sess, batch_size)
                    if summaries_last is not None and log_dir is not None:
                        self.fw.add_summary(
                            summaries_last, global_step=global_step_count)
                    tf.logging.info('Finished evaluation')

    def used_arg_scope(self, use_batch_norm):
        return nets_arg_scope(
            is_traing=False, use_batch_norm=use_batch_norm)

    def eval_step(self, sess, global_step, *args):
        tensors_to_run = [global_step]
        tensors_to_run.extend(args)
        start_time = time.time()
        tensor_values = sess.run(tensors_to_run)
        time_elapsed = time.time() - start_time
        global_step_count = tensor_values[0]
        return global_step_count, time_elapsed, tensor_values[1:]

    def last_step_log_info(self, sess, batch_size):
        return self.step_log_info(sess)

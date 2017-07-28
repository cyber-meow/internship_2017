from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import abc

import numpy as np
import tensorflow as tf

from data.color_depth import load_batch_color_depth, get_split_color_depth
from nets_base.arg_scope import nets_arg_scope

slim = tf.contrib.slim


class EvaluateAbstract(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def evaluate(self, *args):
        pass

    @abc.abstractmethod
    def get_data(self, split_name, tfrecord_dir, batch_size):
        pass

    @abc.abstractmethod
    def used_arg_scope(self, batch_stat, use_batch_norm):
        pass

    @abc.abstractmethod
    def compute(self, *args):
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

    def compute_log_data(self):
        pass


class Evaluate(EvaluateAbstract):

    def evaluate(self,
                 tfrecord_dir,
                 checkpoint_dirs,
                 log_dir=None,
                 number_of_steps=None,
                 batch_size=None,
                 split_name='validation',
                 shuffle=False,
                 use_batch_norm=True,
                 batch_stat=False,
                 **kwargs):

        if log_dir is not None and not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)

        if not isinstance(checkpoint_dirs, (tuple, list)):
            checkpoint_dirs = [checkpoint_dirs]

        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)

            with tf.name_scope('Data_provider'):
                dataset = self.get_data(
                    split_name, tfrecord_dir, batch_size, shuffle)

            if batch_size is None:
                batch_size = dataset.num_samples
            if number_of_steps is None:
                number_of_steps = int(np.ceil(dataset.num_samples/batch_size))

            with slim.arg_scope(self.used_arg_scope(
                    batch_stat, use_batch_norm)):
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
                        self.step_log_info(sess)
                    self.last_step_log_info(sess, batch_size)
                    tf.logging.info('Finished evaluation')

    def used_arg_scope(self, batch_stat, use_batch_norm):
        return nets_arg_scope(
            is_training=batch_stat, use_batch_norm=use_batch_norm)

    def last_step_log_info(self, sess, batch_size):
        return self.step_log_info(sess)

    def init_model(self, sess, checkpoint_dirs):
        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        saver = tf.train.Saver(tf.model_variables())
        saver.restore(sess, checkpoint_path)


class EvaluateColorDepth(Evaluate):

    def __init__(self, image_size=299, color_channels=3, depth_channels=3):
        self.image_size = image_size
        self.color_channels = color_channels
        self.depth_channels = depth_channels

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_color_depth(
            split_name,
            tfrecord_dir,
            color_channels=self.color_channels,
            depth_channels=self.depth_channels)
        self.images_color, self.images_depth, self.labels = \
            load_batch_color_depth(
                self.dataset, height=self.image_size,
                width=self.image_size, batch_size=batch_size, shuffle=shuffle)
        return self.dataset


def evaluate(evaluate_class,
             used_structure,
             tfrecord_dir,
             checkpoint_dirs,
             log_dir,
             number_of_steps=None,
             **kwargs):
    evaluate_instance = evaluate_class(used_structure)
    for key in kwargs.copy():
        if hasattr(evaluate_instance, key):
            setattr(evaluate_instance, key, kwargs[key])
            del kwargs[key]
    evaluate_instance.evaluate(
        tfrecord_dir, checkpoint_dirs, log_dir,
        number_of_steps=number_of_steps, **kwargs)

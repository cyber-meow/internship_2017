"""Implement a shadow CAE that reads raw image as input"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from data.color_depth.read_TFRecord import get_split_color_depth
from data.color_depth.load_batch import load_batch_color_depth
from routines.train import Train

slim = tf.contrib.slim


class TrainFusionAE(Train):

    def __init__(self, structure, image_size=299,
                 initial_learning_rate=0.01, **kwargs):
        super(TrainFusionAE, self).__init__(**kwargs)
        self.initial_learning_rate = initial_learning_rate
        self.image_size = image_size
        self.structure = structure

    @property
    def default_trainable_scopes(self):
        return ['Fusion', 'Seperation']

    def get_data(self, tfrecord_dir, batch_size):
        self.dataset_train = get_split_color_depth('train', tfrecord_dir)
        self.images_color_train, self.images_depth_train, _ = \
            load_batch_color_depth(
                self.dataset_train, height=self.image_size,
                width=self.image_size, batch_size=batch_size)
        self.dataset_test = get_split_color_depth('validation', tfrecord_dir)
        self.images_color_test, self.images_depth_test, _ = \
            load_batch_color_depth(
                self.dataset_test, height=self.image_size,
                width=self.image_size, batch_size=batch_size)
        return self.dataset_train

    def decide_used_data(self):
        self.images_color_orignial = tf.cond(
            self.training, lambda: self.images_color_train,
            lambda: self.images_color_test)
        self.images_depth_original = tf.cond(
            self.training, lambda: self.images_depth_train,
            lambda: self.images_depth_test)

    def compute(self, **kwargs):
        self.compute_reconstruction(
            self.images_color_original, self.images_depth_original, **kwargs)

    def compute_reconstruction(self, color_inputs, depth_inputs,
                               dropout_position='fc'):
        color_keep_prob = np.random.random()
        depth_keep_prob = 1 - color_keep_prob

        images_color_corrupted = slim.dropout(
            color_inputs, keep_prob=color_keep_prob,
            scope='Color/Input/Dropout')
        images_depth_corrupted = slim.dropout(
            depth_inputs, keep_prob=depth_keep_prob,
            scope='Depth/Input/Dropout')

        assert dropout_position in ['fc', 'input']
        dropout_input = dropout_position == 'input'

        if dropout_input:
            self.images_color = images_color_corrupted
            self.images_depth = images_depth_corrupted
        else:
            self.images_color = color_inputs
            self.images_depth = depth_inputs

        if dropout_input:
            color_keep_prob = depth_keep_prob = 1

        (self.reconstructions_color, self.reconstructions_depth),  _ = \
            self.structure(
                self.images_color, self.images_depth,
                color_keep_prob=color_keep_prob,
                depth_keep_prob=depth_keep_prob)

    def get_total_loss(self):
        self.reconstruction_loss_color = tf.losses.mean_squared_error(
            self.reconstructions_color, self.images_color_original)
        self.reconstruction_loss_depth = tf.losses.mean_squared_error(
            self.reconstructions_depth, self.images_depth_original)
        self.total_loss = tf.losses.get_total_loss()
        return self.total_loss

    def get_metric_op(self):
        self.metric_op = None
        return self.metric_op

    def get_summary_op(self):
        tf.summary.scalar(
            'losses/reconstruction/color', self.reconstruction_loss_color)
        tf.summary.scalar(
            'losses/reconstruction/depth', self.reconstruction_loss_depth)
        tf.summary.scalar('losses/total', self.total_loss)
        tf.summary.image('input/color', self.images_color)
        tf.summary.image('input/depth', self.images_depth)
        tf.summary.image('reconstruction/color', self.reconstructions_color)
        tf.summary.image('reconstruction/depth', self.reconstructions_depth)
        self.summary_op = tf.summary.merge_all()

    def get_init_fn(self, checkpoint_dirs):
        """Returns a function run by the chief worker to
           warm-start the training."""
        checkpoint_dir_color, checkpoint_dir_depth = checkpoint_dirs

        variables_color = {}
        variables_depth = {}

        for var in tf.model_variables():
            if var.op.name.startswith('Color'):
                variables_color[var.op.name[6:]] = var
            if var.op.name.startswith('depth'):
                variables_depth[var.op.name[6:]] = var

        saver_color = tf.train.Saver(variables_color)
        saver_depth = tf.train.Saver(variables_depth)

        checkpoint_path_color = tf.train.latest_checkpoint(
            checkpoint_dir_color)
        checkpoint_path_depth = tf.train.latest_checkpoint(
            checkpoint_dir_depth)

        def restore(sess):
            saver_color.restore(sess, checkpoint_path_color)
            saver_depth.restore(sess, checkpoint_path_depth)
        return restore

    def normal_log_info(self, sess):
        self.loss, _, summaries = self.train_step(
            sess, self.train_op, self.sv.global_step, self.summary_op)
        return summaries

    def final_log_info(self, sess):
        tf.logging.info('Finished training. Final Loss: %s', self.loss)
        tf.logging.info('Saving model to disk now.')

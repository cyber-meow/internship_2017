"""Implement a shadow CAE that reads raw image as input"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets_base.arg_scope import nets_arg_scope
from routines.train import TrainColorDepth
from routines.evaluate import EvaluateImages, EvaluateColorDepth

slim = tf.contrib.slim


class TrainFusionAE(TrainColorDepth):

    # default_trainable_scopes = ['Fusion', 'Seperation']
    default_trainable_scopes = None

    def __init__(self, structure, **kwargs):
        super(TrainFusionAE, self).__init__(**kwargs)
        self.structure = structure

    def decide_used_data(self):
        self.images_color_original = tf.cond(
            self.training, lambda: self.images_color_train,
            lambda: self.images_color_test)
        self.images_depth_original = tf.cond(
            self.training, lambda: self.images_depth_train,
            lambda: self.images_depth_test)

    def compute(self, **kwargs):
        self.compute_reconstruction(
            self.images_color_original, self.images_depth_original, **kwargs)

    def compute_reconstruction(self, color_inputs, depth_inputs,
                               dropout_position='fc', threshold=0.15,
                               color_keep_prob=None):
        if color_keep_prob is None:
            color_keep_prob = tf.random_uniform([])
        else:
            color_keep_prob = tf.constant(color_keep_prob, tf.float32)
        color_keep_prob = tf.cond(
            color_keep_prob < tf.constant(threshold, tf.float32),
            lambda: tf.constant(0, tf.float32), lambda: color_keep_prob)
        color_keep_prob = tf.cond(
            color_keep_prob > tf.constant(1-threshold, tf.float32),
            lambda: tf.constant(1, tf.float32), lambda: color_keep_prob)
        depth_keep_prob = tf.constant(1, dtype=tf.float32) - color_keep_prob

        images_color_corrupted = tf.nn.dropout(
            color_inputs, keep_prob=color_keep_prob,
            name='Color/Input/Dropout')
        images_color_corrupted = tf.cond(
            tf.equal(color_keep_prob, tf.constant(0, tf.float32)),
            lambda: tf.zeros_like(color_inputs),
            lambda: images_color_corrupted)

        images_depth_corrupted = tf.nn.dropout(
            depth_inputs, keep_prob=depth_keep_prob,
            name='Depth/Input/Dropout')
        images_depth_corrupted = tf.cond(
            tf.equal(depth_keep_prob, tf.constant(0, tf.float32)),
            lambda: tf.zeros_like(depth_inputs),
            lambda: images_depth_corrupted)

        assert dropout_position in ['fc', 'input']
        dropout_input = dropout_position == 'input'

        if dropout_input:
            self.images_color = images_color_corrupted
            self.images_depth = images_depth_corrupted
        else:
            self.images_color = color_inputs
            self.images_depth = depth_inputs

        if dropout_input:
            color_keep_prob = depth_keep_prob = tf.constant(1, tf.float32)

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
        self.get_batch_norm_summary()
        tf.summary.scalar(
            'losses/reconstruction/color', self.reconstruction_loss_color)
        tf.summary.scalar(
            'losses/reconstruction/depth', self.reconstruction_loss_depth)
        tf.summary.scalar('losses/total', self.total_loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
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
            if var.op.name.startswith('Fusion/Color'):
                variables_color['CAE'+var.op.name[12:]] = var
            if var.op.name.startswith('Fusion/Depth'):
                variables_depth['CAE'+var.op.name[12:]] = var

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

    def used_arg_scope(self, use_batch_norm, renorm):
        return nets_arg_scope(is_training=self.training,
                              use_batch_norm=use_batch_norm,
                              renorm=renorm,
                              batch_norm_decay=0.99,
                              renorm_decay=0.99)


class EvaluateFusionAE(EvaluateColorDepth):

    def __init__(self, structure, **kwargs):
        super(EvaluateFusionAE, self).__init__(**kwargs)
        self.structure = structure

    def compute(self, **kwargs):
        self.reconstructions_color, self.reconstructions_depth = \
            self.compute_reconstruction(
                self.images_color, self.images_depth, **kwargs)

    def compute_reconstruction(self,
                               color_inputs,
                               depth_inputs,
                               color_keep_prob=0.5,
                               depth_keep_prob=None,
                               dropout_position='input'):
        if color_keep_prob is None:
            if depth_keep_prob is None:
                color_keep_prob = tf.random_uniform([])
            else:
                color_keep_prob = tf.constant(1-depth_keep_prob, tf.float32)
        else:
            color_keep_prob = tf.constant(color_keep_prob, tf.float32)
        if depth_keep_prob is None:
            depth_keep_prob = tf.constant(1, tf.float32) - color_keep_prob
        else:
            depth_keep_prob = tf.constant(depth_keep_prob, tf.float32)

        images_color_corrupted = tf.nn.dropout(
            color_inputs, keep_prob=color_keep_prob,
            name='Color/Input/Dropout')
        images_color_corrupted = tf.cond(
            tf.equal(color_keep_prob, tf.constant(0, tf.float32)),
            lambda: tf.zeros_like(color_inputs),
            lambda: images_color_corrupted)

        images_depth_corrupted = tf.nn.dropout(
            depth_inputs, keep_prob=depth_keep_prob,
            name='Depth/Input/Dropout')
        images_depth_corrupted = tf.cond(
            tf.equal(depth_keep_prob, tf.constant(0, tf.float32)),
            lambda: tf.zeros_like(depth_inputs),
            lambda: images_depth_corrupted)
        assert dropout_position in ['fc', 'input']
        dropout_input = dropout_position == 'input'

        if dropout_input:
            self.images_color = images_color_corrupted
            self.images_depth = images_depth_corrupted
        else:
            self.images_color = color_inputs
            self.images_depth = depth_inputs

        if dropout_input:
            color_keep_prob = depth_keep_prob = tf.constant(1, tf.float32)

        (reconstructions_color, reconstructions_depth), _ = \
            self.structure(self.images_color, self.images_depth,
                           color_keep_prob=color_keep_prob,
                           depth_keep_prob=depth_keep_prob)
        return reconstructions_color, reconstructions_depth

    def compute_log_data(self):
        tf.summary.image('input/color', self.images_color)
        tf.summary.image('input/depth', self.images_depth)
        tf.summary.image('reconstruction/color', self.reconstructions_color)
        tf.summary.image('reconstruction/depth', self.reconstructions_depth)
        self.summary_op = tf.summary.merge_all()

    def step_log_info(self, sess):
        self.global_step_count, time_elapsed, tensor_values = \
            self.eval_step(
                sess, self.global_step_op, self.summary_op)
        tf.logging.info(
            'global step %s: %.2f sec/step',
            self.global_step_count, time_elapsed)
        return self.global_step_count, tensor_values[0]

    def used_arg_scope(self, use_batch_norm):
        return nets_arg_scope(is_training=True, use_batch_norm=use_batch_norm)


class EvaluateFusionAESingle(EvaluateImages):

    def __init__(self, structure, **kwargs):
        super(EvaluateFusionAESingle, self).__init__(**kwargs)
        self.structure = structure

    def compute(self, **kwargs):
        self.reconstructions_color, self.reconstructions_depth = \
            self.compute_reconstruction(self.images, **kwargs)

    def compute_reconstruction(self, inputs, modality='color'):
        assert modality in ['color', 'depth']

        if modality == 'color':
            (reconstructions_color, reconstructions_depth), _ = \
                self.structure(inputs, tf.zeros_like(inputs),
                               color_keep_prob=tf.constant(1, tf.float32))
        elif modality == 'depth':
            (reconstructions_color, reconstructions_depth), _ = \
                self.structure(tf.zeros_like(inputs), inputs,
                               depth_keep_prob=tf.constant(1, tf.float32))

        return reconstructions_color, reconstructions_depth

    def compute_log_data(self):
        tf.summary.image('input', self.images)
        tf.summary.image('reconstruction/color', self.reconstructions_color)
        tf.summary.image('reconstruction/depth', self.reconstructions_depth)
        self.summary_op = tf.summary.merge_all()

    def step_log_info(self, sess):
        self.global_step_count, time_elapsed, tensor_values = \
            self.eval_step(
                sess, self.global_step_op, self.summary_op)
        tf.logging.info(
            'global step %s: %.2f sec/step',
            self.global_step_count, time_elapsed)
        return self.global_step_count, tensor_values[0]

    def used_arg_scope(self, use_batch_norm):
        return nets_arg_scope(is_training=True, use_batch_norm=use_batch_norm)

"""Classify images using intern representations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from multimodal.gesture.basics import TrainColorDepth, EvaluateColorDepth
from classify.train import TrainClassify
from classify.evaluate import EvaluateClassify
from images.classify_routines import TrainClassifyImages
from images.classify_routines import EvaluateClassifyImages

slim = tf.contrib.slim


class TrainClassifyCommonRepr(TrainClassifyImages):

    @property
    def default_trainable_scopes(self):
        return ['Logits']

    def __init__(self, architecture, **kwargs):
        super(TrainClassifyCommonRepr, self).__init__(**kwargs)
        self.architecture = architecture

    def compute_logits(self, inputs, num_classes,
                       modality='color', dropout_keep_prob=0.8):
        assert modality in ['color', 'depth']
        if modality == 'color':
            net = self.architecture(
                inputs, tf.zeros_like(inputs),
                final_endpoint='Middle',
                color_keep_prob=tf.constant(1, tf.float32))
        elif modality == 'depth':
            net = self.architecture(
                tf.zeros_like(inputs), inputs,
                final_endpoint='Middle',
                color_keep_prob=tf.constant(0, tf.float32))
        net = slim.dropout(net, dropout_keep_prob, scope='PreLogitsDropout')
        net = slim.flatten(net, scope='PreLogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits

    def get_init_fn(self, checkpoint_dirs):
        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        assert checkpoint_path is not None
        variables_to_restore = self.get_variables_to_restore(['Fusion'])
        return slim.assign_from_checkpoint_fn(
            checkpoint_path, variables_to_restore)


class EvaluateClassifyCommonRepr(EvaluateClassifyImages):

    def __init__(self, architecture, **kwargs):
        super(EvaluateClassifyCommonRepr, self).__init__(**kwargs)
        self.architecture = architecture

    def compute_logits(self, inputs, num_classes, modality='color'):
        assert modality in ['color', 'depth']
        if modality == 'color':
            net = self.architecture(
                inputs, tf.zeros_like(inputs),
                final_endpoint='Middle',
                color_keep_prob=tf.constant(1, tf.float32))
        elif modality == 'depth':
            net = self.architecture(
                tf.zeros_like(inputs), inputs,
                final_endpoint='Middle',
                color_keep_prob=tf.constant(0, tf.float32))
        net = slim.flatten(net, scope='PreLogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits


class TrainClassifyFusion(TrainColorDepth, TrainClassify):
    """
    To train a lineair classifier from a common representation
    that is learned by an auto-encoder, use endpoint='Middle'
    an trainables_scopes='Logits'
    """

    def __init__(self, architecture, **kwargs):
        super(TrainClassifyFusion, self).__init__(**kwargs)
        self.architecture = architecture

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.images_color, self.images_depth,
            self.dataset_train.num_classes, **kwargs)

    def compute_logits(self, color_inputs, depth_inputs, num_classes,
                       dropout_keep_prob=0.8, endpoint=None):
        """Use endpoint='Middle' for CAE architectures"""
        if endpoint is None:
            net = self.architecture(color_inputs, depth_inputs)
        else:
            net = self.architecture(
                color_inputs, depth_inputs, final_endpoint=endpoint)
        net = slim.dropout(net, dropout_keep_prob, scope='PreLogitsDropout')
        net = slim.flatten(net, scope='PrelogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits

    def get_summary_op(self):
        super(TrainClassifyFusion, self).get_summary_op()
        tf.summary.image('train/color', self.images_color)
        tf.summary.image('train/depth', self.images_depth)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        summary_op = super(TrainClassifyFusion, self).get_test_summary_op()
        imgs_color_test = tf.summary.image('test/color', self.images_color)
        imgs_depth_test = tf.summary.image('test/depth', self.images_depth)
        self.test_summary_op = tf.summary.merge(
            [summary_op, imgs_color_test, imgs_depth_test])
        return self.test_summary_op

    def get_init_fn(self, checkpoint_dirs):
        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        assert checkpoint_path is not None
        variables_to_restore = self.get_variables_to_restore(['Fusion'])
        return slim.assign_from_checkpoint_fn(
            checkpoint_path, variables_to_restore)


class EvaluateClassifyFusion(EvaluateColorDepth, EvaluateClassify):

    def __init__(self, architecture, **kwargs):
        super(EvaluateClassifyFusion, self).__init__(**kwargs)
        self.architecture = architecture

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.images_color, self.images_depth,
            self.dataset.num_classes, **kwargs)

    def compute_logits(self, color_inputs, depth_inputs,
                       num_classes, endpoint=None):
        if endpoint is None:
            net = self.architecture(color_inputs, depth_inputs)
        else:
            net = self.architecture(
                color_inputs, depth_inputs, final_endpoint=endpoint)
        net = slim.flatten(net, scope='PrelogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits


class TrainClassifyEmbedding(TrainClassifyImages):

    @property
    def default_trainable_scopes(self):
        return ['Logits']

    def __init__(self, architecture, **kwargs):
        super(TrainClassifyEmbedding, self).__init__(**kwargs)
        self.architecture = architecture

    def compute_logits(self, inputs, num_classes, endpoint='Middle',
                       modality='color', feature_length=512,
                       dropout_keep_prob=0.8, unit_normalization=True):
        assert modality in ['color', 'depth']
        scope_name = 'Color' if modality == 'color' else 'Depth'

        with tf.variable_scope(scope_name):
            net = self.architecture(
                inputs, final_endpoint=endpoint)
            net = slim.flatten(net)
        with tf.variable_scope('Embedding'):
            net = slim.fully_connected(net, feature_length, scope=scope_name)
        if unit_normalization:
            net = slim.unit_norm(net, 1)
        net = slim.dropout(net, dropout_keep_prob, scope='PreLogitsDropout')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits

    def get_init_fn(self, checkpoint_dirs):
        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        assert checkpoint_path is not None
        variables_to_restore = self.get_variables_to_restore(
            ['Embedding', 'Color', 'Depth'])
        return slim.assign_from_checkpoint_fn(
            checkpoint_path, variables_to_restore)


class EvaluateClassifyEmbedding(EvaluateClassifyImages):

    def __init__(self, architecture, **kwargs):
        super(EvaluateClassifyEmbedding, self).__init__(**kwargs)
        self.architecture = architecture

    def compute_logits(self, inputs, num_classes,
                       endpoint='Middle', modality='color',
                       feature_length=512,
                       unit_normalization=True):
        assert modality in ['color', 'depth']
        scope_name = 'Color' if modality == 'color' else 'Depth'

        with tf.variable_scope(scope_name):
            net = self.architecture(
                inputs, final_endpoint=endpoint)
            net = slim.flatten(net)
        with tf.variable_scope('Embedding'):
            net = slim.fully_connected(net, feature_length, scope=scope_name)
        if unit_normalization:
            net = slim.unit_norm(net, 1)
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits

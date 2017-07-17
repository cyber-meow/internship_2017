"""Classify images using intern representations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from routines.train import TrainColorDepth
from classify.train import TrainClassify

slim = tf.contrib.slim


class TrainClassifyCommonRepr(TrainClassify):

    default_trainable_scopes = ['Logits']

    def __init__(self, structure, **kwargs):
        super(TrainClassifyCommonRepr, self).__init__(**kwargs)
        self.structure = structure

    def compute_logits(self, inputs, num_classes,
                       modality='color', dropout_keep_prob=0.8):
        assert modality in ['color', 'depth']
        if modality == 'color':
            net, _ = self.structure(
                inputs, tf.zeros_like(inputs),
                final_endpoint='Middle',
                color_keep_prob=tf.constant(1, tf.float32))
        elif modality == 'depth':
            net, _ = self.structure(
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


class TrainClassifyFusion(TrainColorDepth, TrainClassify):

    def __init__(self, structure, **kwargs):
        super(TrainClassifyFusion, self).__init__(**kwargs)
        self.structure = structure

    def decide_used_data(self):
        self.images_color = tf.cond(
            self.training, lambda: self.images_color_train,
            lambda: self.images_color_test)
        self.images_depth = tf.cond(
            self.training, lambda: self.images_depth_train,
            lambda: self.images_depth_test)
        self.labels = tf.cond(
            self.training, lambda: self.labels_train,
            lambda: self.labels_test)

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.images_color, self.images_depth,
            self.dataset_train.num_classes, **kwargs)

    def compute_logits(self, color_inputs, depth_inputs, num_classes,
                       dropout_keep_prob=0.8, endpoint=None):
        if endpoint is None:
            net, _ = self.structure(color_inputs, depth_inputs)
        else:
            net, _ = self.structure(color_inputs, depth_inputs,
                                    final_endpoint=endpoint)
        net = slim.dropout(net, dropout_keep_prob, scope='PreLogitsDropout')
        net = slim.flatten(net, scope='PrelogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits

    def get_summary_op(self):
        self.get_batch_norm_summary()
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.histogram('logits', self.logits)
        tf.summary.scalar('losses/train/cross_entropy',
                          self.cross_entropy_loss)
        tf.summary.scalar('losses/train/total', self.total_loss)
        tf.summary.scalar('accuracy/train', self.accuracy_no_streaming)
        tf.summary.scalar('accuracy/train/streaming', self.accuracy)
        tf.summary.image('train/color', self.images_color)
        tf.summary.image('train/depth', self.images_depth)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        # Summaries for the test part
        ac_test_summary = tf.summary.scalar(
            'accuracy/test', self.accuracy_no_streaming)
        ls_test_summary = tf.summary.scalar(
            'losses/test/total', self.total_loss)
        imgs_color_test = tf.summary.image(
            'test/color', self.images_color)
        imgs_depth_test = tf.summary.image(
            'test/depth', self.images_depth)
        self.test_summary_op = tf.summary.merge(
            [ac_test_summary, ls_test_summary,
             imgs_color_test, imgs_depth_test])
        return self.test_summary_op

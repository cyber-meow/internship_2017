"""Classify images using intern representations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from classify.train import TrainClassify

slim = tf.contrib.slim


class TrainClassifyFusion(TrainClassify):

    def __init__(self, structure, **kwargs):
        super(TrainClassifyFusion, self).__init__(**kwargs)
        self.structure = structure

    @property
    def default_trainable_scopes(self):
        return ['Logits']

    def compute_logits(self, inputs, num_classes,
                       modality='color', dropout_keep_prob=0.8):
        assert modality in ['color', 'depth']
        if modality == 'color':
            net, _ = self.structure(
                inputs, tf.zeros(inputs.get_shape()),
                final_endpoint='Middle',
                color_keep_prob=tf.constant(1-1e-5, tf.float32))
        elif modality == 'depth':
            net, _ = self.structure(
                tf.zeros(inputs.get_shape()), inputs,
                final_endpoint='Middle',
                color_keep_prob=tf.constant(1e-5, tf.float32))
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


def train_classify_fusion(structure,
                          tfrecord_dir,
                          checkpoint_dirs,
                          log_dir,
                          number_of_steps=None,
                          **kwargs):
    train_classify = TrainClassifyFusion(structure)
    for key in kwargs.copy():
        if hasattr(train_classify, key):
            setattr(train_classify, key, kwargs[key])
            del kwargs[key]
    train_classify.train(
        tfrecord_dir, checkpoint_dirs, log_dir,
        number_of_steps=number_of_steps, **kwargs)

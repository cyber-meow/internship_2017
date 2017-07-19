"""Classify images using intern representations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from classify.evaluate import EvaluateClassify
from classify.train import TrainClassify

slim = tf.contrib.slim


class TrainClassifyCAE(TrainClassify):

    def __init__(self, CAE_structure, endpoint='Middle', **kwargs):
        super(TrainClassifyCAE, self).__init__(**kwargs)
        self.CAE_structure = CAE_structure
        self.endpoint = endpoint

    @property
    def default_trainable_scopes(self):
        return ['Logits']

    def compute_logits(self, inputs, num_classes,
                       dropout_keep_prob=0.8, do_avg=False):
        if self.CAE_structure is not None:
            net, _ = self.CAE_structure(
                inputs, dropout_keep_prob=1, final_endpoint=self.endpoint)
        else:
            net = inputs
        net = slim.dropout(net, dropout_keep_prob, scope='PreLogitsDropout')
        if do_avg:
            net = slim.avg_pool2d(
                net, net.get_shape()[1:3], padding='VALID',
                scope='PreLogitsAvgPool')
        self.representation_shape = net.get_shape()
        net = slim.flatten(net, scope='PreLogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits

    def get_init_fn(self, checkpoint_dirs):
        if self.CAE_structure is None:
            return None
        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        assert checkpoint_path is not None
        variables_to_restore = self.get_variables_to_restore(['CAE'])
        return slim.assign_from_checkpoint_fn(
            checkpoint_path, variables_to_restore)

    def extra_log_info(self):
        tf.logging.info('representation shape: %s', self.representation_shape)


class EvaluateClassifyCAE(EvaluateClassify):

    def __init__(self, CAE_structure, endpoint='Middle', **kwargs):
        super(EvaluateClassifyCAE, self).__init__(**kwargs)
        self.CAE_structure = CAE_structure
        self.endpoint = endpoint

    def compute_logits(self, inputs):
        if self.CAE_structure is not None:
            net, _ = self.CAE_structure(
                inputs, final_endpoint=self.endpoint)
        else:
            net = inputs
        net = slim.flatten(net, scope='PreLogitsFlatten')
        self.logits = slim.fully_connected(
            net, self.dataset.num_classes, activation_fn=None, scope='Logits')
        return self.logits

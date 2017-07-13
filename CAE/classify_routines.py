"""Classify images using intern representations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from classify.evaluate import EvaluateClassify
from classify.train import TrainClassify, TrainClassifyGray

slim = tf.contrib.slim


class TrainClassifyCAE(TrainClassify):

    def __init__(self, CAE_structure, endpoint, **kwargs):
        super(TrainClassifyCAE, self).__init__(**kwargs)
        self.CAE_structure = CAE_structure
        self.endpoint = endpoint

    @property
    def default_trainable_scopes(self):
        return ['Logits']

    def compute_logits(self, inputs, num_classes, dropout_keep_prob=0.8):
        if self.CAE_structure is not None:
            net, _ = self.CAE_structure(
                inputs, dropout_keep_prob=1, final_endpoint=self.endpoint)
        else:
            net = inputs
        self.representation_shape = net.get_shape()
        net = slim.dropout(net, dropout_keep_prob, scope='PreLogitsDropout')
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


def train_classify_CAE(CAE_structure,
                       tfrecord_dir,
                       checkpoint_dirs,
                       log_dir,
                       number_of_steps=None,
                       endpoint='Middle',
                       **kwargs):
    train_classify = TrainClassifyCAE(CAE_structure, endpoint)
    for key in kwargs.copy():
        if hasattr(train_classify, key):
            setattr(train_classify, key, kwargs[key])
            del kwargs[key]
    train_classify.train(
        tfrecord_dir, checkpoint_dirs, log_dir,
        number_of_steps=number_of_steps, **kwargs)


class TrainClassifyGrayCAE(TrainClassifyGray, TrainClassifyCAE):
    pass


def train_classify_gray_CAE(CAE_structure,
                            tfrecord_dir,
                            checkpoint_dirs,
                            log_dir,
                            number_of_steps=None,
                            endpoint='Middle',
                            **kwargs):
    train_classify = TrainClassifyGrayCAE(CAE_structure, endpoint)
    for key in kwargs.copy():
        if hasattr(train_classify, key):
            setattr(train_classify, key, kwargs[key])
            del kwargs[key]
    train_classify.train(
        tfrecord_dir, checkpoint_dirs, log_dir,
        number_of_steps=number_of_steps, **kwargs)


class EvaluateClassifyCAE(EvaluateClassify):

    def __init__(self, CAE_structure, endpoint, image_size=299):
        self.image_size = image_size
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


def evaluate_classify_CAE(CAE_structure,
                          tfrecord_dir,
                          checkpoint_dirs,
                          log_dir=None,
                          number_of_steps=None,
                          endpoint='Middle',
                          **kwargs):
    evaluate_classify = EvaluateClassifyCAE(CAE_structure, endpoint)
    for key in kwargs.copy():
        if hasattr(evaluate_classify, key):
            setattr(evaluate_classify, key, kwargs[key])
            del kwargs[key]
    evaluate_classify.evaluate(
        tfrecord_dir, checkpoint_dirs, log_dir, number_of_steps, **kwargs)

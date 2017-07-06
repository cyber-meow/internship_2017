"""Classify images using intern representations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from classify.evaluate import classify_evaluate_inception
from classify.train import TrainClassify

slim = tf.contrib.slim


class TrainClassifyCAE(TrainClassify):

    def __init__(self, CAE_structure, endpoint, **kwargs):
        super(TrainClassifyCAE, self).__init__(**kwargs)
        self.CAE_structure = CAE_structure
        self.endpoint = endpoint

    @property
    def default_trainable_scopes(self):
        return ['Logits']

    def compute_logits(self, inputs, dropout_keep_prob=0.8):
        num_classes = self.datasets['train'].num_classes
        if self.CAE_structure is not None:
            net, _ = self.CAE_structure(
                inputs, dropout_keep_prob=1, final_endpoint=self.endpoint)
        else:
            net = inputs
        self.representation_shape = tf.shape(net)
        net = slim.dropout(net, dropout_keep_prob, scope='PreLogitsDropout')
        net = slim.flatten(net, scope='PreLogitsFlatten')
        self.logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')

    def get_init_fn(self, checkpoint_dirs):
        if self.CAE_structure is None:
            return None
        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        assert checkpoint_path is not None
        variables_to_restore = tf.get_collection(
            tf.GraphKeys.MODEL_VARIABLES, scope='CAE')
        return slim.assign_from_checkpoint_fn(
            checkpoint_path, variables_to_restore)

    def extra_log_info(self):
        tf.logging.info('representation shape: %s', self.representation_shape)


class classify_evaluate_CAE(classify_evaluate_inception):

    def __init__(self, CAE_structure, endpoint, image_size=299):
        self._image_size = image_size
        self.CAE_structure = CAE_structure
        self.endpoint = endpoint

    def compute_logits(self, inputs, num_classes):
        if self.CAE_structure is not None:
            net, _ = self.CAE_structure(
                inputs, final_endpoint=self.endpoint)
        else:
            net = inputs
        net = slim.flatten(net, scope='PreLogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits


def classify_evaluate_CAE_fn(CAE_structure,
                             tfrecord_dir,
                             checkpoint_dirs,
                             log_dir,
                             endpoint='Middle',
                             **kwargs):
    classify_evaluate = classify_evaluate_CAE(CAE_structure, endpoint)
    if 'image_size' in kwargs:
        classify_evaluate._image_size = kwargs['image_size']
        del kwargs['image_size']
    classify_evaluate.evaluate(
        tfrecord_dir, checkpoint_dirs, log_dir, **kwargs)


def train_classify_CAE(CAE_structure,
                       tfrecord_dir,
                       checkpoint_dirs,
                       log_dir,
                       number_of_steps=None,
                       endpoint='Middle',
                       **kwargs):
    classify_train = TrainClassifyCAE(CAE_structure, endpoint)
    for key in kwargs:
        if hasattr(classify_train, key):
            setattr(classify_train, key, kwargs[key])
            del classify_train[key]
    classify_train.train(
        tfrecord_dir, checkpoint_dirs, log_dir,
        number_of_steps=number_of_steps, **kwargs)

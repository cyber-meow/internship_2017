from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from nets import inception_v4

from images.basics import TrainImages
from classify.train import TrainClassifyCNN

slim = tf.contrib.slim


class TrainClassifyImages(TrainImages):

    def decide_used_data(self):
        self.images = tf.cond(
            self.training, lambda: self.images_train,
            lambda: self.images_test)
        self.labels = tf.cond(
            self.training, lambda: self.labels_train,
            lambda: self.labels_test)

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.images, self.dataset_train.num_classes, **kwargs)

    def get_summary_op(self):
        super(TrainClassifyImages, self).get_summary_op()
        tf.summary.image('train')
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        summary_op = super(TrainClassifyImages, self).get_test_summary_op()
        images_test_summary = tf.summary.image('test', self.images)
        self.test_summary_op = tf.summary.merge(
            [summary_op, images_test_summary])
        return self.test_summary_op


class TrainClassifyInception(TrainClassifyImages):

    @property
    def default_trainable_scopes(self):
        return ['InceptionV4/Mixed_7d', 'InceptionV4/Logits']

    def compute_logits(self, inputs, num_classes, **kwargs):
        logits, _ = inception_v4.inception_v4(
            inputs, num_classes=num_classes,
            is_training=self.batch_stat, **kwargs)
        return logits

    def get_init_fn(self, checkpoint_dirs):
        checkpoint_exclude_scopes = [
            'InceptionV4/Logits', 'InceptionV4/AuxLogits']
        variables_to_restore = self.get_variables_to_restore(
            scopes=None, exclude=checkpoint_exclude_scopes)

        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                checkpoint_dirs[0], 'inception_v4.ckpt')

        return slim.assign_from_checkpoint_fn(
            checkpoint_path, variables_to_restore)


def fine_tune_inception(tfrecord_dir,
                        checkpoint_dir,
                        log_dir,
                        number_of_steps=None,
                        image_size=299,
                        **kwargs):
    fine_tune = TrainClassifyInception(image_size)
    for key in kwargs.copy():
        if hasattr(fine_tune, key):
            setattr(fine_tune, key, kwargs[key])
            del kwargs[key]
    fine_tune.train(
        tfrecord_dir, checkpoint_dir, log_dir,
        number_of_steps=number_of_steps, **kwargs)


class TrainClassifyImagesCNN(TrainClassifyImages, TrainClassifyCNN):
    pass


class TrainClassifyImagesCAE(TrainClassifyImages):

    def __init__(self, CAE_structure, endpoint='Middle', **kwargs):
        super(TrainClassifyImagesCAE, self).__init__(**kwargs)
        self.CAE_structure = CAE_structure
        self.endpoint = endpoint

    @property
    def default_trainable_scopes(self):
        return ['Logits']

    def compute_logits(self, inputs, num_classes,
                       dropout_keep_prob=0.8, do_avg=False):
        net, _ = self.CAE_structure(
            inputs, dropout_keep_prob=1, final_endpoint=self.endpoint)
        net = slim.dropout(net, dropout_keep_prob, scope='PreLogitsDropout')
        if do_avg:
            net = slim.avg_pool2d(
                net, net.get_shape()[1:3], padding='VALID',
                scope='PreLogitsAvgPool')
        print('Representation shape', net.get_shape())
        net = slim.flatten(net, scope='PreLogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits

    def get_init_fn(self, checkpoint_dirs):
        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        assert checkpoint_path is not None
        variables_to_restore = self.get_variables_to_restore(['CAE'])
        return slim.assign_from_checkpoint_fn(
            checkpoint_path, variables_to_restore)


class EvaluateClassifyCAE(EvaluateClassifyImages):

    def __init__(self, CAE_structure, endpoint='Middle', **kwargs):
        super(EvaluateClassifyCAE, self).__init__(**kwargs)
        self.CAE_structure = CAE_structure
        self.endpoint = endpoint

    def compute_logits(self, inputs, num_classes):
        net, _ = self.CAE_structure(
            inputs, final_endpoint=self.endpoint)
        net = slim.flatten(net, scope='PreLogitsFlatten')
        self.logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return self.logits

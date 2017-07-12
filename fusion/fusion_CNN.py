from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from fusion.fusion_AE import TrainFusion
from classify.train import TrainClassify

slim = tf.contrib.slim


class TrainFusionCNN(TrainFusion, TrainClassify):

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.images_color, self.images_depth,
            self.dataset_train.num_classes, **kwargs)

    def compute_logits(self, color_inputs, depth_inputs, num_classes):
        net = self.structure(color_inputs, depth_inputs)
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


def train_fusion_CNN(structure,
                     tfrecord_dir,
                     log_dir,
                     number_of_steps=None,
                     **kwargs):
    train_classify = TrainFusionCNN(structure)
    for key in kwargs.copy():
        if hasattr(train_classify, key):
            setattr(train_classify, key, kwargs[key])
            del kwargs[key]
    train_classify.train(
        tfrecord_dir, None, log_dir,
        number_of_steps=number_of_steps, **kwargs)

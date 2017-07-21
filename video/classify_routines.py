from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from video.basics import TrainVideo, EvaluateVideo
from classify.train import TrainClassifyCNN
from classify.evaluate import EvaluateClassifyCNN

slim = tf.contrib.slim


def CNN_lips(inputs,
             final_endpoint='MaxPool_b_2x2x2',
             scope=None,
             per_layer_dropout=False,
             dropout_keep_prob=0.8):

    with tf.variable_scope(scope, 'CNN', [inputs]):
        with slim.arg_scope([slim.convolution, slim.pool],
                            stride=1, padding='VALID'):

            # 60 x 80 x 24 x 1
            endpoint = 'Conv3d_a_2x2x2'
            net = slim.convolution(inputs, 5, [2, 2, 2],
                                   scope='Conv3d_a_2x2x2')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 59 x 79 x 23 x 5
            endpoint = 'Conv3d_b_3x3x2'
            net = slim.convolution(net, 13, [3, 3, 2], stride=[2, 2, 1],
                                   scope='Conv3d_b_3x3x2', padding='SAME')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 30 x 40 x 23 x 13
            endpoint = 'MaxPool_a_2x2x2'
            net = slim.pool(net, [2, 2, 2], 'MAX', stride=2,
                            scope='MaxPool_a_2x2x1')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 15 x 20 x 11 x 13
            endpoint = 'Conv3d_c_2x2x2'
            net = slim.convolution(inputs, 26, [2, 2, 2],
                                   scope='Conv3d_c_2x2x2')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 14 x 19 x 10 x 26
            endpoint = 'Conv3d_d_3x3x3'
            net = slim.convolution(inputs, 73, [3, 3, 3], stride=2,
                                   scope='Conv3d_d_3x3x3', padding='SAME')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 7 x 10 x 5 x 73
            endpoint = 'Conv3d_e_2x3x2'
            net = slim.convolution(inputs, 93, [2, 3, 2],
                                   scope='Conv3d_e_2x2x2')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 6 x 8 x 4 x 93
            endpoint = 'MaxPool_b_2x2x2'
            net = slim.pool(inputs, [2, 2, 2], 'MAX', stride=2,
                            scope='MaxPool_b_2x2x2')
            if final_endpoint == endpoint:
                return net

            # 3 x 4 x 2 x 93
            raise ValueError('Unknown final endpoint %s' % final_endpoint)


class TrainClassifyVideo(TrainVideo, TrainClassifyCNN):

    def decide_used_data(self):
        self.videos = tf.cond(
            self.training, lambda: self.videos_train, lambda: self.videos_test)
        self.images_ex = tf.transpose(self.videos[0], [2, 0, 1, 3])
        self.labels = tf.cond(
            self.training, lambda: self.labels_train, lambda: self.labels_test)

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.videos, self.dataset_train.num_classes, **kwargs)

    def get_summary_op(self):
        self.get_batch_norm_summary()
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.histogram('logits', self.logits)
        tf.summary.image('train', self.images_ex, max_outputs=24)
        tf.summary.scalar('losses/train/cross_entropy',
                          self.cross_entropy_loss)
        tf.summary.scalar('losses/train/total', self.total_loss)
        tf.summary.scalar('accuracy/train', self.accuracy_no_streaming)
        tf.summary.scalar('accuracy/train/streaming', self.accuracy)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        # Summaries for the test part
        ac_test_summary = tf.summary.scalar(
            'accuracy/test', self.accuracy_no_streaming)
        ls_test_summary = tf.summary.scalar(
            'losses/test/total', self.total_loss)
        img_test_summary = tf.summary.image(
            'test', self.images_ex, max_outputs=24)
        self.test_summary_op = tf.summary.merge(
            [ac_test_summary, ls_test_summary, img_test_summary])
        return self.test_summary_op


class EvaluateClassifyVideo(EvaluateVideo, EvaluateClassifyCNN):

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.videos, self.dataset.num_classes, **kwargs)

    def last_step_log_info(self, sess, batch_size):
        return self.step_log_info(sess)

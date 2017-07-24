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

            # 60 x 80 x 12 x 1
            print(inputs.get_shape())
            endpoint = 'Conv3d_a_2x2x2'
            net = slim.convolution(inputs, 5, [2, 2, 2],
                                   scope='Conv3d_a_2x2x2')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 59 x 79 x 11 x 5
            print(net.get_shape())
            endpoint = 'Conv3d_b_3x3x2'
            net = slim.convolution(net, 13, [3, 3, 2], stride=[2, 2, 1],
                                   scope='Conv3d_b_3x3x2', padding='SAME')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 30 x 40 x 11 x 13
            print(net.get_shape())
            endpoint = 'MaxPool_a_2x2x2'
            net = slim.pool(net, [2, 2, 1], 'MAX', stride=[2, 2, 1],
                            scope='MaxPool_a_2x2x1')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 15 x 20 x 11 x 13
            print(net.get_shape())
            endpoint = 'Conv3d_c_2x2x2'
            net = slim.convolution(net, 26, [2, 2, 2],
                                   scope='Conv3d_c_2x2x2')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 14 x 19 x 10 x 26
            print(net.get_shape())
            endpoint = 'Conv3d_d_3x3x3'
            net = slim.convolution(net, 73, [3, 3, 3], stride=2,
                                   scope='Conv3d_d_3x3x3', padding='SAME')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 7 x 10 x 5 x 73
            print(net.get_shape())
            endpoint = 'Conv3d_e_2x3x2'
            net = slim.convolution(net, 93, [2, 3, 2],
                                   scope='Conv3d_e_2x2x2')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 6 x 8 x 4 x 93
            print(net.get_shape())
            endpoint = 'MaxPool_b_2x2x2'
            net = slim.pool(net, [2, 2, 2], 'MAX', stride=2,
                            scope='MaxPool_b_2x2x2')
            print(net.get_shape())
            if final_endpoint == endpoint:
                return net

            # 3 x 4 x 2 x 93
            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def CNN_lips5(inputs,
              final_endpoint='Conv3d_d_4x5x3',
              scope=None,
              per_layer_dropout=False,
              dropout_keep_prob=0.8):

    with tf.variable_scope(scope, 'CNN', [inputs]):
        with slim.arg_scope([slim.convolution, slim.pool],
                            stride=2, padding='SAME'):

            # 60 x 80 x 12 x 1
            print(inputs.get_shape())
            endpoint = 'Conv3d_a_3x3x2'
            net = slim.convolution(inputs, 7, [3, 3, 2], stride=[2, 2, 1],
                                   scope='Conv3d_a_2x2x2')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 30 x 40 x 12 x 7
            print(net.get_shape())
            endpoint = 'Conv3d_b_3x3x2'
            net = slim.convolution(net, 17, [3, 3, 2], stride=[2, 2, 1],
                                   scope='Conv3d_b_3x3x2')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 15 x 20 x 12 x 17
            print(net.get_shape())
            endpoint = 'MaxPool_a_2x2x2'
            net = slim.pool(net, [2, 2, 2], 'MAX', scope='MaxPool_a_2x2x2')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 8 x 10 x 6 x 17
            print(net.get_shape())
            endpoint = 'Conv3d_c_3x3x3'
            net = slim.convolution(net, 67, [3, 3, 3],
                                   scope='Conv3d_c_3x3x3')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 4 x 5 x 3 x 67
            print(net.get_shape())
            endpoint = 'Conv3d_d_4x5x3'
            net = slim.convolution(net, 737, [4, 5, 3], stride=1,
                                   scope='Conv3d_d_4x5x3', padding='VALID')
            print(net.get_shape())
            if final_endpoint == endpoint:
                return net

            # 1 x 1 x 1 x 737
            raise ValueError('Unknown final endpoint %s' % final_endpoint)


class TrainClassifyVideo(TrainVideo, TrainClassifyCNN):

    def decide_used_data(self):
        self.videos = tf.cond(
            self.training, lambda: self.videos_train, lambda: self.videos_test)
        self.images_ex0 = tf.transpose(self.videos[0], [2, 0, 1, 3])
        self.images_ex1 = tf.transpose(self.videos[1], [2, 0, 1, 3])
        self.labels = tf.cond(
            self.training, lambda: self.labels_train, lambda: self.labels_test)

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.videos, self.dataset_train.num_classes, **kwargs)

    def get_summary_op(self):
        self.get_batch_norm_summary()
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.histogram('logits', self.logits)
        tf.summary.image('train_0', self.images_ex0, max_outputs=12)
        tf.summary.image('train_1', self.images_ex1, max_outputs=12)
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
        img_test_summary0 = tf.summary.image(
            'test_0', self.images_ex0, max_outputs=12)
        img_test_summary1 = tf.summary.image(
            'test_1', self.images_ex1, max_outputs=12)
        self.test_summary_op = tf.summary.merge(
            [ac_test_summary, ls_test_summary,
             img_test_summary0, img_test_summary1])
        return self.test_summary_op


class EvaluateClassifyVideo(EvaluateVideo, EvaluateClassifyCNN):

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.videos, self.dataset.num_classes, **kwargs)

    def last_step_log_info(self, sess, batch_size):
        return self.step_log_info(sess)

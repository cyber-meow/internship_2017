"""Implement a shadow CAE that reads raw image as input"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from nets import inception_v4

slim = tf.contrib.slim


def convolutional_autoencoder_shadow(inputs,
                                     final_endpoint='Final',
                                     dropout_keep_prob=0.5,
                                     scope=None):
    end_points = {}

    with tf.variable_scope(scope, 'CAE', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            stride=2, padding='VALID'):

            # 299 x 299 x 3
            net = slim.conv2d(inputs, 32, [3, 3], scope='Conv2d_3x3')
            end_points['Middle'] = net
            if final_endpoint == 'Middle':
                return net, end_points

            # 149 x 149 x 32
            net = slim.dropout(net, keep_prob=dropout_keep_prob,
                               scope='Dropout')
            net = slim.conv2d_transpose(
                net, 3, [3, 3], scope='ConvTrans2d_3x3')
            end_points['Final'] = net
            if final_endpoint == 'Final':
                return net, end_points

            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def convolutional_autoencoder_6layer(inputs,
                                     final_endpoint='Final',
                                     dropout_keep_prob=0.5,
                                     scope=None):
    end_points = {}

    with tf.variable_scope(scope, 'CAE', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            stride=2, padding='VALID'):

            # 299 x 299 x 3
            net = slim.conv2d(
                inputs, 32, [5, 5], stride=3, scope='Conv2d_a_5x5')
            # 99 x 99 x 32
            net = slim.conv2d(net, 48, [3, 3], scope='Conv2d_b_3x3')
            # 49 x 49 x 48
            net = slim.conv2d(net, 64, [3, 3], scope='Conv2d_c_3x3')
            end_points['Middle'] = net
            if final_endpoint == 'Middle':
                return net, end_points

            # 24 x 24 x 64
            net = slim.dropout(net, keep_prob=dropout_keep_prob,
                               scope='Dropout')
            net = slim.conv2d_transpose(
                net, 48, [3, 3], scope='ConvTrans2d_c_3x3')
            # 49 x 49 x 48
            net = slim.conv2d_transpose(
                net, 32, [3, 3], scope='ConvTrans2d_b_3x3')
            # 99 x 99 x 32
            net = slim.conv2d_transpose(
                net, 3, [5, 5], stride=3, scope='ConvTrans2d_a_5x5')
            end_points['Final'] = net
            if final_endpoint == 'Final':
                return net, end_points

            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def CAE_inception(inputs,
                  final_endpoint='Final',
                  dropout_keep_prob=0.5,
                  scope=None):

    net, end_points = inception_v4.inception_v4_base(
        inputs, final_endpoint='Mixed_5a')

    end_points['Middle'] = net
    if final_endpoint == 'Middle':
        return net, end_points

    with tf.variable_scope(scope, 'CAE', [inputs]):
        with slim.arg_scope([slim.conv2d_transpose],
                            stride=1, padding='VALID'):
            # 35 x 35 x 384
            net = slim.dropout(net, keep_prob=dropout_keep_prob,
                               scope='Dropout')
            net = slim.conv2d_transpose(
                net, 192, [3, 3], stride=2, scope='ConvTrans_a_3x3')
            # 71 x 71 x 192
            net = slim.conv2d_transpose(
                net, 96, [3, 3], scope='ConvTrans_b_3x3')
            # 73 x 31 x 96
            net = slim.conv2d_transpose(
                net, 64, [1, 1], padding='SAME', scope='ConvTrans_c_1x1')
            # 73 x 73 x 64
            net = slim.conv2d_transpose(
                net, 64, [3, 3], stride=2, scope='ConvTrans_d_3x3')
            # 147 x 147 x 64
            net = slim.conv2d_transpose(
                net, 32, [3, 3], padding='SAME', scope='ConvTrans_e_3x3')
            # 147 x 147 x 32
            net = slim.conv2d_transpose(
                net, 32, [3, 3], scope='ConvTrans_f_3x3')
            # 149 x 149 x 32
            net = slim.conv2d_transpose(
                net, 3, [3, 3], stride=2, scope='ConvTrans_g_3x3')

            end_points['Final'] = net
            if final_endpoint == 'Final':
                return net, end_points

            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def get_init_fn_inception(checkpoint_dirs):
    assert len(checkpoint_dirs) == 1
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
    if checkpoint_path is None:
        checkpoint_path = os.path.join(checkpoint_dirs[0], 'inception_v4.ckpt')
    variables_to_restore = tf.get_collection(
        tf.GraphKeys.MODEL_VARIABLES, scope='InceptionV4')
    saver = tf.train.Saver(variables_to_restore)

    def restore(sess):
        saver.restore(sess, checkpoint_path)
    return restore

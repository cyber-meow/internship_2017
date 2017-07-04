"""Implement a shadow CAE that reads raw image as input"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def convolutional_autoencoder_shadow(inputs,
                                     final_endpoint='Final',
                                     dropout_keep_prob=0.5,
                                     is_training=True,
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
                                     is_training=True,
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
            net = slim.dropout(
                net, keep_prob=dropout_keep_prob, scope='Dropout')
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

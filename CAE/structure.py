"""Implement some CAEs that read raw image as input"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import inception_v4

slim = tf.contrib.slim


def CAE_shadow(inputs,
               final_endpoint='Final',
               dropout_keep_prob=0.5,
               scope=None):
    endpoints = {}

    with tf.variable_scope(scope, 'CAE', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            stride=2, padding='VALID'):

            # 299 x 299 x 3
            net = slim.conv2d(inputs, 32, [3, 3], scope='Conv2d_3x3')
            endpoints['Middle'] = net
            if final_endpoint == 'Middle':
                return net, endpoints

            # 149 x 149 x 32
            net = slim.dropout(net, keep_prob=dropout_keep_prob,
                               scope='Dropout')
            net = slim.conv2d_transpose(
                net, inputs.get_shape()[3], [3, 3], scope='ConvTrans2d_3x3')
            endpoints['Final'] = net
            if final_endpoint == 'Final':
                return net, endpoints

            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def CAE_6layers(inputs,
                final_endpoint='Final',
                dropout_keep_prob=0.5,
                scope=None):
    endpoints = {}
    in_channels = inputs.get_shape()[3]

    with tf.variable_scope(scope, 'CAE', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            stride=2, padding='VALID'):

            # 299 x 299 x 3 / 83 x 83 x 1
            net = slim.conv2d(
                inputs, in_channels*7, [5, 5], stride=3, scope='Conv2d_a_5x5')

            # 99 x 99 x 21 / 27 x 27 x 7
            endpoint = 'Conv2d_b_3x3'
            net = slim.conv2d(net, in_channels*13, [3, 3],
                              scope='Conv2d_b_3x3')
            endpoints[endpoint] = net
            if final_endpoint == endpoint:
                return net, endpoints

            # 49 x 49 x 39 / 13 x 13 x 13
            endpoint = 'Middle'
            net = slim.conv2d(net, in_channels*19, [3, 3],
                              scope='Conv2d_c_3x3')
            endpoints[endpoint] = net
            if final_endpoint == endpoint:
                return net, endpoints

            # 24 x 24 x 57 / 6 x 6 x 19
            net = slim.dropout(net, keep_prob=dropout_keep_prob,
                               scope='Dropout')
            net = slim.conv2d_transpose(
                net, in_channels*13, [3, 3], scope='ConvTrans2d_a_3x3')

            # 49 x 49 x 36 / 13 x 13 x 13
            net = slim.conv2d_transpose(
                net, in_channels*7, [3, 3], scope='ConvTrans2d_b_3x3')

            # 99 x 99 x 21 / 27 x 27 x 7
            endpoint = 'Final'
            net = slim.conv2d_transpose(
                net, in_channels, [5, 5], stride=3,
                scope='ConvTrans2d_c_5x5')
            endpoints[endpoint] = net
            if final_endpoint == endpoint:
                return net, endpoints

            # 299 x 299 x 3 / 83 x 83 x 1
            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def CAE_inception(inputs,
                  final_endpoint='Final',
                  dropout_keep_prob=0.5,
                  scope=None):

    net, endpoints = inception_v4.inception_v4_base(
        inputs, final_endpoint='Mixed_5a')

    endpoints['Middle'] = net
    if final_endpoint == 'Middle':
        return net, endpoints

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

            endpoints['Final'] = net
            if final_endpoint == 'Final':
                return net, endpoints

            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def CAE_12layers(inputs,
                 final_endpoint='Final',
                 dropout_keep_prob=0.5,
                 scope=None):
    endpoints = {}
    in_channels = inputs.get_shape()[3]

    with tf.variable_scope(scope, 'CAE', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            stride=2, padding='VALID'):

            # 83 x 83 x 1
            net = slim.conv2d(
                inputs, in_channels*3, [3, 3], scope='Conv2d_a_3x3')

            # 41 x 41 x 3
            net = slim.conv2d(
                net, in_channels*9, [3, 3], scope='Conv2d_b_3x3')

            # 20 x 20 x 9
            net = slim.conv2d(
                net, in_channels*18, [2, 2], stride=1, scope='Conv2d_c_2x2')

            # 19 x 19 x 18
            net = slim.conv2d(
                net, in_channels*32, [3, 3], scope='Conv2d_d_3x3')

            # 9 x 9 x 32
            net = slim.conv2d(
                net, in_channels*32, [1, 1], stride=1, scope='Conv2d_e_1x1')

            # 9 x 9 x 32
            endpoint = 'Middle'
            net = slim.conv2d(
                net, in_channels*64, [3, 3], scope='Conv2d_f_3x3')
            endpoints[endpoint] = net
            if final_endpoint == endpoint:
                return net, endpoints

            # 4 x 4 x 64
            net = slim.dropout(
                net, keep_prob=dropout_keep_prob, scope='Dropout')
            net = slim.conv2d_transpose(
                net, in_channels*32, [3, 3], scope='ConvTrans2d_a_3x3')

            # 9 x 9 x 32
            net = slim.conv2d_transpose(
                net, in_channels*32, [1, 1],
                stride=1, scope='ConvTrans2d_b_1x1')

            # 9 x 9 x 32
            net = slim.conv2d_transpose(
                net, in_channels*18, [3, 3], scope='ConvTrans2d_c_3x3')

            # 19 x 19 x 18
            net = slim.conv2d_transpose(
                net, in_channels*9, [2, 2],
                stride=1, scope='ConvTrans2d_d_2x2')

            # 20 x 20 x 9
            net = slim.conv2d_transpose(
                net, in_channels*3, [3, 3], scope='ConvTrans2d_e_3x3')

            # 41 x 41 x 3
            endpoint = 'Final'
            net = slim.conv2d_transpose(
                net, in_channels, [3, 3], scope='ConvTrans2d_f_3x3')
            endpoints[endpoint] = net
            if final_endpoint == endpoint:
                return net, endpoints

            raise ValueError('Unknown final endpoint %s' % final_endpoint)

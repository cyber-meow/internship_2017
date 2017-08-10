"""Implement some CNN architectures for audio used dring my internship."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def CNN_mfcc6(inputs,
              final_endpoint='Conv2d_e_3x3',
              scope=None):
    """Suppose that the input is of size 26 x 24.
    This is the architecture used for the transfer learning task.
    """

    with tf.variable_scope(scope, 'CNN', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            stride=1, padding='VALID'):

            # 26 x 24 x 1
            print(inputs.get_shape())
            endpoint = 'Conv2d_a_2x2'
            net = slim.conv2d(inputs, 5, [2, 2], scope='Conv2d_a_2x2')
            if final_endpoint == endpoint:
                return net

            # 25 x 23 x 5
            print(net.get_shape())
            endpoint = 'Conv2d_b_3x3'
            net = slim.conv2d(net, 13, [3, 3], stride=2,
                              scope='Conv2d_b_3x3')
            if final_endpoint == endpoint:
                return net

            # 12 x 11 x 13
            print(net.get_shape())
            endpoint = 'MaxPool_a_2x2'
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME',
                                  scope='MaxPool_a_2x2')
            if final_endpoint == endpoint:
                return net

            # 6 x 6 x 13
            print(net.get_shape())
            endpoint = 'Conv2d_c_2x2'
            net = slim.conv2d(net, 23, [2, 2], scope='Conv2d_c_2x2')
            if final_endpoint == endpoint:
                return net

            # 5 x 5 x 23
            print(net.get_shape())
            endpoint = 'Conv2d_d_3x3'
            net = slim.conv2d(net, 31, [3, 3], scope='Conv2d_d_3x3')
            if final_endpoint == endpoint:
                return net

            # 3 x 3 x 31
            print(net.get_shape())
            endpoint = 'Conv2d_e_3x3'
            net = slim.conv2d(net, 43, [3, 3], scope='Conv2d_e_3x3')
            print(net.get_shape())
            if final_endpoint == endpoint:
                return net

            # 1 x 1 x 43
            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def CNN_mfcc5(inputs,
              final_endpoint='Conv2d_d_3x3',
              scope=None):
    """Suppose that the input is of size 26 x 20"""

    with tf.variable_scope(scope, 'CNN', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            stride=2, padding='VALID'):

            # 26 x 20 x 1
            print(inputs.get_shape())
            endpoint = 'Conv2d_a_2x2'
            net = slim.conv2d(inputs, 13, [2, 2], stride=1,
                              scope='Conv2d_a_2x2')
            if final_endpoint == endpoint:
                return net

            # 25 x 19 x 13
            print(net.get_shape())
            endpoint = 'Conv2d_b_3x3'
            net = slim.conv2d(inputs, 29, [3, 3], scope='Conv2d_b_3x3')
            if final_endpoint == endpoint:
                return net

            # 12 x 9 x 29
            print(net.get_shape())
            endpoint = 'MaxPool_a_2x2'
            net = slim.max_pool2d(net, [2, 2], padding='SAME',
                                  scope='MaxPool_a_2x2')
            if final_endpoint == endpoint:
                return net

            # 6 x 5 x 29
            print(net.get_shape())
            endpoint = 'Conv2d_c_3x3'
            net = slim.conv2d(net, 43, [3, 3], padding='SAME',
                              scope='Conv2d_c_3x3')
            if final_endpoint == endpoint:
                return net

            # 3 x 3 x 43
            print(net.get_shape())
            endpoint = 'Conv2d_d_3x3'
            net = slim.conv2d(net, 61, [3, 3], stride=1,
                              scope='Conv2d_d_3x3')
            print(net.get_shape())
            if final_endpoint == endpoint:
                return net

            # 1 x 1 x 61
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

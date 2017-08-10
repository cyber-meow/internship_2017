"""Implement some CNN architectures for images used dring my internship."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def CNN_9layers(inputs,
                final_endpoint='AvgPool_a_2x2',
                scope=None):
    """This architecture is used for an input size of 299 x 299."""

    in_channels = inputs.get_shape()[3]

    with tf.variable_scope(scope, 'CNN', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):

            # 299 x 299 x 3
            print(inputs.get_shape())
            endpoint = 'Conv2d_a_3x3'
            net = slim.conv2d(inputs, in_channels*10, [3, 3],
                              stride=2, scope='Conv2d_a_3x3')
            if final_endpoint == endpoint:
                return net

            # 149 x 149 x 30
            print(net.get_shape())
            endpoint = 'Conv2d_b_3x3'
            net = slim.conv2d(net, in_channels*10, [3, 3],
                              scope='Conv2d_b_3x3')
            if final_endpoint == endpoint:
                return net

            # 147 x 147 x 30
            print(net.get_shape())
            endpoint = 'Conv2d_c_3x3'
            net = slim.conv2d(net, in_channels*21, [3, 3],
                              padding='SAME', scope='Conv2d_c_3x3')
            if final_endpoint == endpoint:
                return net

            # 147 x 147 x 63
            print(net.get_shape())
            endpoint = 'MaxPool_a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_a_3x3')
            if final_endpoint == endpoint:
                return net

            # 73 x 73 x 63
            print(net.get_shape())
            endpoint = 'Conv2d_d_3x3'
            net = slim.conv2d(net, in_channels*32, [3, 3],
                              scope='Conv2d_d_3x3')
            if final_endpoint == endpoint:
                return net

            # 71 x 71 x 96
            print(net.get_shape())
            endpoint = 'Maxpool_b_3x3'
            net = slim.max_pool2d(net, [3, 3],
                                  stride=2, scope='MaxPool_b_3x3')
            if final_endpoint == endpoint:
                return net

            # 35 x 35 x 96
            print(net.get_shape())
            endpoint = 'Conv2d_e_3x3'
            net = slim.conv2d(net, in_channels*64, [3, 3],
                              stride=2, scope='Conv2d_e_3x3')
            if final_endpoint == endpoint:
                return net

            # 17 x 17 x 192
            print(net.get_shape())
            endpoint = 'Conv2d_f_3x3'
            net = slim.conv2d(net, in_channels*96, [3, 3],
                              stride=2, scope='Conv2d_f_3x3')
            if final_endpoint == endpoint:
                return net

            # 8 x 8 x 288
            print(net.get_shape())
            endpoint = 'AvgPool_a_2x2'
            net = slim.avg_pool2d(net, [2, 2],
                                  stride=2, scope='AvgPool_a_2x2')
            print(net.get_shape())
            if final_endpoint == endpoint:
                return net

            # 4 x 4 x 288
            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def CNN_10layers(inputs,
                 final_endpoint='Conv2d_h_3x3',
                 scope=None):
    """This architecture is used for an input size of 83 x 83"""

    in_channels = inputs.get_shape()[3]

    with tf.variable_scope(scope, 'CNN', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            stride=1, padding='VALID'):

            # 83 x 83 x 1
            print(inputs.get_shape())
            net = slim.conv2d(inputs, in_channels*3, [3, 3],
                              stride=2, scope='Conv2d_a_3x3')

            # 41 x 41 x 3
            print(net.get_shape())
            net = slim.conv2d(net, in_channels*10,
                              [3, 3], scope='Conv2d_b_3x3')

            # 39 x 39 x 10
            print(net.get_shape())
            net = slim.conv2d(net, in_channels*32, [3, 3],
                              padding='SAME', scope='Conv2d_c_3x3')

            # 39 x 39 x 32
            print(net.get_shape())
            net = slim.max_pool2d(net, [3, 3],
                                  stride=2, scope='MaxPool_a_3x3')

            # 19 x 19 x 32
            print(net.get_shape())
            net = slim.conv2d(net, in_channels*48,
                              [3, 3], scope='Conv2d_d_3x3')

            # 17 x 17 x 48
            print(net.get_shape())
            endpoint = 'MaxPool_b_3x3'
            net = slim.max_pool2d(net, [3, 3],
                                  stride=2, scope='MaxPool_b_3x3')
            if final_endpoint == endpoint:
                return net

            # 8 x 8 x 48
            print(net.get_shape())
            endpoint = 'Conv2d_e_2x2'
            net = slim.conv2d(net, in_channels*96, [2, 2],
                              padding='SAME', scope='Conv2d_e_2x2')
            if final_endpoint == endpoint:
                return net

            # 8 x 8 x 96
            print(net.get_shape())
            endpoint = 'Conv2d_f_2x2'
            net = slim.conv2d(net, in_channels*192, [2, 2],
                              scope='Conv2d_f_2x2')
            if final_endpoint == endpoint:
                return net

            # 7 x 7 x 192
            print(net.get_shape())
            endpoint = 'Conv2d_g_3x3'
            net = slim.conv2d(net, in_channels*256, [3, 3],
                              stride=2, scope='Conv2d_g_3x3')
            if final_endpoint == endpoint:
                return net

            # 3 x 3 x 256
            print(net.get_shape())
            endpoint = 'Conv2d_h_3x3'
            net = slim.conv2d(net, in_channels*512, [3, 3],
                              scope='Conv2d_h_3x3')
            print(net.get_shape())
            if final_endpoint == endpoint:
                return net

            # 1 x 1 x 512
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

"""Implement some CAEs that read raw image as input"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def CNN_9layers(inputs,
                final_endpoint='AvgPool_a_2x2',
                scope=None):

    with tf.variable_scope(scope, 'CNN', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):

            # 299 x 299 x 3
            net = slim.conv2d(inputs, 32, [3, 3],
                              stride=2, scope='Conv2d_a_3x3')

            # 149 x 149 x 32
            endpoint = 'Conv2d_a_3x3'
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_b_3x3')
            if final_endpoint == endpoint:
                return net

            # 147 x 147 x 32
            endpoint = 'Conv2d_b_3x3'
            net = slim.conv2d(net, 64, [3, 3],
                              padding='SAME', scope='Conv2d_c_3x3')
            if final_endpoint == endpoint:
                return net

            # 147 x 147 x 64
            endpoint = 'Conv2d_c_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_a_3x3')
            if final_endpoint == endpoint:
                return net

            # 73 x 73 x 64
            endpoint = 'MaxPool_a_3x3'
            net = slim.conv2d(net, 96, [3, 3], scope='Conv2d_d_3x3')
            if final_endpoint == endpoint:
                return net

            # 71 x 71 x 96
            endpoint = 'Conv2d_d_3x3'
            net = slim.max_pool2d(net, [3, 3],
                                  stride=2, scope='MaxPool_b_3x3')
            if final_endpoint == endpoint:
                return net

            # 35 x 35 x 96
            endpoint = 'Maxpool_b_3x3'
            net = slim.conv2d(net, 192, [3, 3],
                              stride=2, scope='Conv2d_e_3x3')
            if final_endpoint == endpoint:
                return net

            # 17 x 17 x 192
            endpoint = 'Conv2d_f_3x3'
            net = slim.conv2d(net, 288, [3, 3],
                              stride=2, scope='Conv2d_f_3x3')
            if final_endpoint == endpoint:
                return net

            # 8 x 8 x 288
            endpoint = 'AvgPool_a_2x2'
            net = slim.avg_pool2d(net, [2, 2],
                                  stride=2, scope='AvgPool_a_2x2')
            if final_endpoint == endpoint:
                return net

            # 4 x 4 x 288
            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def CNN_8layers(inputs,
                scope=None):

    with tf.variable_scope(scope, 'CNN', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):

            # 83 x 83 x 3
            net = slim.conv2d(inputs, 32, [3, 3],
                              stride=2, scope='Conv2d_a_3x3')

            # 41 x 41 x 32
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_b_3x3')

            # 39 x 39 x 32
            net = slim.conv2d(net, 64, [3, 3],
                              padding='SAME', scope='Conv2d_c_3x3')

            # 39 x 39 x 64
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_a_3x3')

            # 19 x 19 x 64
            net = slim.conv2d(net, 96, [3, 3], scope='Conv2d_d_3x3')

            # 17 x 17 x 64
            net = slim.max_pool2d(net, [3, 3],
                                  stride=2, scope='MaxPool_b_3x3')

            # 8 x 8 x 96
            net = slim.conv2d(net, 192, [2, 2],
                              padding='SAME', scope='Conv2d_f_2x2')

            # 8 x 8 x 192
            net = slim.avg_pool2d(net, [2, 2],
                                  stride=2, scope='AvgPool_a_2x2')

            # 4 x 4 x 192
            return net

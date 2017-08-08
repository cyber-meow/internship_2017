"""Implement some modality fusion architectures"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from images.CAE_architecture import CAE_6layers
from images.CNN_architecture import CNN_9layers, CNN_10layers

slim = tf.contrib.slim


def deconvolve_3layer(inputs, channels, scope=None):
    with tf.variable_scope(scope, 'Deconvolution', [inputs]):
        with slim.arg_scope(
                [slim.conv2d_transpose], stride=2, padding='VALID'):
            # 24 x 24 x 87 / 6 x 6 x 29
            net = slim.conv2d_transpose(
                inputs, channels*23, [3, 3], scope='ConvTrans2d_a_3x3')
            # 49 x 49 x 69 / 13 x 13 x 23
            net = slim.conv2d_transpose(
                net, channels*13, [3, 3], scope='ConvTrans2d_b_3x3')
            # 99 x 99 x 39 / 27 x 27 x 13
            net = slim.conv2d_transpose(
                net, channels, [5, 5], stride=3,
                activation_fn=None, scope='ConvTrans2d_c_5x5')
            return net


def fusionAE_6layers(color_inputs, depth_inputs,
                     final_endpoint='Final',
                     scope=None,
                     color_keep_prob=None, depth_keep_prob=None):
    if color_keep_prob is None:
        if depth_keep_prob is None:
            color_keep_prob = tf.random_uniform([])
        else:
            color_keep_prob = tf.constant(1, tf.float32) - depth_keep_prob
    if depth_keep_prob is None:
        depth_keep_prob = tf.constant(1, tf.float32) - color_keep_prob

    in_channels = color_inputs.get_shape()[3]
    print(color_inputs.get_shape())
    print(depth_inputs.get_shape())
    # endpoints = {}

    with tf.variable_scope(scope, 'Fusion'):
        # 299 x 299 x 3 / 83 x 83 x 1
        color_net = CAE_6layers(
            color_inputs, final_endpoint='Conv2d_b_3x3', scope='Color')
        depth_net = CAE_6layers(
            depth_inputs, final_endpoint='Conv2d_b_3x3', scope='Depth')

        # 49 x 49 x 69 / 13 x 13 x 23
        color_net = tf.cond(
            tf.equal(color_keep_prob, tf.constant(0, tf.float32)),
            lambda: tf.zeros_like(color_net, name='Color/Dropout'),
            lambda: tf.nn.dropout(
                color_net, keep_prob=color_keep_prob, name='Color/Dropout'))
        depth_net = tf.cond(
            tf.equal(depth_keep_prob, tf.constant(0, tf.float32)),
            lambda: tf.zeros_like(depth_net, name='Depth/Dropout'),
            lambda: tf.nn.dropout(
                depth_net, keep_prob=depth_keep_prob, name='Depth/Dropout'))

        with slim.arg_scope([slim.conv2d], stride=2, padding='VALID'):
            # 24 x 24 x 87 / 6 x 6 x 29
            color_net = slim.conv2d(
                color_net, in_channels*29, [3, 3],
                scope='Fusion/Color/ConvTrans2d_c_3x3')
            depth_net = slim.conv2d(
                depth_net, in_channels*29, [3, 3],
                scope='Fusion/Depth/ConvTrans2d_c_3x3')

        color_net = tf.cond(
            tf.equal(depth_keep_prob, tf.constant(0, tf.float32)),
            lambda: color_net * 2, lambda: color_net)
        depth_net = tf.cond(
            tf.equal(color_keep_prob, tf.constant(0, tf.float32)),
            lambda: depth_net * 2, lambda: depth_net)

        # 24 x 24 x 24
        endpoint = 'Middle'
        net = color_net + depth_net
        # endpoints[endpoint] = net
        if final_endpoint == endpoint:
            return net

        endpoint = 'Final'
        color_net = deconvolve_3layer(
            net, in_channels, scope='Separation/Color')
        depth_net = deconvolve_3layer(
            net, in_channels, scope='Separation/Depth')
        # endpoints[endpoint] = (color_net, depth_net)
        if final_endpoint == endpoint:
            return color_net, depth_net

        raise ValueError('Unknown final endpoint %s' % final_endpoint)


def fusion_CNN_11layers(color_inputs, depth_inputs, scope=None):
    with tf.variable_scope(scope, 'FusionCNN'):
        # 299 x 299 x 3
        color_net = CNN_9layers(
            color_inputs, final_endpoint='Conv2d_f_3x3', scope='Color')
        depth_net = CNN_9layers(
            depth_inputs, final_endpoint='Conv2d_f_3x3', scope='Depth')

        # 8 x 8 x 288
        net = color_net + depth_net

        with tf.variable_scope('AfterFusion'):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                stride=1, padding='VALID'):
                net = slim.conv2d(net, 288, [2, 2], scope='Conv2d_a_2x2')

                # 7 x 7 x 288
                net = slim.conv2d(net, 512, [3, 3], stride=2,
                                  scope='Conv2d_b_3x3')

                # 3 x 3 x 512
                net = slim.conv2d(net, 1024, [3, 3], scope='Conv2d_c_3x3')

                # 1 x 1 x 1024
                return net


def fusion_CNN_10layers(color_inputs, depth_inputs, scope=None):
    with tf.variable_scope(scope, 'FusionCNN'):
        # 83 x 83 x 1
        color_net = CNN_10layers(
            color_inputs, final_endpoint='Conv2d_e_2x2', scope='Color')
        depth_net = CNN_10layers(
            depth_inputs, final_endpoint='Conv2d_e_2x2', scope='Depth')

        # 8 x 8 x 96
        net = color_net + depth_net
        in_channels = color_inputs.get_shape()[3]

        with tf.variable_scope('AfterFusion'):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                stride=1, padding='VALID'):
                net = slim.conv2d(net, in_channels*192, [2, 2],
                                  scope='Conv2d_a_2x2')

                # 7 x 7 x 192
                net = slim.conv2d(net, 256, [3, 3], stride=2,
                                  scope='Conv2d_b_3x3')

                # 3 x 3 x 256
                net = slim.conv2d(net, 512, [3, 3], scope='Conv2d_c_3x3')

                # 1 x 1 x 512
                return net

"""Implement some modality fusion structures"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from CAE.structure import CAE_6layers
from classify.CNN_structure import CNN_9layers

slim = tf.contrib.slim


def deconvolve_3layer(inputs, channels, scope=None):
    with tf.variable_scope(scope, 'Deconvolution', [inputs]):
        with slim.arg_scope(
                [slim.conv2d_transpose], stride=2, padding='VALID'):
            # 24 x 24 x ?
            net = slim.conv2d_transpose(
                inputs, 48, [3, 3], scope='ConvTrans2d_a_3x3')
            # 49 x 49 x 48
            net = slim.conv2d_transpose(
                net, 32, [3, 3], scope='ConvTrans2d_b_3x3')
            # 99 x 99 x 32
            net = slim.conv2d_transpose(
                net, channels, [5, 5], stride=3, scope='ConvTrans2d_c_5x5')
            return net


def fusion_AE_6layers(color_inputs, depth_inputs,
                      final_endpoint='Final',
                      color_keep_prob=None, depth_keep_prob=None):
    if color_keep_prob is None:
        if depth_keep_prob is None:
            color_keep_prob = tf.random_uniform([])
        else:
            color_keep_prob = tf.constant(1, tf.float32) - depth_keep_prob
    if depth_keep_prob is None:
        depth_keep_prob = tf.constant(1, tf.float32) - color_keep_prob
    endpoints = {}

    with tf.variable_scope('Fusion'):
        # 299 x 299 x 3, 299 x 299 x 3
        color_net, _ = CAE_6layers(
            color_inputs, final_endpoint='Conv2d_b_3x3', scope='Color')
        depth_net, _ = CAE_6layers(
            depth_inputs, final_endpoint='Conv2d_b_3x3', scope='Depth')

        # 49 x 49 x 48, 49 x 49 x 48
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
            # 24 x 24 x 64, 24 x 24 x 64
            color_net = slim.conv2d(
                color_net, 64, [3, 3], scope='Fusion/Color/ConvTrans2d_c_3x3')
            depth_net = slim.conv2d(
                depth_net, 64, [3, 3], scope='Fusion/Depth/ConvTrans2d_c_3x3')

        # 24 x 24 x 24
        endpoint = 'Middle'
        net = color_net + depth_net
        endpoints[endpoint] = net
        if final_endpoint == endpoint:
            return net, endpoints

        endpoint = 'Final'
        color_net = deconvolve_3layer(
            net, color_inputs.get_shape()[3], scope='Separation/Color')
        depth_net = deconvolve_3layer(
            net, depth_inputs.get_shape()[3], scope='Separation/Depth')
        endpoints[endpoint] = (color_net, depth_net)
        if final_endpoint == endpoint:
            return (color_net, depth_net), endpoints

        raise ValueError('Unknown final endpoint %s' % final_endpoint)


def fusion_CNN(color_inputs, depth_inputs):
    with tf.variable_scope('FusionCNN'):
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
                return net, None

"""Implement some modality fusion structures"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from CAE.structure import CAE_6layer

slim = tf.contrib.slim


def deconvolve_3layer(inputs, scope=None):
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
                net, 3, [5, 5], stride=3, scope='ConvTrans2d_c_5x5')
            return net


def fusion_AE_6layer(color_inputs, depth_inputs,
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
        color_net, _ = CAE_6layer(
            color_inputs, final_endpoint='Conv2d_b_3x3', scope='Color')
        depth_net, _ = CAE_6layer(
            depth_inputs, final_endpoint='Conv2d_b_3x3', scope='Depth')

        # 49 x 49 x 48, 49 x 49 x 48
        color_net = tf.nn.dropout(
            color_net, keep_prob=color_keep_prob, name='Color/Dropout')
        depth_net = tf.nn.dropout(
            depth_net, keep_prob=depth_keep_prob, name='Depth/Dropout')

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
        color_net = deconvolve_3layer(net, scope='Separation/Color')
        depth_net = deconvolve_3layer(net, scope='Separation/Depth')
        endpoints[endpoint] = (color_net, depth_net)
        if final_endpoint == endpoint:
            return (color_net, depth_net), endpoints

        raise ValueError('Unknown final endpoint %s' % final_endpoint)
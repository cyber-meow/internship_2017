from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def CNN_lips8(inputs,
              final_endpoint='Conv3d_f_3x4x2',
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

            # 6 x 8 x 4 x 137
            print(net.get_shape())
            endpoint = 'MaxPool_b_2x2x2'
            net = slim.pool(net, [2, 2, 2], 'MAX', stride=2,
                            scope='MaxPool_b_2x2x2')
            if final_endpoint == endpoint:
                return net
            if per_layer_dropout:
                net = slim.dropout(net, dropout_keep_prob)

            # 3 x 4 x 2 x 137
            print(net.get_shape())
            endpoint = 'Conv3d_f_3x4x2'
            net = slim.convolution(net, 657, [3, 4, 2],
                                   scope='Conv3d_f_3x4x2')
            print(net.get_shape())
            if final_endpoint == endpoint:
                return net

            # 1 x 1 x 1 x 657
            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def CNN_lips5(inputs,
              entry_point='inputs',
              final_endpoint='Conv3d_d_4x5x3',
              scope=None,
              per_layer_dropout=False,
              dropout_keep_prob=0.8):

    with tf.variable_scope(scope, 'CNN', [inputs]):
        with slim.arg_scope([slim.convolution, slim.pool],
                            stride=2, padding='SAME'):

            # 60 x 80 x 12 x 1
            if entry_point == 'inputs':
                endpoint = 'Conv3d_a_3x3x2'
                print(inputs.get_shape())
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
            elif entry_point == 'Conv3d_b_3x3x2':
                net = inputs
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

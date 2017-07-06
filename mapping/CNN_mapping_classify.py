"""Classify color images using the mapping"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import inception_v4

from classify.evaluate import classify_evaluate_inception

slim = tf.contrib.slim


class classify_evaluate_mapping_inception(classify_evaluate_inception):

    @staticmethod
    def inception_feature(net):
        with tf.variable_scope('InceptionV4', [net]):
            # 8 x 8 x 1536
            net = slim.avg_pool2d(net, net.get_shape()[1:3],
                                  padding='VALID', scope='AvgPool_1a')
            # 1 x 1 x 1536
            net = slim.flatten(net, scope='PreLogitsFlatten')
            return net

    def compute_logits(self, inputs, num_classes):
        with tf.variable_scope('Color', values=[inputs]):
            net_color, _ = inception_v4.inception_v4_base(inputs)
            net_color = self.inception_feature(net_color)
        mapping = slim.fully_connected(
            net_color, net_color.get_shape()[1].value,
            activation_fn=None, scope='Mapping')
        with tf.variable_scope('Depth', values=[mapping]):
            logits = slim.fully_connected(
                mapping, num_classes,
                activation_fn=None, scope='Logits')
        return logits

    # Don't put / at the end of directory name

    def init_models(self, sess, checkpoint_dirs):
        assert len(checkpoint_dirs) == 2
        checkpoints_dir_mapping, checkpoints_dir_classify = checkpoint_dirs

        variables_mapping = []
        variables_classify = {}

        for var in tf.model_variables():
            if (var.op.name.startswith('Color') or
                    var.op.name.startswith('Mapping')):
                variables_mapping.append(var)
            if var.op.name.startswith('Depth'):
                checkpoint_var_name = 'InceptionV4/Logits/'+var.op.name[6:]
                variables_classify[checkpoint_var_name] = var

        saver_mapping = tf.train.Saver(variables_mapping)
        saver_classify = tf.train.Saver(variables_classify)

        checkpoint_path_mapping = tf.train.latest_checkpoint(
            checkpoints_dir_mapping)
        checkpoint_path_classify = tf.train.latest_checkpoint(
            checkpoints_dir_classify)

        saver_mapping.restore(sess, checkpoint_path_mapping)
        saver_classify.restore(sess, checkpoint_path_classify)


classify_evaluate_mapping_inception_fn = \
    classify_evaluate_mapping_inception().evaluate

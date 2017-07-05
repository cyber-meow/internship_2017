"""Common arg_scope used by many nets"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def nets_arg_scope(weight_decay=0.0004,
                   use_batch_norm=True,
                   batch_norm_decay=0.9,
                   batch_norm_epsilon=0.001,
                   is_training=True):
    """Defines the default arg scope for some net models.

    Args:
      weight_decay: The weight decay to use for regularizing the model.
      use_batch_norm: "If `True`, batch_norm is applied after each convolution.
      batch_norm_decay: Decay for batch norm moving average.
      batch_norm_epsilon: Small float added to variance to avoid dividing
        by zero in batch norm.
      is_training: The model is being trained or not

    Returns:
      An `arg_scope` to use for the inception models.
  """
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': batch_norm_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': batch_norm_epsilon,
        # collection containing update_ops.
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d, slim.conv2d_transpose],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params):
            with slim.arg_scope(
                    [slim.batch_norm, slim.dropout],
                    is_training=is_training) as sc:
                return sc

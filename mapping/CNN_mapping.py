"""Try to find a mapping from color to depth image representation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import os
import time

import numpy as np
import tensorflow as tf
from nets import inception_v4

from data.color_depth.read_TFRecord import get_split_color_depth
from data.color_depth.load_batch import load_batch_color_depth
from nets_base.arg_scope import nets_arg_scope

slim = tf.contrib.slim


# Don't put / at the end of directory name

def get_init_fn(checkpoints_dir_color, checkpoints_dir_depth):
    """Returns a function run by the chief worker to
       warm-start the training."""

    variables_color = {}
    variables_depth = {}

    for var in tf.model_variables():
        if var.op.name.startswith('color'):
            variables_color[var.op.name[6:]] = var
        if var.op.name.startswith('depth'):
            variables_depth[var.op.name[6:]] = var

    saver_color = tf.train.Saver(variables_color)
    saver_depth = tf.train.Saver(variables_depth)

    if tf.train.checkpoint_exists(checkpoints_dir_color):
        checkpoint_path_color = tf.train.latest_checkpoint(
            checkpoints_dir_color)
    else:
        raise IndexError
        checkpoint_path_color = os.path.join(
            checkpoints_dir_color, 'inception_v4.ckpt')

    if tf.train.checkpoint_exists(checkpoints_dir_depth):
        checkpoint_path_depth = tf.train.latest_checkpoint(
            checkpoints_dir_depth)
    else:
        checkpoint_path_depth = os.path.join(
            checkpoints_dir_depth, 'inception_v4.ckpt')

    def restore(sess):
        saver_color.restore(sess, checkpoint_path_color)
        saver_depth.restore(sess, checkpoint_path_depth)

    return restore


def inception_feature(net):
    with tf.variable_scope('InceptionV4', [net]):
        # 8 x 8 x 1536
        net = slim.avg_pool2d(net, net.get_shape()[1:3],
                              padding='VALID', scope='AvgPool_1a')
        # 1 x 1 x 1536
        net = slim.flatten(net, scope='PreLogitsFlatten')
        return net


def train_step(sess, train_op, global_step, *args):

    tensors_to_run = [train_op, global_step]
    tensors_to_run.extend(args)

    start_time = time.time()
    tensor_values = sess.run(tensors_to_run, feed_dict={'training:0': True})
    time_elapsed = time.time() - start_time

    total_loss = tensor_values[0]
    global_step_count = tensor_values[1]

    tf.logging.info(
        'global step %s: loss: %.4f (%.2f sec/step)',
        global_step_count, total_loss, time_elapsed)

    return tensor_values


def train_mapping(tfrecord_dir,
                  checkpoints_dir_color,
                  checkpoints_dir_depth,
                  log_dir,
                  number_of_steps=None,
                  number_of_epochs=5,
                  batch_size=24,
                  save_summaries_step=5,
                  do_test=True,
                  dropout_keep_prob=0.8,
                  initial_learning_rate=0.005,
                  lr_decay_steps=100,
                  lr_decay_rate=0.8):

    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

    image_size = 299

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        with tf.name_scope('Data_provider'):
            dataset = get_split_color_depth('train', tfrecord_dir)
            images_color_train, images_depth_train, _ = \
                load_batch_color_depth(
                    dataset, height=image_size, width=image_size,
                    batch_size=batch_size)

            dataset_test = get_split_color_depth('validation', tfrecord_dir)
            images_color_test, images_depth_test, _ = \
                load_batch_color_depth(
                    dataset_test, height=image_size, width=image_size,
                    batch_size=batch_size)

        training = tf.placeholder(tf.bool, shape=(), name='training')
        images_color = tf.cond(training, lambda: images_color_train,
                               lambda: images_color_test)
        images_depth = tf.cond(training, lambda: images_depth_train,
                               lambda: images_depth_test)

        if number_of_steps is None:
            number_of_steps = int(np.ceil(
                dataset.num_samples * number_of_epochs / batch_size))

        with slim.arg_scope(nets_arg_scope(is_training=training)):
            with tf.variable_scope('Color', values=[images_color]):
                net_color, _ = inception_v4.inception_v4_base(images_color)
                net_color = inception_feature(net_color)

            with tf.variable_scope('Depth', values=[images_depth]):
                net_depth, _ = inception_v4.inception_v4_base(images_depth)
                net_depth = inception_feature(net_depth)

            mapping = slim.fully_connected(
                net_color, net_depth.get_shape()[1].value, activation_fn=None,
                scope='Mapping')

        tf.losses.mean_squared_error(mapping, net_depth)
        total_loss = tf.losses.get_total_loss()

        # Create the global step for monitoring training
        global_step = tf.train.get_or_create_global_step()

        # Exponentially decaying learning rate
        learning_rate = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=lr_decay_steps,
            decay_rate=lr_decay_rate, staircase=True)

        # Optimizer and train op
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = slim.learning.create_train_op(
            total_loss, optimizer,
            variables_to_train=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='Mapping'))

        # Track moving mean and moving varaince
        try:
            last_moving_mean = [
                v for v in tf.model_variables()
                if v.op.name.endswith('moving_mean')][0]
            last_moving_variance = [
                v for v in tf.model_variables()
                if v.op.name.endswith('moving_variance')][0]
            tf.summary.histogram('batch_norm/last_layer/moving_mean',
                                 last_moving_mean)
            tf.summary.histogram('batch_norm/last_layer/moving_variance',
                                 last_moving_variance)
        except IndexError:
            pass

        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('losses/train/total_loss', total_loss)
        tf.summary.image('train/color', images_color)
        tf.summary.image('train/depth', images_depth)
        summary_op = tf.summary.merge_all()

        ls_test_summary = tf.summary.scalar(
            'losses/test/total_loss', total_loss)
        imgs_test_color_summary = tf.summary.image('test/color', images_color)
        imgs_test_depth_summary = tf.summary.image('test/depth', images_depth)
        test_summary_op = tf.summary.merge(
            [ls_test_summary, imgs_test_color_summary,
             imgs_test_depth_summary])

        sv = tf.train.Supervisor(
            logdir=log_dir, summary_op=None,
            init_fn=get_init_fn(checkpoints_dir_color, checkpoints_dir_depth))

        with sv.managed_session() as sess:
            for step in xrange(number_of_steps):
                if (step+1) % save_summaries_step == 0:
                    loss, _, summaries = train_step(
                        sess, train_op, sv.global_step, summary_op)
                    sv.summary_computed(sess, summaries)
                    if do_test:
                        ls, summaries_test = sess.run(
                            [total_loss, test_summary_op],
                            feed_dict={training: False})
                        tf.logging.info('Current Test Loss: %s', ls)
                        sv.summary_computed(sess, summaries_test)
                else:
                    loss = train_step(
                        sess, train_op, sv.global_step)[0]

            tf.logging.info('Finished training. Final Loss: %s', loss)
            tf.logging.info('Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

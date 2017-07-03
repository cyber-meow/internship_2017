"""Implement a shadow CAE that reads raw image as input"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import time

import numpy as np
import tensorflow as tf

import data.read_TFRecord as read_TFRecord
from data.load_batch import load_batch


slim = tf.contrib.slim


def convolutional_autoencoder_shadow(inputs,
                                     dropout_keep_prob=0.5,
                                     is_training=True,
                                     scope=None):
    with tf.variable_scope(scope, 'CAE', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            stride=2, padding='VALID'):
            # 299 x 299 x 3
            net = slim.conv2d(inputs, 32, [3, 3], scope='Conv2d_3x3')
            # 149 x 149 x 32
            net = slim.dropout(net, keep_prob=dropout_keep_prob,
                               scope='Dropout')
            net = slim.conv2d_transpose(
                net, 3, [3, 3], scope='ConvTrans2d_3x3')
            # 299 x 299 x 3
            return net


def train(dataset_dir,
          log_dir,
          number_of_steps=None,
          number_of_epochs=5,
          batch_size=24,
          save_summaries_step=5,
          dropout_keep_prob=0.5):

    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

    image_size = 299

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        with tf.name_scope('data_provider'):
            dataset = read_TFRecord.get_split('train', dataset_dir)

            # Don't crop images
            images, _, labels = load_batch(
                dataset, height=image_size, width=image_size,
                batch_size=batch_size, is_training=False)

        if number_of_steps is None:
            number_of_steps = int(np.ceil(
                dataset.num_samples * number_of_epochs / batch_size))

        # Create the model, use the default arg scope to configure the
        # batch norm parameters
        with slim.arg_scope(
                [slim.conv2d, slim.conv2d_transpose],
                weights_initializer=slim.variance_scaling_initializer(),
                weights_regularizer=slim.l2_regularizer(4e-3)):
            reconstruction = convolutional_autoencoder_shadow(
                images, dropout_keep_prob=dropout_keep_prob)

        tf.losses.mean_squared_error(reconstruction, images)
        total_loss = tf.losses.get_total_loss()

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        tf.summary.scalar('losses/total_loss', total_loss)
        tf.summary.image('input', images)
        tf.summary.image('reconstruction', reconstruction)
        summary_op = tf.summary.merge_all()

        sv = tf.train.Supervisor(logdir=log_dir, summary_op=None)

        with sv.managed_session() as sess:
            for step in xrange(number_of_steps):
                start_time = time.time()
                total_loss, global_step_count, summaries = sess.run(
                    [train_op, sv.global_step, summary_op])
                time_elapsed = time.time() - start_time

                tf.logging.info(
                    'global step %s: loss: %.4f (%.2f sec/step',
                    global_step_count, total_loss, time_elapsed)
                sv.summary_computed(sess, summaries)

            tf.logging.info('Finished training. Final Loss: %s', total_loss)
            tf.logging.info('Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)


def evaluate(dataset_dir,
             train_dir,
             log_dir,
             number_of_steps=None,
             batch_size=12):

    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

    image_size = 299

    with tf.Graph().as_default():

        tf.logging.set_verbosity(tf.logging.INFO)

        dataset = read_TFRecord.get_split('validation', dataset_dir)
        images, _, labels = load_batch(
            dataset, height=image_size, width=image_size,
            batch_size=batch_size, is_training=False)

        if number_of_steps is None:
            number_of_steps = int(np.ceil(dataset.num_samples / batch_size))

        with slim.arg_scope(
                [slim.conv2d, slim.conv2d_transpose],
                weights_initializer=slim.variance_scaling_initializer(),
                weights_regularizer=slim.l2_regularizer(4e-3)):
            reconstruction = convolutional_autoencoder_shadow(
                images, is_training=False)

        tf.losses.mean_squared_error(reconstruction, images)
        total_loss = tf.losses.get_total_loss()

        # Define global step to be show in tensorboard
        global_step = tf.train.get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step+1)

        # Definie summaries
        tf.summary.scalar('losses/total_loss', total_loss)
        tf.summary.image('input', images)
        tf.summary.image('reconstruction', reconstruction)
        summary_op = tf.summary.merge_all()

        # File writer for the tensorboard
        fw = tf.summary.FileWriter(log_dir)

        checkpoint_path = tf.train.latest_checkpoint(train_dir)
        saver = tf.train.Saver(tf.model_variables())

        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):
                sess.run(tf.variables_initializer([global_step]))
                saver.restore(sess, checkpoint_path)

                for step in xrange(number_of_steps-1):
                    start_time = time.time()
                    loss, global_step_count, summaries = sess.run(
                        [total_loss, global_step_op, summary_op])
                    time_elapsed = time.time() - start_time

                    tf.logging.info(
                        'global step %s: loss: %.4f (%.2f sec/step',
                        global_step_count, loss, time_elapsed)
                    fw.add_summary(summaries, global_step=global_step_count)

                tf.logging.info('Finished evaluation.')

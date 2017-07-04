"""Implement a shadow CAE that reads raw image as input"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import time

import numpy as np
import tensorflow as tf

import data.images.read_TFRecord as read_TFRecord
from data.images.load_batch import load_batch
from nets_base.arg_scope import nets_arg_scope

slim = tf.contrib.slim


def train_step(sess, train_op, global_step, *args):

    tensors_to_run = [train_op, global_step]
    tensors_to_run.extend(args)

    start_time = time.time()
    tensor_values = sess.run(tensors_to_run)
    time_elapsed = time.time() - start_time

    total_loss = tensor_values[0]
    global_step_count = tensor_values[1]

    tf.logging.info(
        'global step %s: loss: %.4f (%.2f sec/step)',
        global_step_count, total_loss, time_elapsed)

    return tensor_values


def train_CAE(dataset_dir,
              log_dir,
              CAE_structure,
              number_of_steps=None,
              number_of_epochs=5,
              batch_size=24,
              save_summaries_step=5,
              dropout_position='fc',
              dropout_keep_prob=0.5):

    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

    image_size = 299

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        with tf.name_scope('data_provider'):
            dataset = read_TFRecord.get_split('train', dataset_dir)

            # Don't crop images
            images_original, _, _ = load_batch(
                dataset, height=image_size, width=image_size,
                batch_size=batch_size, is_training=False)

            images_corrupted = slim.dropout(
                images_original, keep_prob=dropout_keep_prob, scope='Dropout')

            assert dropout_position in ['fc', 'input']
            dropout_input = dropout_position == 'input'
            images = images_corrupted if dropout_input else images_original

            if dropout_input:
                dropout_keep_prob = 1

        if number_of_steps is None:
            number_of_steps = int(np.ceil(
                dataset.num_samples * number_of_epochs / batch_size))

        # Create the model, use the default arg scope to configure the
        # batch norm parameters
        with slim.arg_scope(nets_arg_scope()):
            reconstruction, _ = CAE_structure(
                images, dropout_keep_prob=dropout_keep_prob)

        tf.losses.mean_squared_error(reconstruction, images_original)
        total_loss = tf.losses.get_total_loss()

        # Create the global step for monitoring training
        global_step = tf.train.get_or_create_global_step()

        # Exponentially decaying learning rate
        learning_rate = tf.train.exponential_decay(
            learning_rate=0.01,
            global_step=global_step,
            decay_steps=200,
            decay_rate=0.8, staircase=True)

        # Optimizer and train op
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        tf.summary.scalar('losses/total_loss', total_loss)
        tf.summary.image('input', images)
        tf.summary.image('reconstruction', reconstruction)
        summary_op = tf.summary.merge_all()

        sv = tf.train.Supervisor(logdir=log_dir, summary_op=None)

        with sv.managed_session() as sess:
            for step in xrange(number_of_steps):
                if (step+1) % save_summaries_step == 0:
                    loss, _, summaries = train_step(
                        sess, train_op, sv.global_step, summary_op)
                    sv.summary_computed(sess, summaries)
                else:
                    loss = train_step(sess, train_op, sv.global_step)[0]

            tf.logging.info('Finished training. Final Loss: %s', loss)
            tf.logging.info('Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)


def evaluate_CAE(dataset_dir,
                 train_dir,
                 log_dir,
                 CAE_structure,
                 number_of_steps=None,
                 batch_size=12,
                 dropout_input=False,
                 dropout_keep_prob=0.5):

    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

    image_size = 299

    with tf.Graph().as_default():

        tf.logging.set_verbosity(tf.logging.INFO)

        with tf.name_scope('data_provider'):
            dataset = read_TFRecord.get_split('validation', dataset_dir)

            images_original, _, labels = load_batch(
                dataset, height=image_size, width=image_size,
                batch_size=batch_size, is_training=False)

            images_corrupted = slim.dropout(
                images_original, keep_prob=dropout_keep_prob, scope='Dropout')

            images = images_corrupted if dropout_input else images_original

        if number_of_steps is None:
            number_of_steps = int(np.ceil(dataset.num_samples / batch_size))

        with slim.arg_scope(nets_arg_scope()):
            reconstruction, _ = CAE_structure(images, is_training=False)

        tf.losses.mean_squared_error(reconstruction, images_original)
        total_loss = tf.losses.get_total_loss()

        # Define global step to be show in tensorboard
        global_step = tf.train.get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step+1)

        # Definie summaries
        tf.summary.scalar('losses/total_loss', total_loss)
        tf.summary.image('input', images_original)
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

"""Implement a shadow CAE that reads raw image as input"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import os
import time

import numpy as np
import tensorflow as tf

import data.images.read_TFRecord as read_TFRecord
from data.images.load_batch import load_batch
from nets_base import inception_preprocessing
from nets_base.arg_scope import nets_arg_scope
from routines.train import Train

slim = tf.contrib.slim


class TrainCAE(Train):

    def __init__(self, CAE_structure, image_size=299,
                 initial_learning_rate=0.01, **kwargs):
        super(TrainCAE, self).__init__(**kwargs)
        self.initial_learning_rate = initial_learning_rate
        self.image_size = image_size
        self.CAE_structure = CAE_structure

    def get_data(self, tfrecord_dir, batch_size):
        dataset = read_TFRecord.get_split('train', tfrecord_dir)
        self.images_original, _ = load_batch(
            dataset, height=self.image_size, width=self.image_size,
            batch_size=batch_size)
        return dataset

    def compute(self, **kwargs):
        self.compute_reconstruction(self.images_original, **kwargs)

    def compute_reconstruction(self, inputs, dropout_position='fc',
                               dropout_keep_prob=0.5):
        images_corrupted = slim.dropout(
            inputs, keep_prob=dropout_keep_prob, scope='Input/Dropout')

        assert dropout_position in ['fc', 'input']
        dropout_input = dropout_position == 'input'
        self.images = images_corrupted if dropout_input else inputs

        if dropout_input:
            dropout_keep_prob = 1

        self.reconstructions, _ = self.CAE_structure(
            self.images, dropout_keep_prob=dropout_keep_prob)

    def get_total_loss(self):
        self.reconstruction_loss = tf.losses.mean_squared_error(
            self.reconstructions, self.images_original)
        self.total_loss = tf.losses.get_total_loss()
        return self.total_loss

    def get_metric_op(self):
        self.metric_op = None
        return self.metric_op

    def get_summary_op(self):
        tf.summary.scalar(
            'losses/reconstruction', self.reconstruction_loss)
        tf.summary.scalar('losses/total', self.total_loss)
        tf.summary.image('input', self.images)
        tf.summary.image('reconstruction', self.reconstructions)
        self.summary_op = tf.summary.merge_all()

    def get_init_fn(self, checkpoint_dirs):
        return None

    def normal_log_info(self, sess):
        self.loss, _, summaries = self.train_step(
            sess, self.train_op, self.sv.global_step, self.summary_op)
        return summaries

    def final_log_info(self, sess):
        tf.logging.info('Finished training. Final Loss: %s', self.loss)
        tf.logging.info('Saving model to disk now.')


class TrainInceptionCAE(TrainCAE):

    def get_init_fn(self, checkpoint_dirs):
        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                checkpoint_dirs[0], 'inception_v4.ckpt')
        variables_to_restore = tf.get_collection(
            tf.GraphKeys.MODEL_VARIABLES, scope='InceptionV4')
        saver = tf.train.Saver(variables_to_restore)

        def restore(sess):
            saver.restore(sess, checkpoint_path)
        return restore


def train_CAE(CAE_structure, tfrecord_dir, log_dir,
              number_of_steps=None, **kwargs):
    train_CAE = TrainCAE(CAE_structure)
    for key in kwargs.copy():
        if hasattr(train_CAE, key):
            setattr(train_CAE, key, kwargs[key])
            del kwargs[key]
    train_CAE.train(tfrecord_dir, None, log_dir,
                    number_of_steps=number_of_steps, **kwargs)


def evaluate_CAE(tfrecord_dir,
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
            dataset = read_TFRecord.get_split('validation', tfrecord_dir)

            images_original, labels = load_batch(
                dataset, height=image_size, width=image_size,
                batch_size=batch_size)

            images_corrupted = slim.dropout(
                images_original, keep_prob=dropout_keep_prob, scope='Dropout')

            images = images_corrupted if dropout_input else images_original

        if number_of_steps is None:
            number_of_steps = int(np.ceil(dataset.num_samples / batch_size))

        with slim.arg_scope(nets_arg_scope(is_training=False)):
            reconstruction, _ = CAE_structure(images)

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


def reconstruct(image_path, train_dir, CAE_structure, log_dir=None):

    image_size = 299

    with tf.Graph().as_default():

        image_string = tf.gfile.FastGFile(image_path, 'r').read()
        _, image_ext = os.path.splitext(image_path)

        if image_ext in ['.jpg', '.jpeg']:
            image = tf.image.decode_jpg(image_string, channels=3)
        if image_ext == '.png':
            image = tf.image.decode_png(image_string, channels=3)
        else:
            raise ValueError('image format not supported, must be jpg or png')

        processed_image = inception_preprocessing.preprocess_image(
            image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        with slim.arg_scope(nets_arg_scope(is_training=False)):
            reconstructions, _ = CAE_structure(processed_images)
        reconstruction = tf.squeeze(reconstructions)

        tf.summary.image('input', processed_images)
        tf.summary.image('reconstruction', reconstructions)
        summary_op = tf.summary.merge_all()

        if log_dir is not None:
            fw = tf.summary.FileWriter(log_dir)

        checkpoint_path = tf.train.latest_checkpoint(train_dir)
        saver = tf.train.Saver(tf.model_variables())

        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            reconstruction, summaries = sess.run([reconstruction, summary_op])
            if log_dir is not None:
                fw.add_summary(summaries)
            return reconstruction

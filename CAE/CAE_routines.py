"""Implement a shadow CAE that reads raw image as input"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from nets_base import inception_preprocessing
from nets_base.arg_scope import nets_arg_scope

from routines.train import TrainImages
from routines.evaluate import EvaluateImages

slim = tf.contrib.slim


class TrainCAE(TrainImages):

    def __init__(self, CAE_structure, initial_learning_rate=0.01, **kwargs):
        super(TrainCAE, self).__init__(**kwargs)
        self.initial_learning_rate = initial_learning_rate
        self.CAE_structure = CAE_structure

    def decide_used_data(self):
        self.images_original = tf.cond(
            self.training, lambda: self.images_train, lambda: self.images_test)

    def compute(self, **kwargs):
        self.reconstructions = \
            self.compute_reconstruction(self.images_original, **kwargs)

    def compute_reconstruction(self, inputs, dropout_position='fc',
                               dropout_keep_prob=0.7):
        images_corrupted = slim.dropout(
            inputs, keep_prob=dropout_keep_prob, scope='Input/Dropout')

        assert dropout_position in ['fc', 'input']
        dropout_input = dropout_position == 'input'
        self.images_original = inputs
        self.images = images_corrupted if dropout_input else inputs

        if dropout_input:
            dropout_keep_prob = 1

        reconstructions, _ = self.CAE_structure(
            self.images, dropout_keep_prob=dropout_keep_prob)
        return reconstructions

    def get_total_loss(self):
        self.reconstruction_loss = tf.losses.mean_squared_error(
            self.reconstructions, self.images_original)
        self.total_loss = tf.losses.get_total_loss()
        return self.total_loss

    def get_metric_op(self):
        self.metric_op = None
        return self.metric_op

    def get_summary_op(self):
        self.get_batch_norm_summary()
        tf.summary.scalar(
            'losses/reconstruction', self.reconstruction_loss)
        tf.summary.scalar('losses/total', self.total_loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.image('original', self.images_original)
        tf.summary.image('input', self.images)
        tf.summary.image('reconstruction', self.reconstructions)
        self.summary_op = tf.summary.merge_all()

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


class EvaluateCAE(EvaluateImages):

    def __init__(self, CAE_structure, **kwargs):
        super(EvaluateCAE, self).__init__(**kwargs)
        self.CAE_structure = CAE_structure

    def compute(self, **kwargs):
        self.reconstructions = \
            self.compute_reconstruction(self.images, **kwargs)

    def compute_reconstruction(self, inputs, dropout_input=False,
                               dropout_keep_prob=0.5):
        images_corrupted = tf.nn.dropout(
            inputs, keep_prob=dropout_keep_prob, name='Input/Dropout')

        self.images_original = self.images
        self.images = images_corrupted if dropout_input else inputs
        reconstructions, _ = self.CAE_structure(self.images)
        return reconstructions

    def compute_log_data(self):
        self.reconstruction_loss = \
            tf.losses.mean_squared_error(self.reconstructions, self.images)
        self.total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('losses/total', self.total_loss)
        tf.summary.scalar('losses/reconstruction', self.reconstruction_loss)
        tf.summary.image('original', self.images_original)
        tf.summary.image('input', self.images)
        tf.summary.image('reconstruction', self.reconstructions)
        self.summary_op = tf.summary.merge_all()

    def step_log_info(self, sess):
        self.global_step_count, time_elapsed, tensor_values = \
            self.eval_step(
                sess, self.global_step_op,
                self.total_loss, self.summary_op)
        self.loss = tensor_values[0]
        tf.logging.info(
            'global step %s: loss: %.4f (%.2f sec/step)',
            self.global_step_count, self.loss, time_elapsed)
        return self.global_step_count, tensor_values[1]


def reconstruct(image_path, train_dir, CAE_structure, log_dir=None):

    image_size = 299

    with tf.Graph().as_default():

        image_string = tf.gfile.FastGFile(image_path, 'r').read()
        _, image_ext = os.path.splitext(image_path)

        if image_ext in ['.jpg', '.jpeg']:
            image = tf.image.decode_jpeg(image_string, channels=3)
        elif image_ext == '.png':
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

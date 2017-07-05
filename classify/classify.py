from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from datasets import dataset_utils
from nets import inception_v4
from nets_base import inception_preprocessing

slim = tf.contrib.slim


def classify_image(image_path, train_dir, label_dir):

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

        labels_to_names = dataset_utils.read_label_file(label_dir)

        # Create the model, use the default arg scope to
        # configure the batch norm parameters.
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits, endpoints = inception_v4.inception_v4(
                processed_images, num_classes=len(labels_to_names),
                is_training=False)
        probabilities = endpoints['Predictions']

        checkpoint_path = tf.train.latest_checkpoint(train_dir)
        saver = tf.train.Saver(tf.model_variables())

        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)

            probabilities = sess.run(probabilities)
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(
                enumerate(-probabilities), key=lambda x:x[1])]

        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%s]' % (
                  probabilities[index] * 100, labels_to_names[index]))

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import imagenet
from nets import inception_v4
from preprocessing import inception_preprocessing

import dataset_tfr.read_TFRecord as read_TFRecord


slim = tf.contrib.slim


def load_batch(dataset, batch_size=32, height=299, width=299,
               is_training=False):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.

    Returns:
      images: A Tensor of size [batch_size, height, width, 3],
        image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3],
        image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between
        0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])

    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(
        image_raw, height, width, is_training=is_training)

    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
          [image, image_raw, label],
          batch_size=batch_size,
          num_threads=1,
          capacity=2 * batch_size)

    return images, images_raw, labels


def get_init_fn(checkpoints_dir):
    """Returns a function run by the chief worker to
       warm-start the training."""
    checkpoint_exclude_scopes = [
        "InceptionV4/Logits", "InceptionV4/AuxLogits"]

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
        variables_to_restore)


def run(dataset_dir, checkpoints_dir, log_dir):

    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

    image_size = inception_v4.inception_v4.default_image_size

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        dataset = read_TFRecord.get_split('train', dataset_dir)
        images, _, labels = load_batch(
            dataset, height=image_size, width=image_size)

        # Create the model, use the default arg scope to configure the
        # batch norm parameters.
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits, _ = inception_v4.inception_v4(
                images, num_classes=dataset.num_classes, is_training=True)

        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        total_loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total_Loss', total_loss)

        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=log_dir,
            init_fn=get_init_fn(checkpoints_dir),
            number_of_steps=2)

    print('Finished training. Last batch loss %f' % final_loss)


def classify_image(image_path, checkpoints_dir):

    image_size = inception_v4.inception_v4.default_image_size

    with tf.Graph().as_default():
        image_string = tf.gfile.FastGFile(image_path, 'r').read()
        image = tf.image.decode_png(image_string, channels=3)

        processed_image = inception_preprocessing.preprocess_image(
            image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to
        # configure the batch norm parameters.
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits, _ = inception_v4.inception_v4(
                processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
            slim.get_model_variables('InceptionV4'))

        with tf.Session() as sess:
            init_fn(sess)
            np_image, probabilities = sess.run([image, probabilities])
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(
                enumerate(-probabilities), key=lambda x:x[1])]

        plt.figure()
        plt.imshow(np_image.astype(np.uint8))
        plt.axis('off')
        plt.show()

        names = imagenet.create_readable_names_for_imagenet_labels()
        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%s]' % (
                  probabilities[index] * 100, names[index]))

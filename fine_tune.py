from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import os
import time

import tensorflow as tf
from datasets import dataset_utils
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
        dataset, common_queue_capacity=800,
        common_queue_min=400)
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
          capacity=2*batch_size)

    return images, images_raw, labels


def get_init_fn(checkpoints_dir):
    """Returns a function run by the chief worker to
       warm-start the training."""
    checkpoint_exclude_scopes = [
        'InceptionV4/Logits', 'InceptionV4/AuxLogits']

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


def get_variables_to_train(scopes):
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def run(dataset_dir, checkpoints_dir, log_dir, number_of_steps):

    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

    image_size = inception_v4.inception_v4.default_image_size

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        dataset = read_TFRecord.get_split('train', dataset_dir)

        # We don't use the given preprocessed funciton that crops images
        images, _, labels = load_batch(
            dataset, height=image_size, width=image_size,
            batch_size=24, is_training=False)

        # Create the model, use the default arg scope to configure the
        # batch norm parameters.
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits, end_points = inception_v4.inception_v4(
                images, num_classes=dataset.num_classes, is_training=True)

        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        tf.losses.softmax_cross_entropy(one_hot_labels, logits)
        total_loss = tf.losses.get_total_loss()

        # Create the global step for monitoring training
        global_step = tf.train.get_or_create_global_step()

        # Exponentially decaying learning rate
        learning_rate = tf.train.exponential_decay(
            learning_rate=0.005,
            global_step=global_step,
            decay_steps=400,
            decay_rate=0.8,
            staircase=True)

        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        scopes = ['InceptionV4/Mixed_7d', 'InceptionV4/Logits',
                  'InceptionV4/AuxLogits']
        variables_to_train = get_variables_to_train(scopes)
        print(variables_to_train)
        train_op = slim.learning.create_train_op(
            total_loss, optimizer,
            variables_to_train=variables_to_train)

        # The metrics to predict
        mean_logits = tf.reduce_mean(logits)
        predictions = tf.argmax(end_points['Predictions'], 1)
        accuracy, accuracy_update = \
            slim.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update)

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.image('images', images, max_outputs=6)
        tf.summary.histogram('logits', logits)
        my_summary_op = tf.summary.merge_all()

        def train_step(sess, train_op, global_step):
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run(
                [train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time

            tf.logging.info(
                'global step %s: loss: %.4f (%.2f sec/step)',
                global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        # Define the supervisor
        sv = tf.train.Supervisor(
            logdir=log_dir, summary_op=None,
            init_fn=get_init_fn(checkpoints_dir))

        with sv.managed_session() as sess:
            for step in xrange(number_of_steps):
                loss, _ = train_step(sess, train_op, sv.global_step)
                lbs, pds = sess.run([labels, predictions])
                print('labels:', lbs)
                print('predictions:', pds)
                if step % 2 == 0:
                    accuracy_rate, mlg = sess.run([accuracy, mean_logits])
                    tf.logging.info('Current Streaming Accuracy:%s',
                                    accuracy_rate)
                    tf.logging.info('Mean Logits Activation:%s', mlg)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

            tf.logging.info('Finished training. Final Loss: %s', loss)
            tf.logging.info('Final Accuracy: %s', sess.run(accuracy))
            tf.logging.info('Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)


def classify_image(image_path, train_dir, dataset_dir):

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
            _, endpoints = inception_v4.inception_v4(
                processed_images, num_classes=11, is_training=False)
        probabilities = endpoints['Predictions']

        checkpoint_path = tf.train.latest_checkpoint(train_dir)

        init_fn = slim.assign_from_checkpoint_fn(
            checkpoint_path,
            slim.get_model_variables('InceptionV4'))

        with tf.Session() as sess:
            init_fn(sess)
            probabilities = sess.run(probabilities)
            print(probabilities)
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(
                enumerate(-probabilities), key=lambda x:x[1])]

            img = tf.summary.image('image', tf.expand_dims(image, 0))
            fw = tf.summary.FileWriter('log')
            fw.add_summary(sess.run(img))

        labels_to_names = dataset_utils.read_label_file(dataset_dir)
        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%s]' % (
                  probabilities[index] * 100, labels_to_names[index]))

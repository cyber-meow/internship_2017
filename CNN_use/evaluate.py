from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import time

import numpy as np
import tensorflow as tf
from nets import inception_v4

import data.images.read_TFRecord as read_TFRecord
from data.images.load_batch import load_batch

slim = tf.contrib.slim


def eval_step(sess, fw, accuracy, accuracy_summary, global_step_op, *args):

    tensors_to_run = [accuracy, accuracy_summary, global_step_op]
    tensors_to_run.extend(args)

    start_time = time.time()
    tensor_values = sess.run(tensors_to_run)
    time_elapsed = time.time() - start_time

    accuracy_rate = tensor_values[0]
    accuracy_summary_serialized = tensor_values[1]
    global_step_count = tensor_values[2]

    fw.add_summary(accuracy_summary_serialized, global_step=global_step_count)

    tf.logging.info(
        'global step %s: accurarcy: %.4f (%.2f sec/step)',
        global_step_count, accuracy_rate, time_elapsed)

    return tensor_values


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

        with tf.name_scope('data_provider'):
            dataset = read_TFRecord.get_split('validation', dataset_dir)
            images, _, labels = load_batch(
                dataset, height=image_size, width=image_size,
                batch_size=batch_size, is_training=False)

        if number_of_steps is None:
            number_of_steps = int(np.ceil(dataset.num_samples / batch_size))

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits, endpoints = inception_v4.inception_v4(
                images, num_classes=dataset.num_classes, is_training=False)

        # Define metric
        predictions = tf.argmax(endpoints['Predictions'], 1)
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(predictions, labels), tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

        # Define global step to be show in tensorboard
        global_step = tf.train.get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step+1)

        # File writer for the tensorboard
        fw = tf.summary.FileWriter(log_dir)

        checkpoint_path = tf.train.latest_checkpoint(train_dir)
        saver = tf.train.Saver(tf.model_variables())

        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):
                sess.run(tf.variables_initializer([global_step]))
                sess.run(tf.local_variables_initializer())
                saver.restore(sess, checkpoint_path)

                for step in xrange(number_of_steps-1):
                    eval_step(sess, fw, accuracy,
                              accuracy_summary, global_step_op)

                global_step_count, labels, predictions, images = eval_step(
                    sess, fw, accuracy, accuracy_summary, global_step_op,
                    labels, predictions, images)[2:]
                true_names = [
                    dataset.labels_to_names[i] for i in labels]
                predicted_names = [
                    dataset.labels_to_names[i] for i in predictions]

                tf.logging.info('Information for the last batch')
                tf.logging.info('Ground Truth: [%s]', true_names)
                tf.logging.info('Prediciotn: [%s]', predicted_names)

                with tf.name_scope('last_images'):
                    for i in range(batch_size):
                        image_pl = tf.placeholder(
                            dtype=tf.float32,
                            shape=(1, image_size, image_size, 3))
                        image_summary = tf.summary.image(
                            'image_true_{}_predicted_{}'.format(
                                true_names[i], predicted_names[i]), image_pl)
                        fw.add_summary(
                            sess.run(image_summary,
                                     feed_dict={image_pl: [images[i]]}),
                            global_step=global_step_count)

                tf.logging.info('Finished evaluation.')

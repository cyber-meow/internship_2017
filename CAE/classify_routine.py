"""Classify images using intern representations"""

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
    tensor_values = sess.run(tensors_to_run, feed_dict={'training:0': True})
    time_elapsed = time.time() - start_time

    total_loss = tensor_values[0]
    global_step_count = tensor_values[1]

    tf.logging.info(
        'global step %s: loss: %.4f (%.2f sec/step)',
        global_step_count, total_loss, time_elapsed)

    return tensor_values


def train_classify(dataset_dir,
                   checkpoints_dir,
                   log_dir,
                   CAE_structure,
                   endpoint,
                   number_of_steps=None,
                   number_of_epochs=5,
                   batch_size=24,
                   save_summaries_step=5,
                   do_test=False,
                   dropout_keep_prob=0.8,
                   initial_learning_rate=0.005,
                   lr_decay_steps=100,
                   lr_decay_rate=0.8):

    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

    image_size = 299

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        with tf.name_scope('data_provider'):
            dataset = read_TFRecord.get_split('train', dataset_dir)
            images_train, _, labels_train = load_batch(
                dataset, height=image_size, width=image_size,
                batch_size=batch_size, is_training=False)

            dataset_test = read_TFRecord.get_split('validation', dataset_dir)
            images_test, _, labels_test = load_batch(
                dataset_test, height=image_size, width=image_size,
                batch_size=batch_size, is_training=False)

        training = tf.placeholder(tf.bool, shape=(), name='training')
        images = tf.cond(training, lambda: images_train, lambda: images_test)
        labels = tf.cond(training, lambda: labels_train, lambda: labels_test)

        if number_of_steps is None:
            number_of_steps = int(np.ceil(
                dataset.num_samples * number_of_epochs / batch_size))

        if CAE_structure is not None:
            with slim.arg_scope(nets_arg_scope()):
                net, _ = CAE_structure(
                    images, dropout_keep_prob=1,
                    is_training=training, final_endpoint=endpoint)
        else:
            net = images

        representation_shape = tf.shape(net)

        net = slim.dropout(net, dropout_keep_prob,
                           scope='Dropout_PreLogits', is_training=training)
        net = slim.flatten(net, scope='PreLogitsFlatten')
        logits = slim.fully_connected(
            net, dataset.num_classes, activation_fn=None, scope='Logits')

        one_hot_labels = tf.one_hot(labels, dataset.num_classes)
        tf.losses.softmax_cross_entropy(one_hot_labels, logits)
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
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='Logits'))

        # The metrics to predict
        predictions = tf.argmax(tf.nn.softmax(logits), 1)
        accuracy, accuracy_update = tf.metrics.accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update)

        accuracy_test = tf.reduce_mean(tf.cast(
            tf.equal(predictions, labels), tf.float32))

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
        tf.summary.histogram('logits', logits)
        tf.summary.scalar('losses/train/total_loss', total_loss)
        tf.summary.scalar('accuracy/train/streaming', accuracy)
        tf.summary.image('train', images)

        summary_op = tf.summary.merge_all()

        ac_test_summary = tf.summary.scalar('accuracy/test', accuracy_test)
        ls_test_summary = tf.summary.scalar(
            'losses/test/total_loss', total_loss)
        imgs_test_summary = tf.summary.image('test', images)
        test_summary_op = tf.summary.merge(
            [ac_test_summary, ls_test_summary, imgs_test_summary])

        checkpoint_path = tf.train.latest_checkpoint(checkpoints_dir)
        variables_to_restore = tf.get_collection(
            tf.GraphKeys.MODEL_VARIABLES, scope='CAE')
        if CAE_structure is None:
            init_fn = None
        else:
            init_fn = slim.assign_from_checkpoint_fn(
                checkpoint_path, variables_to_restore)

        sv = tf.train.Supervisor(
            logdir=log_dir, summary_op=None,
            init_fn=init_fn)

        with sv.managed_session() as sess:
            tf.logging.info('intern representation shape: %s',
                            sess.run([representation_shape]))
            for step in xrange(number_of_steps):
                if (step+1) % save_summaries_step == 0:
                    loss, _, _, summaries, accuracy_rate = train_step(
                        sess, train_op, sv.global_step, metrics_op,
                        summary_op, accuracy)
                    tf.logging.info('Current Streaming Accuracy:%s',
                                    accuracy_rate)
                    sv.summary_computed(sess, summaries)
                    if do_test:
                        ls, acu, summaries_test = sess.run(
                            [total_loss, accuracy_test, test_summary_op],
                            feed_dict={training: False})
                        tf.logging.info('Current Test Loss: %s', ls)
                        tf.logging.info('Current Test Accuracy: %s', acu)
                        sv.summary_computed(sess, summaries_test)
                else:
                    loss = train_step(
                        sess, train_op, sv.global_step, metrics_op)[0]

            tf.logging.info('Finished training. Final Loss: %s', loss)
            tf.logging.info('Final Accuracy: %s', sess.run(accuracy))
            tf.logging.info('Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import time

import abc

import numpy as np
import tensorflow as tf
from nets import inception_v4

import data.images.read_TFRecord as read_TFRecord
from data.images.load_batch import load_batch
from nets_base.arg_scope import nets_arg_scope

slim = tf.contrib.slim


class classify_evaluate(object):

    __meta_class__ = abc.ABCMeta

    def __init__(self, image_size=299):
        self._image_size = image_size

    @property
    def image_size(self):
        return self._image_size

    @abc.abstractmethod
    def get_data(self, split_name, tfrecord_dir, batch_size):
        pass

    @abc.abstractmethod
    def compute_logits(self, inputs):
        pass

    @abc.abstractmethod
    def init_model(self, sess, checkpoint_dirs):
        pass

    def eval_step(self, sess, *args):

        tensors_to_run = [self.accuracy, self.accuracy_summary,
                          self.global_step_op]
        tensors_to_run.extend(args)

        start_time = time.time()
        tensor_values = sess.run(tensors_to_run)
        time_elapsed = time.time() - start_time

        accuracy_rate = tensor_values[0]
        accuracy_summary_serialized = tensor_values[1]
        global_step_count = tensor_values[2]

        self.fw.add_summary(accuracy_summary_serialized,
                            global_step=global_step_count)

        tf.logging.info(
            'global step %s: accurarcy: %.4f (%.2f sec/step)',
            global_step_count, accuracy_rate, time_elapsed)

        return tensor_values

    def evaluate(self,
                 tfrecord_dir,
                 checkpoint_dirs,
                 log_dir,
                 number_of_steps=None,
                 batch_size=12,
                 split_name='validation',
                 image_size=299):

        if not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)

        if not isinstance(checkpoint_dirs, (tuple, list)):
            checkpoint_dirs = [checkpoint_dirs]

        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)

            with tf.name_scope('Data_provider'):
                images, labels, dataset = self.get_data(
                    tfrecord_dir, split_name, batch_size)

            if number_of_steps is None:
                number_of_steps = int(np.ceil(dataset.num_samples/batch_size))

            with slim.arg_scope(nets_arg_scope(is_training=False)):
                logits = self.compute_logits(images, dataset.num_classes)

            # Define metric
            predictions = tf.argmax(tf.nn.softmax(logits), 1)
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.equal(predictions, labels), tf.float32))
            self.accuracy_summary = tf.summary.scalar(
                'accuracy', self.accuracy)

            # Define global step to be show in tensorboard
            global_step = tf.train.get_or_create_global_step()
            self.global_step_op = tf.assign(global_step, global_step+1)

            # File writer for the tensorboard
            self.fw = tf.summary.FileWriter(log_dir)

            with tf.Session() as sess:
                with slim.queues.QueueRunners(sess):
                    sess.run(tf.variables_initializer([global_step]))
                    sess.run(tf.local_variables_initializer())
                    self.init_model(sess, checkpoint_dirs)

                    for step in xrange(number_of_steps-1):
                        self.eval_step(sess)

                    global_step_count, labels, predictions, images = \
                        self.eval_step(sess, labels, predictions, images)[2:]

                    true_names = [
                        dataset.labels_to_names[i] for i in labels]
                    predicted_names = [
                        dataset.labels_to_names[i] for i in predictions]

                    tf.logging.info('Information for the last batch')
                    tf.logging.info('Ground Truth: [%s]', true_names)
                    tf.logging.info('Prediciotn: [%s]', predicted_names)

                    with tf.name_scope('Last_images'):
                        for i in range(batch_size):
                            image_pl = tf.placeholder(
                                dtype=tf.float32,
                                shape=(1, image_size, image_size, 3))
                            image_summary = tf.summary.image(
                                'image_true_{}_predicted_{}'.format(
                                    true_names[i], predicted_names[i]),
                                image_pl)
                            self.fw.add_summary(
                                sess.run(image_summary,
                                         feed_dict={image_pl: [images[i]]}),
                                global_step=global_step_count)

                    tf.logging.info('Finished evaluation.')


class classify_evaluate_CNN(classify_evaluate):

    def get_data(self, tfrecord_dir, split_name, batch_size):
        dataset = read_TFRecord.get_split(split_name, tfrecord_dir)
        images, labels = load_batch(
            dataset, height=self.image_size,
            width=self.image_size, batch_size=batch_size)
        return images, labels, dataset

    def compute_logits(self, inputs, num_classes):
        logits, _ = inception_v4.inception_v4(
            inputs, num_classes=num_classes, is_training=False)
        return logits

    def init_model(self, sess, checkpoint_dirs):
        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        saver = tf.train.Saver(tf.model_variables())
        saver.restore(sess, checkpoint_path)


classify_evaluate_CNN_fn = classify_evaluate_CNN().evaluate

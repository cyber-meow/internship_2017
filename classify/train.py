from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import abc

import tensorflow as tf
from nets import inception_v4

import data.images.read_TFRecord as read_TFRecord
from data.images.load_batch import load_batch
from routines.train import Train

slim = tf.contrib.slim


class TrainClassify(Train):

    __meta_class__ = abc.ABCMeta

    def __init__(self, image_size=299, **kwargs):
        super(TrainClassify, self).__init__(**kwargs)
        self._image_size = image_size

    @property
    def image_size(self):
        return self._image_size

    def get_data(self, split_name, tfrecord_dir, batch_size):
        self.dataset_train = read_TFRecord.get_split('train', tfrecord_dir)
        images_train, labels_train = load_batch(
            self.dataset_train, height=self.image_size,
            width=self.image_size, batch_size=batch_size)
        self.dataset_test = read_TFRecord.get_split('validation', tfrecord_dir)
        images_test, labels_test = load_batch(
            self.dataset_test, height=self.image_size,
            width=self.image_size, batch_size=batch_size)
        return self.dataset_train

    def decide_used_data(self):
        self.images = tf.cond(
            self.training, lambda: self.images_train, lambda: self.images_test)
        self.labels = tf.cond(
            self.training, lambda: self.labels_train, lambda: self.labels_test)

    def compute(self, **kwargs):
        self.compute_logits(self.images, **kwargs)

    @abc.abstractmethod
    def compute_logits(self, inputs):
        pass

    def get_total_loss(self):
        num_classes = self.dataset_train.num_classes
        one_hot_labels = tf.one_hot(self.labels, num_classes)
        self.cross_entropy_loss = \
            tf.losses.softmax_cross_entropy(one_hot_labels, self.logits)
        self.total_loss = tf.losses.get_total_loss()
        return self.total_loss

    def get_metric_op(self):
        self.predictions = tf.argmax(tf.nn.softmax(self.logits), 1)
        self.accuracy, self.accuracy_update = \
            tf.metrics.accuracy(self.predictions, self.labels)
        self.metric_op = tf.group(self.accuracy_update)
        return self.metric_op

    def get_test_metrics(self):
        self.accuracy_test = tf.reduce_mean(tf.cast(
            tf.equal(self.predictions, self.labels), tf.float32))
        return self.accuracy_test

    def get_summary_op(self):
        self.get_batch_norm_summary()
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.histogram('logits', self.logits)
        tf.summary.scalar('losses/train/cross_entropy',
                          self.cross_entropy_loss)
        tf.summary.scalar('losses/train/total', self.total_loss)
        tf.summary.scalar('accuracy/train/streaming', self.accuracy)
        tf.summary.image('train', self.images, max_outputs=4)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        # Summaries for the test part
        ac_test_summary = tf.summary.scalar(
            'accuracy/test', self.accuracy_test)
        ls_test_summary = tf.summary.scalar(
            'losses/test/total_loss', self.total_loss)
        imgs_test_summary = tf.summary.image(
            'test', self.images, max_outputs=4)
        self.test_summary_op = tf.summary.merge(
            [ac_test_summary, ls_test_summary, imgs_test_summary])
        return self.test_summary_op

    def normal_log_info(self, sess):
        loss, _, _, summaries, accuracy_rate = \
            self.train_step(
                sess, self.train_op, self.sv.global_step, self.metric_op,
                self.summary_op, self.accuracy)
        tf.logging.info('Current Streaming Accuracy:%s', accuracy_rate)
        return summaries

    def test_log_info(self, sess):
        ls, acu, summaries_test = sess.run(
            [self.total_loss, self.accuracy_test, self.test_summary_op],
            feed_dict={self.training: False})
        tf.logging.info('Current Test Loss: %s', ls)
        tf.logging.info('Current Test Accuracy: %s', acu)
        return summaries_test

    def final_log_info(self, sess):
        tf.logging.info('Finished training. Final Loss: %s', self.loss)
        tf.logging.info('Final Accuracy: %s', sess.run(self.accuracy))
        tf.logging.info('Saving model to disk now.')


class TrainClassifyInception(TrainClassify):

    @property
    def default_trainable_scopes(self):
        return ['InceptionV4/Mixed_7d', 'InceptionV4/Logits',
                'InceptionV4/AuxLogits']

    def compute_logits(self, inputs, **kwargs):
        self.logits, _ = inception_v4.inception_v4(
            inputs, num_classes=self.dataset_train.num_classes,
            is_training=self.training, **kwargs)

    def get_init_fn(self, checkpoint_dirs):
        """Returns a function run by the chief worker to
           warm-start the training."""
        checkpoint_exclude_scopes = [
            'InceptionV4/Logits', 'InceptionV4/AuxLogits']

        variables_to_restore = []
        for var in tf.model_variables():
            excluded = False
            for exclusion in checkpoint_exclude_scopes:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                checkpoint_dirs[0], 'inception_v4.ckpt')

        return slim.assign_from_checkpoint_fn(
            checkpoint_path, variables_to_restore)


fine_tune_inception = TrainClassifyInception().train

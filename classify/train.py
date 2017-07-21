from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import abc

import tensorflow as tf
from nets import inception_v4

from routines.train import TrainImages

slim = tf.contrib.slim


class TrainClassify(TrainImages):

    __meta_class__ = abc.ABCMeta

    def decide_used_data(self):
        self.images = tf.cond(
            self.training, lambda: self.images_train, lambda: self.images_test)
        self.labels = tf.cond(
            self.training, lambda: self.labels_train, lambda: self.labels_test)

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.images, self.dataset_train.num_classes, **kwargs)

    @abc.abstractmethod
    def compute_logits(self, inputs, num_classes):
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
        self.accuracy_no_streaming = tf.reduce_mean(tf.cast(
            tf.equal(self.predictions, self.labels), tf.float32))
        return self.metric_op

    def get_summary_op(self):
        self.get_batch_norm_summary()
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.histogram('logits', self.logits)
        tf.summary.scalar('losses/train/cross_entropy',
                          self.cross_entropy_loss)
        tf.summary.scalar('losses/train/total', self.total_loss)
        tf.summary.scalar('accuracy/train', self.accuracy_no_streaming)
        tf.summary.scalar('accuracy/train/streaming', self.accuracy)
        tf.summary.image('train', self.images, max_outputs=4)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        # Summaries for the test part
        ac_test_summary = tf.summary.scalar(
            'accuracy/test', self.accuracy_no_streaming)
        ls_test_summary = tf.summary.scalar(
            'losses/test/total', self.total_loss)
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

    def test_log_info(self, sess, test_use_batch):
        ls, acu, summaries_test = sess.run(
            [self.total_loss, self.accuracy_no_streaming,
             self.test_summary_op],
            feed_dict={self.training: False,
                       self.batch_stat: test_use_batch})
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

    def compute_logits(self, inputs, num_classes, **kwargs):
        logits, _ = inception_v4.inception_v4(
            inputs, num_classes=num_classes,
            is_training=self.training, **kwargs)
        return logits

    def get_init_fn(self, checkpoint_dirs):
        """Returns a function run by the chief worker to
           warm-start the training."""
        checkpoint_exclude_scopes = [
            'InceptionV4/Logits', 'InceptionV4/AuxLogits']
        variables_to_restore = self.get_variables_to_restore(
            scopes=None, exclude=checkpoint_exclude_scopes)

        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                checkpoint_dirs[0], 'inception_v4.ckpt')

        return slim.assign_from_checkpoint_fn(
            checkpoint_path, variables_to_restore)


def fine_tune_inception(tfrecord_dir,
                        checkpoint_dirs,
                        log_dir,
                        number_of_steps=None,
                        image_size=299,
                        **kwargs):
    fine_tune = TrainClassifyInception(image_size)
    for key in kwargs.copy():
        if hasattr(fine_tune, key):
            setattr(fine_tune, key, kwargs[key])
            del kwargs[key]
    fine_tune.train(
        tfrecord_dir, checkpoint_dirs, log_dir,
        number_of_steps=number_of_steps, **kwargs)


class TrainClassifyCNN(TrainClassify):

    def __init__(self, CNN_structure, **kwargs):
        super(TrainClassifyCNN, self).__init__(**kwargs)
        self.CNN_structure = CNN_structure

    def compute_logits(self, inputs, num_classes,
                       dropout_keep_prob=0.8, endpoint=None,
                       per_layer_dropout=None):
        if self.CNN_structure is not None:
            if endpoint is not None:
                if per_layer_dropout is not None:
                    net = self.CNN_structure(
                        inputs, final_endpoint=endpoint,
                        per_layer_dropout=per_layer_dropout,
                        dropout_keep_prob=dropout_keep_prob)
                else:
                    net = self.CNN_structure(inputs, final_endpoint=endpoint)
            else:
                if per_layer_dropout is not None:
                    net = self.CNN_structure(
                        inputs,
                        per_layer_dropout=per_layer_dropout,
                        dropout_keep_prob=dropout_keep_prob)
                else:
                    net = self.CNN_structure(inputs)
        else:
            net = inputs
        net = slim.dropout(net, dropout_keep_prob, scope='PreLogitsDropout')
        net = slim.flatten(net, scope='PreLogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits

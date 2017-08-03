from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

from nets_base import inception_v4
from images.basics import TrainImages, EvaluateImages
from classify.train import TrainClassify, TrainClassifyCNN
from classify.evaluate import EvaluateClassify, EvaluateClassifyCNN

slim = tf.contrib.slim


class TrainClassifyImages(TrainImages, TrainClassify):

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.images, self.dataset_train.num_classes, **kwargs)

    def get_summary_op(self):
        super(TrainClassifyImages, self).get_summary_op()
        tf.summary.image('train', self.images)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        summary_op = super(TrainClassifyImages, self).get_test_summary_op()
        images_test_summary = tf.summary.image('test', self.images)
        self.test_summary_op = tf.summary.merge(
            [summary_op, images_test_summary])
        return self.test_summary_op


class EvaluateClassifyImages(EvaluateImages, EvaluateClassify):

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.images, self.dataset.num_classes, **kwargs)

    def last_step_log_info(self, sess, batch_size):
        start_time = time.time()
        global_step_count, accuracy_rate, ac_summary, labels, \
            predictions, images = sess.run([
                self.global_step_op, self.accuracy,
                self.accuracy_summary,
                self.labels, self.predictions, self.images])
        time_elapsed = time.time() - start_time

        tf.logging.info(
            'global step %s: accurarcy: %.4f (%.2f sec/step)',
            global_step_count, accuracy_rate, time_elapsed)

        dataset = self.dataset
        true_names = [dataset.labels_to_names[i] for i in labels]
        predicted_names = [dataset.labels_to_names[i] for i in predictions]

        if batch_size > 20:
            batch_size = 20
        image_size = self.image_size

        tf.logging.info('Information for the last batch')
        tf.logging.info('Ground Truth: [%s]', true_names[:batch_size])
        tf.logging.info('Prediciotn: [%s]', predicted_names[:batch_size])

        if hasattr(self, 'fw'):
            with tf.name_scope('last_images'):
                for i in range(batch_size):
                    image_pl = tf.placeholder(
                        dtype=tf.float32,
                        shape=(1, image_size, image_size, self.channels))
                    image_summary = tf.summary.image(
                        'image_true_{}_predicted_{}'.format(
                            true_names[i], predicted_names[i]), image_pl)
                    self.fw.add_summary(
                        sess.run(image_summary,
                                 feed_dict={image_pl: [images[i]]}),
                        global_step=self.global_step_count)
            self.fw.add_summary(ac_summary, global_step=global_step_count)


class TrainClassifyInception(TrainClassifyImages):

    @property
    def default_trainable_scopes(self):
        return ['InceptionV4/Mixed_7d', 'InceptionV4/Logits']

    def compute_logits(self, inputs, num_classes, **kwargs):
        logits, _ = inception_v4.inception_v4(
            inputs, num_classes=num_classes,
            is_training=self.batch_stat, **kwargs)
        return logits

    def get_init_fn(self, checkpoint_dirs):
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
                        checkpoint_dir,
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
        tfrecord_dir, checkpoint_dir, log_dir,
        number_of_steps=number_of_steps, **kwargs)


class EvaluateClassifyInception(EvaluateClassifyImages):

    def compute_logits(self, inputs, num_classes):
        logits, _ = inception_v4.inception_v4(
            inputs, num_classes=num_classes, is_training=False)
        return logits


class TrainClassifyImagesCNN(TrainClassifyCNN, TrainClassifyImages):
    pass


class EvaluateClassifyImagesCNN(EvaluateClassifyCNN, EvaluateClassifyImages):
    pass


class TrainClassifyImagesCAE(TrainClassifyImages):

    def __init__(self, CAE_structure, endpoint='Middle', **kwargs):
        super(TrainClassifyImagesCAE, self).__init__(**kwargs)
        self.CAE_structure = CAE_structure
        self.endpoint = endpoint

    @property
    def default_trainable_scopes(self):
        return ['Logits']

    def compute_logits(self, inputs, num_classes,
                       dropout_keep_prob=0.8, do_avg=False):
        net = self.CAE_structure(
            inputs, dropout_keep_prob=1, final_endpoint=self.endpoint)
        net = slim.dropout(net, dropout_keep_prob, scope='PreLogitsDropout')
        if do_avg:
            net = slim.avg_pool2d(
                net, net.get_shape()[1:3], padding='VALID',
                scope='PreLogitsAvgPool')
        print('Representation shape', net.get_shape())
        net = slim.flatten(net, scope='PreLogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits

    def get_init_fn(self, checkpoint_dirs):
        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        assert checkpoint_path is not None
        variables_to_restore = self.get_variables_to_restore(['CAE'])
        return slim.assign_from_checkpoint_fn(
            checkpoint_path, variables_to_restore)


class EvaluateClassifyImagesCAE(EvaluateClassifyImages):

    def __init__(self, CAE_structure, endpoint='Middle', **kwargs):
        super(EvaluateClassifyImagesCAE, self).__init__(**kwargs)
        self.CAE_structure = CAE_structure
        self.endpoint = endpoint

    def compute_logits(self, inputs, num_classes):
        net = self.CAE_structure(
            inputs, final_endpoint=self.endpoint)
        net = slim.flatten(net, scope='PreLogitsFlatten')
        self.logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return self.logits

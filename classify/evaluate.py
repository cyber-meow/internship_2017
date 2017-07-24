from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf
from nets import inception_v4

from routines.evaluate import EvaluateImages

slim = tf.contrib.slim


class EvaluateClassify(EvaluateImages):

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.images, self.dataset.num_classes, **kwargs)

    @abc.abstractmethod
    def compute_logits(self, inputs):
        pass

    def compute_log_data(self):
        self.predictions = tf.argmax(tf.nn.softmax(self.logits), 1)
        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(self.predictions, self.labels), tf.float32))
        self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)

    def step_log_info(self, sess):
        self.global_step_count, time_elapsed, tensor_values = \
            self.eval_step(
                sess, self.global_step_op,
                self.accuracy, self.accuracy_summary)
        self.accuracy_rate = tensor_values[0]
        tf.logging.info(
            'global step %s: accurarcy: %.4f (%.2f sec/step)',
            self.global_step_count, self.accuracy_rate, time_elapsed)
        return self.global_step_count, tensor_values[1]

    def last_step_log_info(self, sess, batch_size):
        self.global_step_count, time_elapsed, tensor_values = \
            self.eval_step(
                sess, self.global_step_op, self.accuracy,
                self.accuracy_summary,
                self.labels, self.predictions, self.images)

        self.accuracy_rate = tensor_values[0]
        tf.logging.info(
            'global step %s: accurarcy: %.4f (%.2f sec/step)',
            self.global_step_count, self.accuracy_rate, time_elapsed)

        labels, predictions, images = tensor_values[2:]
        dataset = self.dataset
        true_names = [dataset.labels_to_names[i] for i in labels]
        predicted_names = [dataset.labels_to_names[i] for i in predictions]

        if hasattr(self, 'fw'):

            tf.logging.info('Information for the last batch')
            tf.logging.info('Ground Truth: [%s]', true_names)
            tf.logging.info('Prediciotn: [%s]', predicted_names)

            with tf.name_scope('last_images'):
                for i in range(batch_size):
                    image_pl = tf.placeholder(
                        dtype=tf.float32,
                        shape=(1, self.image_size, self.image_size, 3))
                    image_summary = tf.summary.image(
                        'image_true_{}_predicted_{}'.format(
                            true_names[i], predicted_names[i]), image_pl)
                    self.fw.add_summary(
                        sess.run(image_summary,
                                 feed_dict={image_pl: [images[i]]}),
                        global_step=self.global_step_count)
        return self.global_step_count, tensor_values[1]


class EvaluateClassifyCNN(EvaluateClassify):

    def __init__(self, CNN_structure, **kwargs):
        super(EvaluateClassifyCNN, self).__init__(**kwargs)
        self.CNN_structure = CNN_structure

    def compute_logits(self, inputs, num_classes,
                       dropout_keep_prob=0.8, endpoint=None):
        if self.CNN_structure is not None:
            if endpoint is not None:
                net = self.CNN_structure(inputs, final_endpoint=endpoint)
            else:
                net = self.CNN_structure(inputs)
        else:
            net = inputs
        net = slim.dropout(net, dropout_keep_prob, scope='PreLogitsDropout')
        net = slim.flatten(net, scope='PreLogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits


class EvaluateClassifyInception(EvaluateClassify):

    def compute_logits(self, inputs, num_classes):
        logits, _ = inception_v4.inception_v4(
            inputs, num_classes=num_classes, is_training=False)
        return logits


evaluate_classify_inception = EvaluateClassifyInception().evaluate

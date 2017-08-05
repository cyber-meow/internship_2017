from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from data.mfcc import load_batch_mfcc, get_split_mfcc
from routines.evaluate import Evaluate


class StoringValuesTest(Evaluate):

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):

        self.dataset = get_split_mfcc(split_name, tfrecord_dir)
        self.mfccs, self.labels = load_batch_mfcc(
            self.dataset, batch_size=batch_size, shuffle=True)

        self.all_mfccs, self.all_labels = load_batch_mfcc(
            self.dataset, shuffle=True,
            batch_size=self.dataset.num_samples)

        # The tfrecords are then only read once (during initialization)
        self.all_mfccs = tf.Variable(self.all_mfccs, trainable=False)
        self.all_labels = tf.Variable(self.all_labels, trainable=False)

        return self.dataset

    def compute(self):
        pass

    def init_model(self, sess, checkpoint_dirs):
        sess.run(tf.variables_initializer([self.all_mfccs, self.all_labels]))

    def step_log_info(self, sess):
        labels, all_labels = sess.run([self.labels, self.all_labels])
        tf.logging.info('label normal: %s', labels[:5])
        tf.logging.info('label all: %s', all_labels[:5])
        return None, None


class KNNTest(StoringValuesTest):

    def compute(self, K=10):
        predicted_labels = []
        n_samples = self.dataset.num_samples
        for i in range(self.mfccs.get_shape()[0]):
            mfcc = tf.tile(
                tf.expand_dims(self.mfccs[i], 0),
                [n_samples, 1, 1, 1])
            mfcc = tf.reshape(mfcc, [n_samples, -1])
            all_mfccs = tf.reshape(self.all_mfccs, [n_samples, -1])
            distances = tf.negative(tf.sqrt(
                tf.reduce_sum(tf.square(mfcc-all_mfccs), axis=1)))
            print(distances.get_shape())
            values, indices = tf.nn.top_k(distances, k=K)
            print(indices.get_shape())
            print(self.all_labels.get_shape())
            nn = tf.stack([self.all_labels[indices[j]] for j in range(K)], 0)
            y, idx, count = tf.unique_with_counts(nn)
            predicted_labels.append(y[tf.argmax(count, 0)])
        self.predicted_labels = tf.stack(predicted_labels, 0)

    def step_log_info(self, sess):
        predicted_labels, true_labels = \
            sess.run([self.predicted_labels, self.labels])
        tf.logging.info('predicted labels: %s', predicted_labels)
        tf.logging.info('true labels: %s', true_labels)
        return None, None

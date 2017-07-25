from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

from classify import TrainClassify
from data.mfcc_lips import load_batch_mfcc_lips, get_split_mfcc_lips

from audio.classify_routines import TrainClassifyAudio, EvaluateClassifyAudio
from audio.classify_routines import CNN_mfcc

from video.classify_routines import TrainClassifyVideo, EvaluateClassifyVideo
from video.classify_routines import CNN_lips5

slim = tf.contrib.slim


class TrainClassifyAudioAll(TrainClassifyAudio):

    def get_data(self, tfrecord_dir, batch_size):
        self.dataset_train = get_split_mfcc_lips('train_all', tfrecord_dir)
        self.mfccs_train, _, self.labels_train = load_batch_mfcc_lips(
            self.dataset_train, batch_size=batch_size)
        self.dataset_test = get_split_mfcc_lips('validation', tfrecord_dir)
        self.mfccs_test, _, self.labels_test = load_batch_mfcc_lips(
            self.dataset_test, batch_size=batch_size, is_training=False)
        return self.dataset_train


class EvaluateClassifyAudioAll(EvaluateClassifyAudio):

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_mfcc_lips(split_name, tfrecord_dir)
        self.mfccs, _, self.labels = load_batch_mfcc_lips(
            self.dataset, batch_size=batch_size,
            shuffle=shuffle, is_training=False)
        return self.dataset


class TrainClassifyVideo20(TrainClassifyVideo):

    def get_data(self, tfrecord_dir, batch_size):
        self.dataset_train = get_split_mfcc_lips('train_AT', tfrecord_dir)
        _, self.videos_train, self.labels_train = load_batch_mfcc_lips(
            self.dataset_train, batch_size=batch_size)
        self.dataset_test = get_split_mfcc_lips('validation', tfrecord_dir)
        _, self.videos_test, self.labels_test = load_batch_mfcc_lips(
            self.dataset_test, batch_size=batch_size, is_training=False)
        return self.dataset_train


class EvaluateClassifyVideo20(EvaluateClassifyVideo):

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_mfcc_lips(split_name, tfrecord_dir)
        _, self.videos, self.labels = load_batch_mfcc_lips(
            self.dataset, batch_size=batch_size,
            shuffle=shuffle, is_training=False)
        return self.dataset


class TrainTransfer(TrainClassify):

    default_trainable_scopes = ['Main']

    def __init__(self, audio_structure='', video_structure='', **kwargs):
        super(TrainTransfer, self).__init__(**kwargs)
        if audio_structure == '':
            self.audio_structure = CNN_mfcc
        else:
            self.audio_structure = audio_structure
        if video_structure == '':
            self.video_structure = CNN_lips5
        else:
            self.video_structure = video_structure

    def get_data(self, tfrecord_dir, batch_size):

        self.dataset_train_AT = get_split_mfcc_lips('train_AT', tfrecord_dir)
        _, self.videos_train, self.labels_train_AT = load_batch_mfcc_lips(
            self.dataset_train_AT, batch_size=batch_size)

        self.dataset_train_UZ = get_split_mfcc_lips('train_UZ', tfrecord_dir)
        self.mfccs_train, _, self.labels_train_UZ = load_batch_mfcc_lips(
            self.dataset_train_UZ, batch_size=batch_size, is_training=False)

        self.dataset_test = get_split_mfcc_lips('validation', tfrecord_dir)
        _, self.videos_test, self.labels_test = load_batch_mfcc_lips(
            self.dataset_test, batch_size=batch_size, is_training=False)

        self.all_mfccs, self.all_videos, all_labels = load_batch_mfcc_lips(
            self.dataset_train_AT, shuffle=False,
            batch_size=self.dataset_train_AT.num_samples)

        self.all_labels = tf.Variable(all_labels, trainable=False)
        self.dataset_train = self.dataset_train_AT

        return self.dataset_train_AT

    def decide_used_data(self):
        self.videos = tf.cond(
            self.training, lambda: self.videos_train,
            lambda: self.videos_test)

    def compute(self, audio_midpoint='Conv2d_b_3x3',
                video_midpoint='Conv3d_b_3x3x2', K=10, **kwargs):

        num_samples = self.dataset_train_AT.num_samples

        with tf.variable_scope('Prepare/Audio'):
            self.all_mfcc_reprs = tf.reshape(tf.Variable(
                self.audio_structure(
                    self.all_mfccs, final_endpoint=audio_midpoint),
                trainable=False), [num_samples, -1])

        with tf.variable_scope('Prepare/Video'):
            self.all_video_reprs = tf.Variable(self.video_structure(
                self.all_videos, final_endpoint=video_midpoint),
                trainable=False)

        num_classes = self.dataset_train_AT.num_classes
        video_logits = self.compute_logits_from_video(
            self.videos, num_classes, **kwargs)
        audio_logits = self.compute_logits_from_audio(
            self.mfccs_train,
            num_classes,
            audio_midpoint=audio_midpoint,
            video_midpoint=video_midpoint, K=K, **kwargs)

        self.video_or_audio = tf.random_uniform([])
        logits = tf.cond(
            self.video_or_audio < 0.25,
            lambda: audio_logits, lambda: video_logits)
        labels = tf.cond(
            self.video_or_audio < 0.25,
            lambda: self.labels_train_UZ, lambda: self.labels_train_AT)

        self.logits = tf.cond(
            self.training, lambda: logits, lambda: video_logits)
        self.labels = tf.cond(
            self.training, lambda: labels, lambda: self.labels_test)

    def compute_logits_from_video(self, inputs, num_classes,
                                  dropout_keep_prob=0.8,
                                  entry_point='inputs', reuse=None):
        with tf.variable_scope('Main', [inputs], reuse=reuse):
            net = self.video_structure(inputs, entry_point=entry_point)
            net = slim.dropout(net, dropout_keep_prob,
                               scope='PreLogitsDropout')
            net = slim.flatten(net, scope='PreLogitsFlatten')
            logits = slim.fully_connected(
                net, num_classes, activation_fn=None, scope='Logits')
            return logits

    def compute_logits_from_audio(self, inputs, num_classes,
                                  audio_midpoint,
                                  video_midpoint,
                                  dropout_keep_prob=0.8, K=10):

        with tf.variable_scope('Audio', [inputs]):
            audio_reprs = self.audio_structure(
                inputs, final_endpoint=audio_midpoint)
        final_video_reprs = []

        for i in range(audio_reprs.get_shape()[0]):
            audio_repr = tf.reshape(audio_reprs[i], [1, -1])
            audio_repr = tf.tile(
                audio_repr, [self.dataset_train_AT.num_samples, -1])
            distances = tf.negative(tf.sqrt(tf.reduce_sum(
                tf.square(audio_repr-self.all_mfcc_reprs), axis=1)))
            values, indices = tf.nn.top_k(distances, k=K)
            video_reprs = tf.stack(
                [self.all_video_reprs[indices[j]] for j in range(K)], 0)
            video_repr = tf.reduce_mean(video_reprs, axis=0)
            final_video_reprs.append(video_repr)

        final_video_reprs = tf.stack(final_video_reprs, axis=0)

        logits = self.compute_logits_from_video(
            final_video_reprs, num_classes,
            dropout_keep_prob=dropout_keep_prob,
            entry_point=video_midpoint, reuse=True)
        return logits

    def get_summary_op(self):
        self.get_batch_norm_summary()
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.histogram('logits', self.logits)
        tf.summary.scalar('losses/train/cross_entropy',
                          self.cross_entropy_loss)
        tf.summary.scalar('losses/train/total', self.total_loss)
        tf.summary.scalar('accuracy/train', self.accuracy_no_streaming)
        tf.summary.scalar('accuracy/train/streaming', self.accuracy)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        ac_test_summary = tf.summary.scalar(
            'accuracy/test', self.accuracy_no_streaming)
        ls_test_summary = tf.summary.scalar(
            'losses/test/total', self.total_loss)
        self.test_summary_op = tf.summary.merge(
            [ac_test_summary, ls_test_summary])
        return self.test_summary_op

    def train_step(self, sess, train_op, global_step, *args):
        tensors_to_run = [train_op, global_step]
        tensors_to_run.extend(args)
        tensors_to_run.append(self.video_or_audio)

        start_time = time.time()
        tensor_values = sess.run(
            tensors_to_run,
            feed_dict={self.training: True, self.batch_stat: True})
        time_elapsed = time.time() - start_time

        total_loss = tensor_values[0]
        global_step_count = tensor_values[1]
        use_audio = tensor_values[-1] < 0.25

        tf.logging.info(
            'global step %s: loss: %.4f (%.2f sec/step)',
            global_step_count, total_loss, time_elapsed)

        if use_audio:
            tf.logging.info('use audios to train')
        else:
            tf.logging.info('use videos to train')
        return tensor_values[:-1]

    def get_init_fn(self, checkpoint_dirs):
        checkpoint_dir_audio, checkpoint_dir_video = checkpoint_dirs

        variables_prepare_audio = {}
        variables_prepare_video = {}
        variables_audio = {}
        variables_main = {}

        for var in tf.model_variables():
            if var.op.name.startswith('Prepare/Audio'):
                variables_prepare_audio[var.op.name[14:]] = var
            if var.op.name.startswith('Prepare/Video'):
                variables_prepare_video[var.op.name[14:]] = var
            if var.op.name.startswith('Audio'):
                variables_audio[var.op.name[6:]] = var
            if var.op.name.startswith('Main'):
                variables_main[var.op.name[5:]] = var

        saver_prepare_audio = tf.train.Saver(variables_prepare_audio)
        saver_prepare_video = tf.train.Saver(variables_prepare_video)
        saver_audio = tf.train.Saver(variables_audio)
        saver_main = tf.train.Saver(variables_main)

        checkpoint_path_audio = tf.train.latest_checkpoint(
            checkpoint_dir_audio)
        checkpoint_path_video = tf.train.latest_checkpoint(
            checkpoint_dir_video)

        def restore(sess):
            saver_prepare_audio.restore(sess, checkpoint_path_audio)
            saver_prepare_video.restore(sess, checkpoint_path_video)
            saver_audio.restore(sess, checkpoint_dir_audio)
            saver_main.restore(sess, checkpoint_dir_video)
            sess.run(tf.variables_initializer([
                self.all_mfcc_reprs, self.all_video_reprs, self.all_labels]),
                feed_dict={self.batch_stat: True, self.training: True})

        return restore

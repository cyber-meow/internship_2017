from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from classify.train import TrainClassify
from data.mfcc_lips import load_batch_mfcc_lips, get_split_mfcc_lips

from audio.classify_routines import TrainClassifyAudio, EvaluateClassifyAudio
from audio.CNN_structure import CNN_mfcc6

from video.classify_routines import TrainClassifyVideo, EvaluateClassifyVideo
from video.CNN_structure import CNN_lips5

slim = tf.contrib.slim


class TrainClassifyAudioAVSR(TrainClassifyAudio):

    def get_data(self, tfrecord_dir, batch_size):
        self.dataset_train = get_split_mfcc_lips('train_all', tfrecord_dir)
        self.mfccs_train, _, self.labels_train = load_batch_mfcc_lips(
            self.dataset_train, batch_size=batch_size)
        self.dataset_test = get_split_mfcc_lips('validation', tfrecord_dir)
        self.mfccs_test, _, self.labels_test = load_batch_mfcc_lips(
            self.dataset_test, batch_size=batch_size, is_training=False)
        return self.dataset_train


class EvaluateClassifyAudioAVSR(EvaluateClassifyAudio):

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_mfcc_lips(split_name, tfrecord_dir)
        if batch_size is None:
            batch_size = self.dataset.num_samples
        self.mfccs, _, self.labels = load_batch_mfcc_lips(
            self.dataset, batch_size=batch_size,
            shuffle=shuffle, is_training=False)
        return self.dataset


class TrainClassifyVideoAVSR(TrainClassifyVideo):

    def get_data(self, tfrecord_dir, batch_size):
        self.dataset_train = get_split_mfcc_lips('train_all', tfrecord_dir)
        _, self.videos_train, self.labels_train = load_batch_mfcc_lips(
            self.dataset_train, batch_size=batch_size)
        self.dataset_test = get_split_mfcc_lips('validation', tfrecord_dir)
        _, self.videos_test, self.labels_test = load_batch_mfcc_lips(
            self.dataset_test, batch_size=batch_size, is_training=False)
        return self.dataset_train


class TrainClassifyVideoAVSRAT(TrainClassifyVideo):

    def get_data(self, tfrecord_dir, batch_size):
        self.dataset_train = get_split_mfcc_lips('trainAT', tfrecord_dir)
        _, self.videos_train, self.labels_train = load_batch_mfcc_lips(
            self.dataset_train, batch_size=batch_size)
        self.dataset_test = get_split_mfcc_lips('validation', tfrecord_dir)
        _, self.videos_test, self.labels_test = load_batch_mfcc_lips(
            self.dataset_test, batch_size=batch_size, is_training=False)
        return self.dataset_train


class EvaluateClassifyVideoAVSR(EvaluateClassifyVideo):

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_mfcc_lips(split_name, tfrecord_dir)
        if batch_size is None:
            batch_size = self.dataset.num_samples
        _, self.videos, self.labels = load_batch_mfcc_lips(
            self.dataset, batch_size=batch_size,
            shuffle=shuffle, is_training=False)
        return self.dataset


class TrainTransfer(TrainClassify):

    @property
    def default_trainable_scopes(self):
        return ['Main']

    def __init__(self, audio_structure='', video_structure='', **kwargs):
        super(TrainTransfer, self).__init__(**kwargs)
        if audio_structure == '':
            self.audio_structure = CNN_mfcc6
        else:
            self.audio_structure = audio_structure
        if video_structure == '':
            self.video_structure = CNN_lips5
        else:
            self.video_structure = video_structure

    def get_data(self, tfrecord_dir, batch_size):

        self.dataset_trainAT = get_split_mfcc_lips('trainAT', tfrecord_dir)
        _, self.videos_train, self.labels_trainAT = load_batch_mfcc_lips(
            self.dataset_trainAT, batch_size=batch_size)

        self.dataset_trainUZ = get_split_mfcc_lips('trainUZ', tfrecord_dir)
        self.mfccs_train, _, self.labels_trainUZ = load_batch_mfcc_lips(
            self.dataset_trainUZ, batch_size=batch_size, is_training=False)

        self.dataset_test = get_split_mfcc_lips('validation', tfrecord_dir)
        _, self.videos_test, self.labels_test = load_batch_mfcc_lips(
            self.dataset_test, batch_size=batch_size, is_training=False)

        self.all_mfccs, self.all_videos, all_labels = load_batch_mfcc_lips(
            self.dataset_trainAT, shuffle=False,
            batch_size=self.dataset_trainAT.num_samples)

        self.dataset_train = self.dataset_trainAT

        return self.dataset_trainAT

    def decide_used_data(self):
        self.videos = tf.cond(
            self.training, lambda: self.videos_train,
            lambda: self.videos_test)

    def compute(self, audio_midpoint='Conv2d_b_3x3',
                video_midpoint='Conv3d_b_3x3x2',
                use_audio_prob=0.9,
                K=10, **kwargs):

        num_samples = self.dataset_trainAT.num_samples

        with tf.variable_scope('Prepare/Audio'):
            self.all_mfcc_reprs = tf.Variable(tf.reshape(
                self.audio_structure(
                    self.all_mfccs, final_endpoint=audio_midpoint),
                [num_samples, -1]), trainable=False, name='mfcc_reprs')

        with tf.variable_scope('Prepare/Video'):
            self.all_video_reprs = tf.Variable(self.video_structure(
                self.all_videos, final_endpoint=video_midpoint),
                trainable=False, name='video_reprs')

        num_classes = self.dataset_trainAT.num_classes
        video_logits = self.compute_logits_from_video(
            self.videos, num_classes, **kwargs)
        audio_logits = self.compute_logits_from_audio(
            self.mfccs_train,
            num_classes,
            audio_midpoint=audio_midpoint,
            video_midpoint=video_midpoint, K=K, **kwargs)

        self.video_or_audio = tf.random_uniform([])
        self.use_audio_prob = use_audio_prob
        logits = tf.cond(
            self.video_or_audio < self.use_audio_prob,
            lambda: audio_logits, lambda: video_logits)
        labels = tf.cond(
            self.video_or_audio < self.use_audio_prob,
            lambda: self.labels_trainUZ, lambda: self.labels_trainAT)

        self.logits = tf.cond(
            self.training, lambda: logits, lambda: video_logits)
        self.labels = tf.cond(
            self.training, lambda: labels, lambda: self.labels_test)

    def compute_logits(self, inputs, num_classes):
        pass

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
                audio_repr, [self.dataset_trainAT.num_samples, 1])
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

    def step_log_info(self, sess):
        self.loss, _,  video_or_audio = self.train_step(
            sess, self.train_op, self.sv.global_step, self.video_or_audio)
        use_audio = video_or_audio < self.use_audio_prob
        if use_audio:
            tf.logging.info('audios were used to train')
        else:
            tf.logging.info('videos were used to train')

    def summary_log_info(self, sess):
        self.loss, _, _, summaries, streaming_accuracy_rate, \
            accuracy_rate, video_or_audio = self.train_step(
                sess, self.train_op, self.sv.global_step,
                self.metric_op, self.summary_op, self.streaming_accuracy,
                self.accuracy, self.video_or_audio)
        use_audio = video_or_audio < self.use_audio_prob
        if use_audio:
            tf.logging.info('audios were used to train')
        else:
            tf.logging.info('videos were used to train')
        tf.logging.info(
            'Current Streaming Accuracy:%s', streaming_accuracy_rate)
        tf.logging.info('Current Accuracy:%s', accuracy_rate)
        self.sv.summary_computed(sess, summaries)

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

        names = ['global_step', 'beta1_power', 'beta2_power']
        variables_to_init = []
        for var in tf.global_variables():
            if (var.op.name in names or
                    var.op.name.endswith('Adam') or
                    var.op.name.endswith('Adam_1')):
                print(var)
                variables_to_init.append(var)
        init_op = tf.variables_initializer(variables_to_init)

        # Since we only have A~T for initialization but audio trained on A~Z
        def restore(sess):
            tf.logging.info('Start restoring parameters.')
            tf.logging.info('Initializing some parameters.')
            sess.run(init_op)
            tf.logging.info('Restoring parameters for audio preparation.')
            saver_prepare_audio.restore(sess, checkpoint_path_audio)
            tf.logging.info('Restoring parameters for video preparation.')
            saver_prepare_video.restore(sess, checkpoint_path_video)
            tf.logging.info('Restoring parameters for main audio part.')
            saver_audio.restore(sess, checkpoint_path_audio)
            tf.logging.info('Restoring parameters for main video part.')
            saver_main.restore(sess, checkpoint_path_video)
        return restore

    def get_supervisor(self, log_dir, init_fn):
        self.extra_init_op = tf.variables_initializer(
            [self.all_mfcc_reprs, self.all_video_reprs])
        return tf.train.Supervisor(
            logdir=log_dir, summary_op=None, init_fn=init_fn,
            init_op=None, ready_op=None, save_model_secs=0)

    def extra_initialization(self, sess):
        tf.logging.info('Preparing pre-computed representations.')
        sess.run(self.extra_init_op, feed_dict={self.batch_stat: True})
        tf.logging.info('Finish restoring and preparing values.')
        self.sv.saver.save(sess, self.sv.save_path,
                           global_step=self.sv.global_step)


class EvaluateTransfer(EvaluateClassifyVideo):

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_mfcc_lips(split_name, tfrecord_dir)
        if batch_size is None:
            batch_size = self.dataset.num_samples
        _, self.videos, self.labels = load_batch_mfcc_lips(
            self.dataset, batch_size=batch_size,
            shuffle=shuffle, is_training=False)
        return self.dataset

    def init_model(self, sess, checkpoint_dirs):
        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        variables_to_restore = {}
        for var in tf.model_variables():
            variables_to_restore['Main/'+var.op.name] = var
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, checkpoint_path)

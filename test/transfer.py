from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from routines import train, evaluate
from audio import CNN_mfcc
from video import CNN_lips5

from video.transfer import TrainClassifyAudioAll, EvaluateClassifyAudioAll
from video.transfer import TrainClassifyVideoAll
from video.transfer import TrainClassifyVideo20, EvaluateClassifyVideo20
from video.transfer import TrainTransfer, EvaluateTransfer


class TransferTest(object):

    tfrecord_dir = '../dataset/avletters/tfrecords/mfcc_lips_transfer2'
    log_dir_audio = 'test/log/tranfer/audio'
    log_dir_video_AT = 'test/log/transfer/video_AT'
    log_dir_video_all = 'test/log/tranfer/video_all'
    log_dir_transfer = 'test/log/transfer/main'

    def train_audio(self, num_steps):
        train(TrainClassifyAudioAll, CNN_mfcc, self.tfrecord_dir, None,
              self.log_dir_audio, num_steps)

    def evaluate_audio(self, num, split_name):
        evaluate(EvaluateClassifyAudioAll, CNN_mfcc, self.tfrecord_dir,
                 self.log_dir_audio, None,
                 batch_size=num, split_name=split_name)

    def train_video_all(self, num_steps):
        TrainClassifyVideoAll(
            CNN_lips5, initial_learning_rate=2e-3, lr_decay_rate=0.96).train(
                self.tfrecord_dir, None, self.log_dir_video_all, num_steps)

    def evaluate_video_all(self, num, split_name):
        evaluate(EvaluateClassifyVideo20, CNN_lips5, self.tfrecord_dir,
                 self.log_dir_video_all, None,
                 batch_size=num, split_name=split_name)

    def train_video_AT(self, num_steps):
        TrainClassifyVideo20(
            CNN_lips5, initial_learning_rate=2e-3, lr_decay_rate=0.96).train(
                self.tfrecord_dir, None, self.log_dir_video_AT, num_steps)

    def evaluate_videoAT(self, num, split_name):
        evaluate(EvaluateClassifyVideo20, CNN_lips5, self.tfrecord_dir,
                 self.log_dir_video_AT, None,
                 batch_size=num, split_name=split_name)

    def train_transfer(self, num_steps):
        TrainTransfer(
            audio_structure=CNN_mfcc, video_structure=CNN_lips5,
            initial_learning_rate=8e-4, lr_decay_rate=0.96).train(
                self.tfrecord_dir, [self.log_dir_audio, self.log_dir_video_AT],
                self.log_dir_transfer, num_steps, K=6, audio_video_prob=0.9)

    def evaluate_transfer(self, num, split_name):
        EvaluateTransfer(CNN_lips5).evaluate(
            self.tfrecord_dir, self.log_dir_transfer, None,
            batch_size=num, split_name=split_name, shuffle=False)

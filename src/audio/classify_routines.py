from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from audio.basics import TrainAudio, EvaluateAudio, delta
from classify.train import TrainClassifyCNN
from classify.evaluate import EvaluateClassifyCNN


class TrainClassifyAudio(TrainClassifyCNN, TrainAudio):

    def compute(self, use_delta=False, **kwargs):
        if use_delta:
            mfcc_deltas = delta(self.mfccs)
            delta_deltas = delta(mfcc_deltas)
            data = tf.concat([self.mfccs, mfcc_deltas, delta_deltas], axis=3)
        else:
            data = self.mfccs
        self.logits = self.compute_logits(
            data, self.dataset_train.num_classes, **kwargs)


class EvaluateClassifyAudio(EvaluateClassifyCNN, EvaluateAudio):

    def compute(self, use_delta=False, **kwargs):
        if use_delta:
            mfcc_deltas = delta(self.mfccs)
            delta_deltas = delta(mfcc_deltas)
            data = tf.concat([self.mfccs, mfcc_deltas, delta_deltas], axis=3)
        else:
            data = self.mfccs
        self.logits = self.compute_logits(
            data, self.dataset.num_classes, **kwargs)

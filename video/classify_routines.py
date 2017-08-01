from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from video.basics import TrainVideo, EvaluateVideo
from classify.train import TrainClassifyCNN
from classify.evaluate import EvaluateClassifyCNN


class TrainClassifyVideo(TrainVideo, TrainClassifyCNN):

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.videos, self.dataset_train.num_classes, **kwargs)

    def get_summary_op(self):
        super(TrainClassifyVideo, self).get_summary_op()
        self.video0 = tf.transpose(self.videos[0], [2, 0, 1, 3])
        self.video1 = tf.transpose(self.videos[1], [2, 0, 1, 3])
        tf.summary.image('train0', self.video0, max_outputs=12)
        tf.summary.image('train1', self.video1, max_outputs=12)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        summary_op = super(TrainClassifyVideo, self).get_test_summary_op()
        v0 = tf.summary.image('test0', self.video0, max_outputs=12)
        v1 = tf.summary.image('test1', self.video1, max_outputs=12)
        self.test_summary_op = tf.summary.merge([summary_op, v0, v1])
        return self.test_summary_op


class EvaluateClassifyVideo(EvaluateVideo, EvaluateClassifyCNN):

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.videos, self.dataset.num_classes, **kwargs)

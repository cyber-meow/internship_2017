from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data.mfcc import load_batch_mfcc, get_split_mfcc
from routines.train import Train
from routines.evaluate import Evaluate


class TrainAudio(Train):

    def get_data(self, tfrecord_dir, batch_size):
        self.dataset_train = get_split_mfcc('train', tfrecord_dir)
        self.mfccs_train, self.labels_train = load_batch_mfcc(
            self.dataset_train, batch_size=batch_size)
        self.dataset_test = get_split_mfcc('validation', tfrecord_dir)
        self.mfccs_test, self.labels_test = load_batch_mfcc(
            self.dataset_test, batch_size=batch_size)
        return self.dataset_train


class EvaluateAudio(Evaluate):

    def get_data(self, split_name, tfrecord_dir, batch_size):
        self.dataset = get_split_mfcc(split_name, tfrecord_dir)
        self.mfccs, self.labels = load_batch_mfcc(
            self.dataset, batch_size=batch_size)
        return self.dataset

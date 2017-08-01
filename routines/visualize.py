from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets_base.arg_scope import nets_arg_scope

slim = tf.contrib.slim


class Visualize(object):

    def visualize(self,
                  tfrecord_dir,
                  checkpoint_dirs,
                  log_dir=None,
                  batch_size=300,
                  split_name='train',
                  use_batch_norm=True,
                  batch_stat=False,
                  shuffle=True,
                  **kwargs):

        if log_dir is not None and not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)

        if (checkpoint_dirs is not None
                and not isinstance(checkpoint_dirs, (tuple, list))):
            checkpoint_dirs = [checkpoint_dirs]

        with tf.Graph().as_default():

            with tf.name_scope('Data_provider'):
                self.get_data(split_name, tfrecord_dir, batch_size, shuffle)

            with slim.arg_scope(self.used_arg_scope(
                    batch_stat, use_batch_norm)):
                self.compute(**kwargs)

            with tf.Session() as sess:
                with slim.queues.QueueRunners(sess):
                    self.init_model(sess, checkpoint_dirs)
                    self.config_embedding(sess, log_dir)

    def used_arg_scope(self, batch_stat, use_batch_norm):
        return nets_arg_scope(
            is_training=batch_stat, use_batch_norm=use_batch_norm)

    def init_model(self, sess, checkpoint_dirs):
        if checkpoint_dirs is not None:
            assert len(checkpoint_dirs) == 1
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
            saver = tf.train.Saver(tf.model_variables())
            saver.restore(sess, checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())


def visualize(visualize_class,
              used_structure,
              tfrecord_dir,
              checkpoint_dirs,
              log_dir,
              **kwargs):
    visualize_instance = visualize_class(used_structure)
    for key in kwargs.copy():
        if hasattr(visualize_instance, key):
            setattr(visualize_instance, key, kwargs[key])
            del kwargs[key]
    visualize_instance.visualize(
        tfrecord_dir, checkpoint_dirs, log_dir, **kwargs)

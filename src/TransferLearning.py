# NOTICE
# This code contains a modified version of code originally from
# the Sketch-RNN repository (https://github.com/magenta/magenta/tree/main/magenta/models/sketch_rnn)

from io import BytesIO

import tensorflow as tf
import numpy as np
import os
import json
import time
import requests

import sketch_rnn_train
import model as sketch_rnn_model
import utils
from SketchToolUtilities import DataTweaker, DataUtilities

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

FLAGS = tf.compat.v1.app.flags.FLAGS

PRETRAINED_MODELS_URL = ('http://download.magenta.tensorflow.org/models/'
                         'sketch_rnn.zip')


models_root_dir = '/tmp/sketch_rnn/models'
model_dir = '/tmp/sketch_rnn/models/aaron_sheep/lstm'


def load_env_compatible(data_dir, model_dir):
    """Loads environment for inference mode, used in jupyter notebook."""
    # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
    # to work with depreciated tf.HParams functionality
    model_params = sketch_rnn_model.get_default_hparams()
    with tf.compat.v1.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        data = json.load(f)
    fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
    for fix in fix_list:
        data[fix] = (data[fix] == 1)
    model_params.parse_json(json.dumps(data))
    return load_dataset(data_dir, model_params)


def load_dataset(data_dir, model_params):
    """Loads the .npz file, and splits the set into train/valid/test."""

    # normalizes the x and y columns using the training set.
    # applies same scaling factor to valid and test set.

    if isinstance(model_params.data_set, list):
        datasets = model_params.data_set
    else:
        datasets = [model_params.data_set]

    train_strokes = None
    valid_strokes = None
    test_strokes = None

    for dataset in datasets:
        if data_dir.startswith('http://') or data_dir.startswith('https://'):
            data_filepath = '/'.join([data_dir, dataset])
            tf.compat.v1.logging.info('Downloading %s', data_filepath)
            response = requests.get(data_filepath, allow_redirects=True)
            data = np.load(BytesIO(response.content), allow_pickle=True, encoding='latin1')
        else:
            data_filepath = os.path.join(data_dir, dataset)
            data = np.load(data_filepath, encoding='latin1', allow_pickle=True)
        tf.compat.v1.logging.info('Loaded {}/{}/{} from {}'.format(
            len(data['train']), len(data['valid']), len(data['test']),
            dataset))
        if train_strokes is None:
            train_strokes = data['train']
            valid_strokes = data['valid']
            test_strokes = data['test']
        else:
            train_strokes = np.concatenate((train_strokes, data['train']))
            valid_strokes = np.concatenate((valid_strokes, data['valid']))
            test_strokes = np.concatenate((test_strokes, data['test']))

    all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))
    num_points = 0
    for stroke in all_strokes:
        num_points += len(stroke)
    avg_len = num_points / len(all_strokes)
    tf.compat.v1.logging.info('Dataset combined: {} ({}/{}/{}), avg len {}'.format(
      len(all_strokes), len(train_strokes), len(valid_strokes),
      len(test_strokes), int(avg_len)))

    # calculate the max strokes we need.
    max_seq_len = utils.get_max_len(all_strokes)
    # overwrite the hps with this calculation.
    model_params.max_seq_len = max_seq_len

    tf.compat.v1.logging.info('model_params.max_seq_len %i.', model_params.max_seq_len)

    train_set = DataTweaker(
        train_strokes,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor,
        augment_stroke_prob=model_params.augment_stroke_prob)

    normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
    train_set.normalize(normalizing_scale_factor)
    train_set.tweak()

    valid_set = DataTweaker(
        valid_strokes,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    valid_set.normalize(normalizing_scale_factor)
    valid_set.tweak()

    result = [
      train_set,
      valid_set,
      model_params,
    ]
    return result


def train(sess, model, train_set, valid_set):
    """Train a sketch-rnn model."""
    # Setup summary writer.
    summary_writer = tf.compat.v1.summary.FileWriter(FLAGS.log_root)

    # Calculate trainable params.
    t_vars = tf.compat.v1.trainable_variables()
    count_t_vars = 0
    for var in t_vars:
        num_param = np.prod(var.get_shape().as_list())
        count_t_vars += num_param
        tf.compat.v1.logging.info('%s %s %i', var.name, str(var.get_shape()), num_param)
    tf.compat.v1.logging.info('Total trainable variables %i.', count_t_vars)
    model_summ = tf.compat.v1.summary.Summary()
    model_summ.value.add(
        tag='Num_Trainable_Params', simple_value=float(count_t_vars))
    summary_writer.add_summary(model_summ, 0)
    summary_writer.flush()

    # main train loop

    hps = model.hps

    start = time.time()

    for step in range(hps.num_steps):

        curr_learning_rate = (
            (hps.learning_rate - hps.min_learning_rate) *
            (hps.decay_rate)**step +
            hps.min_learning_rate
        )
        curr_kl_weight = (
            hps.kl_weight -
            (hps.kl_weight - hps.kl_weight_start) *
            (hps.kl_decay_rate)**step
        )

        _, x, s = train_set.random_batch()
        feed = {
            model.input_data: x,
            model.sequence_lengths: s,
            model.lr: curr_learning_rate,
            model.kl_weight: curr_kl_weight,
        }

        (train_cost, r_cost, kl_cost, _, train_step, _) = sess.run([
            model.cost, model.r_cost, model.kl_cost, model.final_state,
            model.global_step, model.train_op
        ], feed)

        if step % 20 == 0 and step > 0:

            end = time.time()
            time_taken = end - start

            cost_summ = tf.compat.v1.summary.Summary()
            cost_summ.value.add(tag='Train_Cost', simple_value=float(train_cost))
            reconstr_summ = tf.compat.v1.summary.Summary()
            reconstr_summ.value.add(
                tag='Train_Reconstr_Cost', simple_value=float(r_cost))
            kl_summ = tf.compat.v1.summary.Summary()
            kl_summ.value.add(tag='Train_KL_Cost', simple_value=float(kl_cost))
            lr_summ = tf.compat.v1.summary.Summary()
            lr_summ.value.add(
                tag='Learning_Rate', simple_value=float(curr_learning_rate))
            kl_weight_summ = tf.compat.v1.summary.Summary()
            kl_weight_summ.value.add(
                tag='KL_Weight', simple_value=float(curr_kl_weight))
            time_summ = tf.compat.v1.summary.Summary()
            time_summ.value.add(
                tag='Time_Taken_Train', simple_value=float(time_taken))

            output_format = ('step: %d, lr: %.6f, klw: %0.4f, cost: %.4f, '
                             'recon: %.4f, kl: %.4f, train_time_taken: %.4f')
            output_values = (step, curr_learning_rate, curr_kl_weight, train_cost,
                             r_cost, kl_cost, time_taken)
            output_log = output_format % output_values

            tf.compat.v1.logging.info(output_log)

            summary_writer.add_summary(cost_summ, train_step)
            summary_writer.add_summary(reconstr_summ, train_step)
            summary_writer.add_summary(kl_summ, train_step)
            summary_writer.add_summary(lr_summ, train_step)
            summary_writer.add_summary(kl_weight_summ, train_step)
            summary_writer.add_summary(time_summ, train_step)
            summary_writer.flush()
            start = time.time()

            if not step % hps.save_every and step > 0:
                print('*** Validation ***')
                (
                    valid_cost,
                    valid_r_cost,
                    valid_kl_cost
                ) = sketch_rnn_train.evaluate_model(
                    sess,
                    model,
                    valid_set
                )

                end = time.time()
                time_taken_valid = end - start
                DataUtilities.strokes_to_svg(sketch_rnn_model.sample(sess, model), filename=str(step))
                start = time.time()

                valid_cost_summ = tf.compat.v1.summary.Summary()
                valid_cost_summ.value.add(
                    tag='Valid_Cost', simple_value=float(valid_cost))
                valid_reconstr_summ = tf.compat.v1.summary.Summary()
                valid_reconstr_summ.value.add(
                    tag='Valid_Reconstr_Cost', simple_value=float(valid_r_cost))
                valid_kl_summ = tf.compat.v1.summary.Summary()
                valid_kl_summ.value.add(
                    tag='Valid_KL_Cost', simple_value=float(valid_kl_cost))
                valid_time_summ = tf.compat.v1.summary.Summary()
                valid_time_summ.value.add(
                    tag='Time_Taken_Valid', simple_value=float(time_taken_valid))

                tf.compat.v1.logging.info(output_log)

                summary_writer.add_summary(valid_cost_summ, train_step)
                summary_writer.add_summary(valid_reconstr_summ, train_step)
                summary_writer.add_summary(valid_kl_summ, train_step)
                summary_writer.add_summary(valid_time_summ, train_step)
                summary_writer.flush()


if __name__ == '__main__':
    sketch_rnn_train.download_pretrained_models(models_root_dir)
    tf.compat.v1.disable_v2_behavior()
    [train_set, valid_set, hps_model] = load_env_compatible(FLAGS.data_dir, model_dir)
    sketch_rnn_train.reset_graph()
    model = sketch_rnn_model.Model(hps_model)
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    sketch_rnn_train.load_checkpoint(sess, model_dir)
    train(sess, model, train_set, valid_set)
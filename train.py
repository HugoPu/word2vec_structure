from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

import reader
import util
from config import get_config, FLAGS
from input import PTBInput
from model import PTBModel


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0.0
    state = session.run(model.initial_state) # Get state
    fetches = {
        'cost': model.cost,
        'final_state': model.final_state
    }
    if eval_op is not None:
        fetches['eval_op'] = eval_op

    # Loop total corpus one time,
    for step in range(model.input.epoch_size):
        feed_dict = {}
        # Get t-1 state, since each epoch is end-to-end
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        # Run one epoch, and update state
        vals = session.run(fetches, feed_dict)
        cost = vals['cost']
        state = vals['final_state']

        costs += cost
        iters += model.input.num_steps

        # Calculate perplexity, but the formula is slightly different? probabilities are changed to cross entropys?
        if verbose and step % (model.input.epoch_size // 10) == 10:
            print('%.3f perplexity: %.3f speed: %.0f wps' %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
                   (time.time() - start_time)))

    return np.exp(costs / iters)

def main(_):
    if not FLAGS.data_path:
        raise ValueError('Must set --data_path to PTB data directory')
    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']

    if FLAGS.num_gpus > len(gpus):
        raise ValueError('Your machine has only %d gpus '
                         'which is less than the requested --num_gpus=%d.'
                         % (len(gpus), FLAGS.num_gpus))

    # Genereate words to ids dictionary and convert words to ids
    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    # Get hyperparameters
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        # Generate
        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss,", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope('Valid'):
            valid_input = PTBInput(config=config, data=valid_data, name='ValidInput')
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar('Validation Loss', mvalid.cost)

        with tf.name_scope('Test'):
            test_input = PTBInput(config=eval_config, data=test_data, name='TestInput')
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)

        # Add ops to collection (tf.add_to_collection), The collection is managed by tensorflow"
        models = {'Train': m, 'Valid': mvalid, 'Test': mtest}
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph() # Export the graph, it can be stored in the disk
        if tf.__version__ < '1.1.0' and FLAGS.num_gpus > 1:
            raise ValueError('num_gpus > 1 is not supported for TensorFlow versions '
                             'below 1.1.0')
        # Parallel config
        soft_placement = False
        if FLAGS.num_gpus > 1:
            soft_placement = True
            util.auto_parallel(metagraph, m)

    with tf.Graph().as_default():

        # Import ops and graph
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()

        # Use supervisor to save and load checkpoint, pre-train variables
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)  # Used to set config for session
        with sv.managed_session(config=config_proto) as session:
            # Times to loop corpusvxcvzxvxvzxvzxvz
            for i in range(config.max_max_epoch):
                time1 = time.time()
                # Calculate learning decay
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print('Epoch: %d Learning rate: %.3f' % (i + 1, session.run(m.lr)))

                train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
                print('Epoch: %d Train Perplexity: %.3f' % (i + 1, train_perplexity))

                valid_perplexity = run_epoch(session, mvalid)
                print('Epoch: %d Valid Perplexity: %.3f' % (i + 1, valid_perplexity))

                print('One loop used %d s' % time.time() - time1)

            test_perplexity = run_epoch(session, mtest)
            print('Epoch: %d Valid Perplexity: %.3f' % test_perplexity)

            if FLAGS.save_path:
                print('Saving model to %s.' % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

if __name__ == '__main__':
    # Run the program with an optional function 'main'
    tf.app.run()
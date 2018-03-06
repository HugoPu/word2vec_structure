import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string('model', 'small', "A type of model. Possible options are: small, medium, large")

flags.DEFINE_string('data_path', './simple-examples/data', 'Where the training/test data is stored.')

flags.DEFINE_string('save_path', './save_path', 'Model output directory.')

flags.DEFINE_bool('use_fp16', False, 'Train using 16-bit floats instead of 32bit float.')

flags.DEFINE_integer('num_gpus', 0, 'If larger than 1, Grappler AutoParallel optimizer '
                     'will create multiple training replicas with each GPU '
                     'running one replica.')

flags.DEFINE_string('rnn_mode', 'BASIC', 'The low level implementation of lstm cell: one of CUDNN, '
                    'BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, '
                    'and lstm_block_cell classes.')

FLAGS = flags.FLAGS
BASIC = 'basic'
CUDNN = 'cudnn'
BLOCK = 'block'

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1 # Use to initialize default variable initializer
  learning_rate = 1.0 # Learning rate
  max_grad_norm = 1 # Threshold to prevent gradient explosion
  num_layers = 1 # The number of rnn layers
  num_steps = 2 # The number of words in one sentence
  hidden_size = 2 # Embedding size and neuron size, i.e. W:(hidden_size(num_units), hidden_size(embedding_size))
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0 # Drop out probability
  lr_decay = 0.5 # Learning rate decay
  batch_size = 20 # Mini-batch size
  vocab_size = 10000 # Vocabulary size
  rnn_mode = BLOCK # RNN model

def get_config():
    """Get model config"""
    config = None
    if FLAGS.model == 'small':
        config = SmallConfig()
    elif FLAGS.model == 'medium':
        config = MediumConfig()
    elif FLAGS.model == 'large':
        config = LargeConfig()
    elif FLAGS.model == 'test':
        config = TestConfig()
    else:
        raise ValueError('Invalid model: %s', FLAGS.model)
    if FLAGS.rnn_mode:
        config.rnn_mode = FLAGS.rnn_mode
    if FLAGS.num_gpus != 1 or tf.__version__ < '1.3.0':
        config.rnn_mode = BASIC
    return config
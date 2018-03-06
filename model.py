import tensorflow as tf
import util

FLAGS = tf.flags.FLAGS
from config import BASIC, BLOCK, CUDNN, data_type
from decorator import define_scope

class PTBModel(object):
    """The PTB model"""

    def __init__(self, is_training, config, input_):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type()) # Initilize embedding matrix
        inputs = tf.nn.embedding_lookup(embedding, input_.input_data) # Generate input matrix (batch_size, num_steps, size)
        # Build model return output and state
        # output (num_steps*batch_size, embedding_siz)
        # c,h (num_steps, embedding_size)
        output, state = self._build_rnn_graph(inputs, config, is_training)

        softmax_w = tf.get_variable('softmax_w', [size, vocab_size], dtype=data_type())

        softmax_b = tf.get_variable('softmax_b', [vocab_size], dtype=data_type())

        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b) # Wx + b

        # Reshape logits to be a 3-D tensor for sequence loss
        # The value before softmax
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # Use the contrib sequence loss and average over the batches
        # Weighted cross-entropy loss for a sequence of logits.
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True
        )

        # Update the cost
        self._cost = tf.reduce_sum(loss)
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables() # Get all trainable variables
        # Prevent gradient explosion
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        # Apply gradients which have been processed to variables
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step()
        )

        self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _build_rnn_graph(self, inputs, config, is_training):
        if config.rnn_mode == CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        """Build the inference graph using CUDNN cell."""
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=config.num_layers,
            num_units=config.hidden_size,
            input_size=config.hidden_size,
            dropout=1 - config.keep_prob if is_training else 0
        )
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
            "lstm_params",
            initializer=tf.random_uniform([params_size_t], -config.init_scale, config.init_scale),
            validate_shape=False
        )

        c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size], tf.float32)

        h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size], tf.float32)

        self._init_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

    def _get_lstm_cell(self, config, is_training):
        # hiden_size: The number of neuron, W:(hidden_size, embedding_size)
        if config.rnn_mode == BASIC:
            return tf.contrib.rnn.BasicLSTMCell(
                config.hidden_size,
                forget_bias=0.0,
                state_is_tuple=True,
                reuse=not is_training
            )
        if config.rnn_mode == BLOCK:
            return tf.contrib.rnnLSTMBlockCell(
                config.hidden_size,
                forget_bias=0.0
            )
        raise ValueError('run_mode %s not supported' % config.rnn_mode)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.

        # Generate one cell
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            return cell

        # Generate multiple layers of cell
        cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        # Generate a0(batch_size, num_units)
        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state
        # Simplified version of tf.nn.static_rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        # outputs, state = tf.nn.static_rnn(cell, inputs,
        #                                   initial_state=self._initial_state)

        outputs = []
        with tf.variable_scope("RNN"):
            # Dynamic generate RNN which is based on the length
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables() # Reuse Ws
                (cell_output, state) = cell(inputs[:, time_step, :], state)# Input data to the xth cell
                outputs.append(cell_output) # outputs:(time_step, batch_size, embedding_size)
        # First, concat to (batch_size, embedding_size*num_steps)
        # Then, reshape to (num_steps*batch_size, embedding_size)
        # The order of 1st dimenssion is num_steps1, num_steps2,...num_steps*batch_size
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """Exports ops to collections. The collection is managed by tensorflow"""
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for n, op in ops.items():
            tf.add_to_collection(n, op)
        self._initial_state_name = util.with_prefix(self._name, 'initial')
        self._final_state_name = util.with_prefix(self._name, 'final')
        util.export_state_tuples(self._initial_state, self._initial_state_name)
        util.export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self):
        """Import ops from collections."""
        if self._is_training:
            self._train_op = tf.get_collection_ref('train_op')[0]
            self._lr = tf.get_collection_ref('lr')[0]
            self._new_lr = tf.get_collection_ref('new_lr')[0]
            self._lr_update = tf.get_collection_ref('lr_update')[0]
            rnn_params = tf.get_collection_ref('rnn_params')
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
                    self._cell,
                    self._cell.params_to_canonical,
                    self._cell.canonical_to_params,
                    rnn_params,
                    base_variable_scope='Model/RNN')
                tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(util.with_prefix(self._name, 'cost'))[0]
        num_replicas = FLAGS.num_gpus if self._name == 'Train' else 1
        self._initial_state = util.import_state_tuples(self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = util.import_state_tuples(self._final_state, self._final_state_name, num_replicas)

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name

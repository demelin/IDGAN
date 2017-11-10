import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.layers import xavier_initializer as xi


class CogLM(object):
    """ A multi-layer RNN-LSTM language model;
    callable variant used as part of the IDGAN system for easier parameter sharing. """

    def __init__(self, vocab, opt, name):
        # Declare attributes
        self.vocab = vocab
        self.opt = opt
        self.name = name
        self.int_type = tf.int32
        self.float_type = tf.float32

        # Construct sub-graph defining the placeholder variables first, as subsequent sub-graph declarations call to
        # placeholder values
        self.static_keep_prob, self.rnn_keep_prob = self.inputs_subgraph()

    def inputs_subgraph(self):
        """ Specifies inputs supplied to the model during graph execution. """
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            # Read-in dropout probabilities from session input
            static_keep_prob = tf.placeholder(dtype=self.float_type, name='static_keep_prob')
            rnn_keep_prob = tf.placeholder(dtype=self.float_type, name='rnn_keep_prob')
        return static_keep_prob, rnn_keep_prob

    def embeddings_subgraph(self):
        """ Initializes the embedding table and output biases; embedding table is jointly used as the projection matrix
        for projecting the RNN-generated logits into the vocabulary space for inference and loss calculation. """
        with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
            embedding_table = tf.get_variable(name='embedding_table',
                                              shape=[self.vocab.n_words, self.opt.embedding_dims],
                                              dtype=self.float_type,
                                              initializer=xi(uniform=False, dtype=self.float_type),
                                              trainable=True)
            output_embedding_biases = tf.get_variable(name='output_embedding_biases', shape=[self.vocab.n_words],
                                                      dtype=self.float_type,
                                                      initializer=tf.zeros_initializer(dtype=self.float_type),
                                                      trainable=True)
        return embedding_table, output_embedding_biases

    def lstm_rnn_subgraph(self):
        """ Defines the LSTM-RNN cell. """
        with tf.variable_scope('lstm_rnn'), tf.device('/gpu:0'):
            # Helper function for defining the RNN cell;
            # here, LSTMs are used
            def _lstm_cell(model_opt):
                """ Defines a basic LSTM cell to which various wrappers can be applied. """
                base_cell = BasicLSTMCell(model_opt.hidden_dims, forget_bias=2.5, state_is_tuple=True)
                if model_opt.is_train:
                    base_cell = DropoutWrapper(base_cell, output_keep_prob=self.rnn_keep_prob)
                return base_cell

            # Instantiate number of layers according to value specified in options
            if self.opt.num_layers > 1:
                cell = MultiRNNCell([_lstm_cell(self.opt) for _ in range(self.opt.num_layers)])
            else:
                cell = _lstm_cell(self.opt)
        return cell

    def projection_subgraph(self):
        """ Defines the weight and bias parameters used to project RNN outputs into the embedding space following
        the completion of each full pass through the RNN. """
        with tf.variable_scope('projection'), tf.device('/gpu:0'):
            projection_weights = tf.get_variable(name='projection_weights',
                                                 shape=[self.opt.hidden_dims, self.opt.embedding_dims],
                                                 dtype=self.float_type,
                                                 initializer=xi(uniform=False, dtype=self.float_type),
                                                 trainable=True)
            projection_biases = tf.get_variable(name='projection_biases', shape=[self.opt.embedding_dims],
                                                dtype=self.float_type,
                                                initializer=tf.zeros_initializer(dtype=self.float_type),
                                                trainable=True)
        return projection_weights, projection_biases

    def __call__(self, input_idx, reuse=False):
        """ Returns the word-wise probabilities for the supplied input data;
        as training is performed using the non-callable implementation, here only ID-relevant OPs are defined,
        to reduce the size of the full IDGAN system; initialized with parameters learned during pre-training. """
        # Share parameters across instantiations
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # Build computational sub-graphs
            # embeddings | lstm_rnn | projection
            embedding_table, output_embedding_biases = self.embeddings_subgraph()
            cell = self.lstm_rnn_subgraph()
            projection_weights, projection_biases = self.projection_subgraph()

            with tf.name_scope('properties'), tf.device('/cpu:0'):
                # Get input properties required by downstream operations
                batch_length = tf.shape(input_idx)[0]
                # Determine lengths of individual input sequences within the processed batch to mask RNN output and
                # exclude <EOS> tokens from loss calculation
                length_mask = tf.count_nonzero(
                    tf.not_equal(input_idx, self.vocab.pad_id), axis=1, keep_dims=False, dtype=self.int_type,
                    name='length_mask')

            with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                # Embed input indices
                input_data = tf.nn.embedding_lookup(embedding_table, input_idx, name='embeddings')
                if self.opt.is_train:
                    # Optionally apply dropout (at training time only)
                    input_data = tf.nn.dropout(input_data, self.static_keep_prob, name='front_dropout')

            with tf.variable_scope('lstm_rnn'), tf.device('/gpu:0'):
                # Run input embeddings through the RNN;
                # helper function resets RNN state between mini-batches
                def _get_zero_state(source_cell):
                    """ Returns the zeroed initial state for the source LSTM cell. """
                    return source_cell.zero_state(batch_length, self.float_type)

                initial_state = _get_zero_state(cell)
                # Obtain RNN output for the current mini-batch
                # time-major format == [batch_size, step_num, hidden_size]
                rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, input_data, sequence_length=length_mask,
                                                             initial_state=initial_state, dtype=self.float_type,
                                                             parallel_iterations=1, swap_memory=True, time_major=False)
                # Flatten output for subsequent projection
                flat_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.opt.hidden_dims], name='reshaped_rnn_outputs')
                # Optionally apply dropout (at training time only)
                flat_rnn_outputs = tf.nn.dropout(flat_rnn_outputs, self.static_keep_prob, name='back_dropout')

            with tf.variable_scope('projection'), tf.device('/gpu:0'):
                # Project RNN-output into the embedding space
                projected_rnn_outputs = tf.nn.xw_plus_b(flat_rnn_outputs, projection_weights, projection_biases)

            with tf.variable_scope('prediction'), tf.device('/gpu:0'):
                # Project RNN-logits into vocabulary space
                logits = tf.nn.xw_plus_b(projected_rnn_outputs, tf.transpose(embedding_table),
                                         output_embedding_biases, name='logits')
                # Construct predictive distribution by passing projected RNN output through the softmax non-linearity
                predictions = tf.nn.softmax(logits, name='predictions')
            return length_mask, predictions

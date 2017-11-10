import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.layers import xavier_initializer as xi


class Encoder(object):
    """ RNN-LSTM encodes used to encode input sentences into dense, continuous representations used in
    content similarity estimation by the similarity classifier."""

    def __init__(self, vocab, opt, name):
        # Declare attributes
        self.vocab = vocab
        self.opt = opt
        self.name = name
        self.float_type = tf.float32
        self.int_type = tf.int32

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
        """ Initializes the embedding table. """
        with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
            embedding_table = tf.get_variable(name='embedding_table',
                                              shape=[self.vocab.n_words, self.opt.embedding_dims],
                                              dtype=self.float_type,
                                              initializer=xi(uniform=False, dtype=self.float_type),
                                              trainable=True)
            return embedding_table

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
                cell_fw = MultiRNNCell([_lstm_cell(self.opt) for _ in range(self.opt.num_layers)])
                cell_bw = MultiRNNCell([_lstm_cell(self.opt) for _ in range(self.opt.num_layers)])
            else:
                cell_fw = _lstm_cell(self.opt)
                cell_bw = _lstm_cell(self.opt)
        return cell_fw, cell_bw

    def attention_subgraph(self):
        """ Designates the parameters for the self-attention mechanism used to obtain improved sentence encodings. """
        with tf.variable_scope('attention'), tf.device('/gpu:0'):
            projection_weights = tf.get_variable(name='projection_weights',
                                                 shape=[self.opt.hidden_dims * 2, self.opt.attention_dims],
                                                 initializer=xi(uniform=False, dtype=self.float_type),
                                                 trainable=True)
            projection_biases = tf.get_variable(name='projection_biases', shape=[self.opt.attention_dims],
                                                initializer=tf.zeros_initializer(dtype=self.float_type),
                                                trainable=True)
            context_vector = tf.get_variable(name='context_vector', shape=[self.opt.attention_dims],
                                             initializer=xi(uniform=False, dtype=self.float_type), trainable=True)
        return projection_weights, projection_biases, context_vector

    def __call__(self, input_idx, encoder_reuse=False):
        """ Returns a low-dimensional sentence encoding used in the calculation of semantic similarity scores;
        only inference-relevant OPs are defined, so as to reduce the number of parameters of the full/ extended IDGAN;
        initialized with parameters learned during pre-training (using the non-callable implementation). """
        # Share parameters across both encoders
        with tf.variable_scope(self.name) as scope:
            if encoder_reuse:
                scope.reuse_variables()

            # Build computational sub-graphs
            # embeddings | lstm_rnn | attention
            embedding_table = self.embeddings_subgraph()
            cell_fw, cell_bw = self.lstm_rnn_subgraph()
            projection_weights, projection_biases, context_vector = self.attention_subgraph()

            with tf.name_scope('properties'), tf.device('/cpu:0'):
                # Get input properties required by downstream operations
                batch_length = tf.shape(input_idx)[0]
                batch_steps = tf.shape(input_idx)[1]
                # Determine lengths of individual input sequences within the processed batch to mask RNN output and
                # exclude <EOS> and <PAD> tokens from contributing to the sentence encoding
                length_mask = tf.count_nonzero(
                    tf.not_equal(input_idx, self.vocab.pad_id), axis=1, keep_dims=False, name='length_mask')

            with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                # Embed input indices
                input_data = tf.nn.embedding_lookup(embedding_table, input_idx, name='embeddings')
                if self.opt.is_train:
                    # Optionally apply dropout (at training time only)
                    input_data = tf.nn.dropout(input_data, self.static_keep_prob, name='front_dropout')

            with tf.variable_scope('lstm_rnn'), tf.device('/gpu:0'):
                # Run input embeddings through the RNN;
                # helper function resets RNN stats of the forward and backward cells between mini-batches
                def _get_zero_state(source_cell_fw, source_cell_bw):
                    """ Returns the zeroed initial state for the source LSTM cell. """
                    zero_fw = source_cell_fw.zero_state(batch_length, self.float_type)
                    zero_bw = source_cell_bw.zero_state(batch_length, self.float_type)
                    return zero_fw, zero_bw

                initial_state_fw, initial_state_bw = _get_zero_state(cell_fw, cell_bw)
                # Obtain RNN output for the current mini-batch
                # time-major format == [batch_size, step_num, hidden_size]
                bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_data,
                                                                        sequence_length=length_mask,
                                                                        initial_state_fw=initial_state_fw,
                                                                        initial_state_bw=initial_state_bw,
                                                                        dtype=self.float_type, parallel_iterations=1,
                                                                        swap_memory=True, time_major=False)
                # RNN output is a concatenation of the outputs obtained from the forward and backward cell(s)
                rnn_outputs = tf.concat(bi_outputs, 2, name='concatenated_rnn_outputs')

            with tf.variable_scope('attention'), tf.device('/gpu:0'):
                # Publication: see www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
                # Publication code: github.com/ematvey/hierarchical-attention-networks/blob/master/HAN_model.py
                # Compute attention values
                memory_values = tf.reshape(rnn_outputs, shape=[-1, self.opt.hidden_dims * 2], name='memory_values')
                projected_memories = tf.nn.tanh(
                    tf.nn.xw_plus_b(memory_values, projection_weights, projection_biases), name='projected_memories')
                projected_memories = tf.reshape(projected_memories, shape=[batch_length, batch_steps, -1])
                # Mask out positions corresponding to padding within the input
                score_mask = tf.sequence_mask(length_mask, maxlen=tf.reduce_max(length_mask), dtype=self.float_type)
                score_mask = tf.expand_dims(score_mask, -1)
                score_mask = tf.matmul(
                    score_mask, tf.ones([batch_length, self.opt.attention_dims, 1]), transpose_b=True)
                projected_memories = tf.where(
                    tf.cast(score_mask, dtype=tf.bool), projected_memories, tf.zeros_like(projected_memories))
                # Calculate the importance of the individual encoder hidden states for the informativeness of the
                # computed sentence representation
                context_product = tf.reduce_sum(
                    tf.multiply(projected_memories, context_vector, name='context_product'), axis=2, keep_dims=True)
                attention_weights = tf.nn.softmax(context_product, dim=1, name='importance_weight')
                # Weigh encoder hidden states according to the calculated importance weights
                weighted_memories = tf.multiply(projected_memories, attention_weights)
                # Sentence encodings are the importance-weighted sums of encoder hidden states / word representations
                sentence_encodings = tf.reduce_sum(weighted_memories, axis=1, name='sentence_encodings')
            return sentence_encodings


class SentSimClassifier(object):
    """ Sequence similarity classifier model trained to calculate sentence similarity scores on the basis of
    informative encodings extracted from two encoder networks. """

    def __init__(self, vocab, opt, name):
        # Declare attributes
        self.vocab = vocab
        self.opt = opt
        self.name = name
        self.float_type = tf.float32
        self.int_type = tf.int32

    def __call__(self, input_idx_a, input_idx_b, reuse=False):
        """ Computes Manhatten-distance based similarity score between both input sentences. """
        # Share parameters across instantiations
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # Initialize the encoders (a single instantiation is used cross both sentences, i.e. in a 'Siamese' setup)
            self.enc = Encoder(self.vocab, self.opt, 'encoder')

            with tf.name_scope('prediction'), tf.device('/gpu:0'):
                # Calculate and return sentence similarity scores
                predictions = tf.exp(-tf.norm(tf.subtract(
                    self.enc(input_idx_a, encoder_reuse=False), self.enc(input_idx_b, encoder_reuse=True)),
                    ord=1, axis=-1, keep_dims=True), name='distance')
            return predictions

import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.layers import xavier_initializer as xi


class Encoder(object):
    """ RNN-LSTM encoder used within the SAE-dyad to obtain dense, low-dimensional sentence
    encodings; generates sentence embeddings on the basis of sequential input; callable variant used as part of the
    IDGAN system for easier parameter sharing. """

    def __init__(self, vocab, opt, name):
        # Declare attributes
        self.vocab = vocab
        self.opt = opt
        self.name = name
        self.float_type = tf.float32
        self.int_type = tf.int32

        # Declare accessible values
        with tf.name_scope(self.name):
            # Initialize input placeholders
            self.static_keep_prob, self.rnn_keep_prob = self.inputs_subgraph()
            # Initialize outputs
            self.batch_length = None
            self.batch_steps = None
            self.length_mask = None
            self.embedding_table = None
            self.output_embedding_biases = None
            self.rnn_outputs = None
            self.h_state = None
            self.encoded_state = None

    def inputs_subgraph(self):
        """ Creates the placeholders used to feed data to the model. """
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            # Read-in dropout probabilities from session input
            static_keep_prob = tf.placeholder(dtype=self.float_type, name='static_keep_prob')
            rnn_keep_prob = tf.placeholder(dtype=self.float_type, name='rnn_keep_prob')
        return static_keep_prob, rnn_keep_prob

    def embeddings_subgraph(self):
        """ Initializes the embedding table and output biases; embedding table is jointly used as the projection matrix
        for projecting the RNN-generated logits into the vocabulary space in the decoder. """
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
            # Helper function for defining the RNN cell; here, LSTMs are used
            def _lstm_cell(model_opt):
                """ Defines a basic LSTM cell to which various wrappers may be applied. """
                base_cell = BasicLSTMCell(model_opt.enc_hidden_dims, forget_bias=2.5, state_is_tuple=True)
                if model_opt.allow_dropout:
                    base_cell = DropoutWrapper(base_cell, output_keep_prob=self.rnn_keep_prob)
                return base_cell

            # Instantiate number of layers according to value specified in options
            if self.opt.enc_num_layers > 1:
                enc_cell = MultiRNNCell([_lstm_cell(self.opt) for _ in range(self.opt.enc_num_layers)])
            else:
                enc_cell = _lstm_cell(self.opt)
        return enc_cell

    def state_projection_subgraph(self):
        """ Defines parameters for the encoder state projection, in case of a state size mismatch between
        encoder and decoder; unused if encoder and decoder states are of identical size. """
        with tf.variable_scope('state_projection'), tf.device('/gpu:0'):
            state_projection_weights = tf.get_variable(name='state_projection_weights',
                                                       shape=[self.opt.enc_hidden_dims * self.opt.enc_num_layers,
                                                              self.opt.dec_hidden_dims * self.opt.dec_num_layers],
                                                       dtype=self.float_type,
                                                       initializer=xi(uniform=False, dtype=self.float_type),
                                                       trainable=True)
            state_projection_biases = tf.get_variable(name='state_projection_biases',
                                                      shape=[self.opt.dec_hidden_dims * self.opt.dec_num_layers],
                                                      dtype=self.float_type,
                                                      initializer=tf.zeros_initializer(self.float_type),
                                                      trainable=True)
        return state_projection_weights, state_projection_biases

    def __call__(self, enc_input_idx):
        """ Generates informative encodings on the basis of input sentences. """
        with tf.variable_scope(self.name):
            # Build computational sub-graphs:
            # embeddings | rnn_lstm | state projection
            embedding_table, output_embedding_biases = self.embeddings_subgraph()
            enc_cell = self.lstm_rnn_subgraph()
            state_projection_weights, state_projection_biases = self.state_projection_subgraph()

            with tf.name_scope('properties'), tf.device('/cpu:0'):
                # Get input properties required by downstream operations
                batch_length = tf.shape(enc_input_idx)[0]
                batch_steps = tf.shape(enc_input_idx)[1]
                # Determine lengths of individual input sequences within the processed batch to mask RNN output and
                # exclude <EOS> and <PAD> tokens from contributing to the sentence encoding
                length_mask = tf.count_nonzero(tf.not_equal(enc_input_idx, self.vocab.pad_id), axis=1, keep_dims=False,
                                               dtype=self.int_type, name='length_mask')

            with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                # Embed input indices
                input_data = tf.nn.embedding_lookup(embedding_table, enc_input_idx, name='embeddings')
                # Optionally apply dropout (at training time only)
                if self.opt.allow_dropout:
                    input_data = tf.nn.dropout(input_data, self.static_keep_prob, name='enc_front_dropout')

            with tf.variable_scope('lstm_rnn'), tf.device('/gpu:0'):
                # Run input embeddings through the RNN;
                # helper function resets RNN state between mini-batches
                def _get_zero_state(cell):
                    """ Returns the zeroed initial state for the source LSTM cell. """
                    zero_state = cell.zero_state(batch_length, self.float_type)
                    return zero_state

                initial_state = _get_zero_state(enc_cell)
                # Obtain RNN output - i.e. sentence encodings - for the current mini-batch
                # time-major format == [batch_size, step_num, hidden_size]
                rnn_outputs, rnn_state = tf.nn.dynamic_rnn(enc_cell, input_data,
                                                           sequence_length=length_mask,
                                                           initial_state=initial_state,
                                                           dtype=self.float_type, parallel_iterations=1,
                                                           swap_memory=True, time_major=False)
                if self.opt.enc_num_layers > 1:
                    # Combine the final hidden and cell states of the LSTM cells from all layers into single matrices
                    # Combined hidden matrix is used as input to the GAN discriminator, as the decoder is initialized
                    # using information from all of encoder's hidden layers
                    c_state = tf.concat([rnn_state[layer_id].c for layer_id in range(self.opt.enc_num_layers)], -1)
                    h_state = tf.concat([rnn_state[layer_id].h for layer_id in range(self.opt.enc_num_layers)], -1)
                else:
                    c_state = rnn_state.c
                    h_state = rnn_state.h

            with tf.variable_scope('state_projection'), tf.device('/gpu:0'):
                # Project encoder's state information to match the decoder's dimensions, if necessary
                if self.opt.enc_num_layers == self.opt.dec_num_layers:
                    # Assign encoder states to decoder states on a by layer basis
                    c_states = tf.split(c_state, self.opt.dec_num_layers, axis=1)
                    h_states = tf.split(h_state, self.opt.dec_num_layers, axis=1)

                elif self.opt.enc_num_layers == 1 and self.opt.dec_num_layers > 1:
                    # Initialize each decoder layer with the a copy of the final encoder layer
                    c_states = [c_state] * self.opt.dec_num_layers
                    h_states = [h_state] * self.opt.dec_num_layers

                else:
                    # Project encoder's hidden and cell states so as to match the decoder state's dimensionality
                    final_state = tf.concat([c_state, h_state], 0)
                    projected_state = tf.nn.xw_plus_b(
                        final_state, state_projection_weights, state_projection_biases)
                    c_state, h_state = tf.split(projected_state, 2, axis=0)
                    c_states = tf.split(c_state, self.opt.dec_num_layers, axis=1)
                    h_states = tf.split(h_state, self.opt.dec_num_layers, axis=1)

                # Assemble the appropriate LSTM cell state tuple used to initialize the decoder
                encoded_state = tuple([tf.contrib.rnn.LSTMStateTuple(c_states[layer_id], h_states[layer_id]) for
                                       layer_id in range(self.opt.dec_num_layers)])

                # Assign computed values to surfaced attributes
                self.batch_length = batch_length
                self.batch_steps = batch_steps
                self.length_mask = length_mask
                self.embedding_table = embedding_table
                self.output_embedding_biases = output_embedding_biases
                self.rnn_outputs = rnn_outputs
                self.h_state = h_state
                self.encoded_state = encoded_state


class Decoder(object):
    """ Attentive RNN-LSTM decoder used to decode the low-dimensional, dense sentence encodings obtained from
    the corresponding encoder network within the context of a sequence auto-encoder model. """

    def __init__(self, vocab, opt, name):
        # Declare attributes
        self.vocab = vocab
        self.opt = opt
        self.name = name
        self.float_type = tf.float32
        self.int_type = tf.int32

        # Declare accessible values
        with tf.name_scope(self.name):
            # Initialize input placeholders
            self.static_keep_prob, self.rnn_keep_prob, self.sampling_bias = self.inputs_subgraph()
            # Keep track of sentence completions
            # i.e. the number of sequences within the currently decoded batch that have already terminated with <EOS>
            self.eos_tracker = None
            # Initialize outputs
            self.batch_length = None
            self.projected_rnn_outputs = None
            self.logits = None

    def inputs_subgraph(self):
        """ Creates the placeholders used to feed data to the model. """
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            # Read-in dropout probabilities from session input
            static_keep_prob = tf.placeholder(dtype=self.float_type, name='static_keep_prob')
            rnn_keep_prob = tf.placeholder(dtype=self.float_type, name='rnn_keep_prob')
            # Read in the probability with which the decoder is fed real prediction targets as input,
            # as opposed to its own predictions from the previous time-step (i.e. 'scheduled sampling')
            sampling_bias = tf.placeholder(dtype=self.float_type, name='sampling_bias')
        return static_keep_prob, rnn_keep_prob, sampling_bias

    def projection_subgraph(self):
        """ Defines the weight and bias parameters used to project RNN outputs into the embedding space following
        the completion of each full pass through the RNN. """
        with tf.variable_scope('decoder_projection'), tf.device('/gpu:0'):
            projection_weights = tf.get_variable(name='projection_weights',
                                                 shape=[self.opt.dec_hidden_dims, self.opt.embedding_dims],
                                                 dtype=self.float_type,
                                                 initializer=xi(uniform=False, dtype=self.float_type),
                                                 trainable=True)
            projection_biases = tf.get_variable(name='projection_biases', shape=[self.opt.embedding_dims],
                                                dtype=self.float_type,
                                                initializer=tf.zeros_initializer(dtype=self.float_type),
                                                trainable=True)
        return projection_weights, projection_biases

    def global_attention_subgraph(self):
        """ Defines the parameters for the global 'Luong' attention mechanism used during decoding;
        Publication: arxiv.org/pdf/1508.04025.pdf; With guidance from:
        github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py """
        with tf.variable_scope('global_decoder_attention'), tf.device('/gpu:0'):
            # Projects the encoder 'memories' (hidden states at each time step) to match decoder dimensions
            memory_key_weights = tf.get_variable(name='memory_key_weights',
                                                 shape=[self.opt.enc_hidden_dims, self.opt.dec_hidden_dims],
                                                 dtype=self.float_type,
                                                 initializer=xi(uniform=False, dtype=self.float_type),
                                                 trainable=True)
            # Used in computing the attention vector describing the alignment between input and output sequences
            attention_weights = tf.get_variable(name='attention_weights',
                                                shape=[self.opt.enc_hidden_dims + self.opt.dec_hidden_dims,
                                                       self.opt.dec_attention_dims],
                                                dtype=self.float_type,
                                                initializer=xi(uniform=False, dtype=self.float_type),
                                                trainable=True)
            # Used for combining the attention information with the decoder's input during the 'input feeding' step
            mixture_weights = tf.get_variable(name='mixture_weights',
                                              shape=[self.opt.embedding_dims + self.opt.dec_attention_dims,
                                                     self.opt.embedding_dims],
                                              dtype=self.float_type,
                                              initializer=xi(uniform=False, dtype=self.float_type),
                                              trainable=True)
            return memory_key_weights, attention_weights, mixture_weights

    def lstm_rnn_subgraph(self):
        """ Defines the forward pass through the specified LSTM-RNN. """
        with tf.variable_scope('lstm_rnn', reuse=None), tf.device('/gpu:0'):
            # Same functionality as within the encoder
            def _lstm_cell(model_opt):
                """ Defines a basic LSTM cell to which various wrappers can be applied. """
                base_cell = BasicLSTMCell(model_opt.dec_hidden_dims, forget_bias=2.5, state_is_tuple=True)
                if model_opt.allow_dropout:
                    base_cell = DropoutWrapper(base_cell, output_keep_prob=self.rnn_keep_prob)
                return base_cell

            if self.opt.dec_num_layers > 1:
                dec_cell = MultiRNNCell([_lstm_cell(self.opt) for _ in range(self.opt.dec_num_layers)])
            else:
                dec_cell = _lstm_cell(self.opt)
        return dec_cell

    def __call__(self, dec_input_idx, enc):
        """ Decodes the sentence encoder generated by the encoder. """
        with tf.variable_scope(self.name):
            # Build computational sub-graphs
            # properties | projection | attention | raw_rnn
            projection_weights, projection_biases = self.projection_subgraph()
            memory_key_weights, attention_weights, mixture_weights = self.global_attention_subgraph()
            dec_cell = self.lstm_rnn_subgraph()

            with tf.name_scope('properties'), tf.device('/cpu:0'):
                # Get input properties required by downstream operations
                batch_length = tf.shape(dec_input_idx)[0]
                batch_steps = tf.shape(dec_input_idx)[1]
                # Determine lengths of individual input sequences within the processed batch to mask RNN output and
                # exclude <EOS> and <PAD> tokens from the reconstruction loss computation
                length_mask = tf.count_nonzero(tf.not_equal(dec_input_idx, self.vocab.pad_id),
                                               axis=1, keep_dims=False, dtype=self.int_type, name='length_mask')

            with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                # Embed the input indices; reuses the embedding table learned by the encoder
                input_data = tf.nn.embedding_lookup(enc.embedding_table, dec_input_idx, name='embeddings')
                if self.opt.allow_dropout:
                    input_data = tf.nn.dropout(input_data, self.static_keep_prob, name='dec_front_dropout')

            with tf.variable_scope('lstm_rnn'), tf.device('/gpu:0'):
                # Obtain sequences decoded from the encoder's sentence representations
                # <PAD> slice output by the decoder after each generated batch sequence has ended in <EOS>
                pad_step_idx = tf.fill([batch_length], self.vocab.pad_id)
                pad_step_embeds = tf.nn.embedding_lookup(enc.embedding_table, pad_step_idx, name='pad_step_embeds')

                # raw_rnn expects input to be served in form of a TensorArray
                inputs_ta = tf.TensorArray(size=batch_steps, dtype=self.float_type) \
                    .unstack(tf.transpose(input_data, perm=[1, 0, 2]), name='rnn_input_array')

                # Initial decoder state set equal to the final state of the encoder
                initial_state = enc.encoded_state

                # Initialize tensor for tracking sentence completion
                if self.eos_tracker is None:
                    self.eos_tracker = tf.zeros([batch_length], dtype=self.int_type)

                # Define the raw_rnn loop which allows for greater control over the generated output, as compared
                # to dynamic_rnn()
                def loop_fn(time, cell_output, cell_state, loop_state):
                    """ Defines the loop function utilized by the raw_rnn. """

                    # Helper function for obtaining the output embeddings
                    def _get_predictions():
                        """ Projects RNN-generated predictions into the vocabulary space and embeds them. """
                        # Flatten RNN output to two dimensions
                        flat_step_outputs = tf.reshape(cell_output, [-1, self.opt.dec_hidden_dims])
                        projected_step_outputs = tf.nn.xw_plus_b(
                            flat_step_outputs, projection_weights, projection_biases)
                        step_logits = tf.nn.xw_plus_b(projected_step_outputs, tf.transpose(enc.embedding_table),
                                                      enc.output_embedding_biases, name='logits')
                        # Isolate highest-probability predictions
                        predicted_scores = tf.nn.softmax(step_logits, -1)
                        idx_predictions = tf.cast(tf.argmax(predicted_scores, axis=-1), dtype=self.int_type)
                        # Embed predicted word indices
                        embedded_predictions = tf.nn.embedding_lookup(enc.embedding_table, idx_predictions)
                        return idx_predictions, embedded_predictions

                    def _attend():
                        """ Executes the decoding-with-attention mechanism utilizing global 'Luong' attention. """
                        # Project encoder hidden states, 'memories', to match the dimensionality of the decoder,
                        # i.e. target, hidden states
                        memory_values = enc.rnn_outputs
                        flat_values = tf.reshape(memory_values, [-1, tf.shape(memory_values)[-1]])
                        flat_keys = tf.matmul(flat_values, memory_key_weights)
                        memory_keys = tf.reshape(
                            flat_keys, [enc.batch_length, enc.batch_steps, self.opt.dec_hidden_dims])

                        # Apply length to the memory keys so as to restrict attention to non-padded positions
                        score_mask = tf.sequence_mask(
                            enc.length_mask, maxlen=tf.reduce_max(enc.length_mask), dtype=self.float_type)
                        score_mask = tf.expand_dims(score_mask, -1)
                        score_mask = tf.matmul(
                            score_mask, tf.ones([enc.batch_length, self.opt.dec_hidden_dims, 1]), transpose_b=True)
                        memory_keys = tf.where(tf.cast(score_mask, dtype=tf.bool), memory_keys,
                                               tf.zeros_like(memory_keys))

                        # Obtain target query, i.e. the current decoder hidden state
                        target_hidden_state = cell_state[-1][-1]
                        target_query = tf.expand_dims(target_hidden_state, 1)

                        # Compute alignments globally, by attending to all encoder states at once
                        score = tf.matmul(target_query, memory_keys, transpose_b=True)
                        score = tf.squeeze(score, [1])
                        alignments = tf.nn.softmax(score)

                        # Compute the context vector by applying calculated alignments to encoder states
                        expanded_alignments = tf.expand_dims(alignments, 1)
                        context = tf.matmul(expanded_alignments, memory_values)
                        context = tf.squeeze(context, [1])

                        # Compute the attentional vector by combining encoder context with decoder query
                        attention = tf.tanh(tf.matmul(tf.concat([context, target_hidden_state], -1), attention_weights))
                        return attention

                    # Initialize the loop function
                    emit_output = cell_output  # no output is emitted during initialization
                    next_loop_state = None
                    # Check if to terminate the loop;
                    # length slack denotes how much longer the output sequence is allowed to be than the input
                    elements_finished = tf.greater_equal(time, length_mask + self.opt.length_slack)
                    # Once stopping conditions are met for all batch elements, terminate loop
                    finished = tf.reduce_all(elements_finished)

                    if cell_output is None:  # i.e. during initialization only
                        # Set initial values
                        self.eos_tracker *= 0
                        next_cell_state = initial_state
                        next_input = inputs_ta.read(0)

                    # At time-step 1+
                    else:
                        # Pass on the cell state
                        next_cell_state = cell_state
                        # Get predictions from previous time-step
                        predicted_idx, predicted_embeds = _get_predictions()
                        # Check if stopping conditions are met
                        # 1. Check if all decoded batch items contain an <EOS> prediction
                        self.eos_tracker += tf.cast(tf.equal(predicted_idx, self.vocab.eos_id), self.int_type)
                        # 2. Check if all decoded batch items are equal in length to corresponding encoder inputs
                        boundary_reached = tf.greater_equal(time, length_mask)
                        if not self.opt.is_train or not self.opt.use_reconstruction_objective:
                            # Extended stopping criterion during inference,
                            # as output length is allowed to exceed input length via the slack_length parameter
                            self.eos_tracker += tf.cast(tf.equal(predicted_idx, self.vocab.eos_id), self.int_type)
                            elements_finished = tf.logical_or(tf.greater(self.eos_tracker, 0),
                                                              tf.greater_equal(
                                                                  time, (length_mask + self.opt.length_slack)))
                            finished = tf.reduce_all(elements_finished)

                        # Scheduled sampling: If flip value is smaller than sampling probability, the output of the
                        # decoder at the current time-step is fed as input to the decoder at the subsequent time-step
                        flip = tf.random_uniform(shape=[], minval=0.0, maxval=1.0)
                        input_tensor = tf.cond(tf.logical_or(
                            tf.less(self.sampling_bias, flip), tf.reduce_all(boundary_reached)),
                            lambda: predicted_embeds, lambda: inputs_ta.read(time))
                        # If stopping conditions have been met, output a <PAD> slice, then terminate loop
                        next_input = tf.cond(finished, lambda: pad_step_embeds, lambda: input_tensor)

                        if self.opt.attentive_decoding:
                            # Input feeding: Combine attentive information with the input to the decoder at the
                            # subsequent time-step (either target tokens or predictions from the current time-step)
                            attentional_vector = _attend()
                            next_input = tf.matmul(tf.concat([next_input, attentional_vector], -1),
                                                   mixture_weights)

                    return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

                # Get RNN outputs
                rnn_outputs_tensor_array, final_state, _ = tf.nn.raw_rnn(dec_cell, loop_fn)
                rnn_outputs = rnn_outputs_tensor_array.stack()
                rnn_outputs = tf.transpose(rnn_outputs, perm=[1, 0, 2])
                flat_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.opt.enc_hidden_dims],
                                              name='reshaped_rnn_outputs')
                # Project RNN outputs into the embedding space, followed by the projection into vocabulary space
                projected_rnn_outputs = tf.nn.xw_plus_b(flat_rnn_outputs, projection_weights, projection_biases)
                logits = tf.nn.xw_plus_b(
                    projected_rnn_outputs, tf.transpose(enc.embedding_table), enc.output_embedding_biases,
                    name='logits')

            # Assign computed values to the surfaced outputs
            self.batch_length = batch_length
            self.projected_rnn_outputs = projected_rnn_outputs
            self.logits = logits


class SeqAE(object):
    """ Sequence autoencoder object linking together the encoder and decoder networks for training and inference. """

    def __init__(self, vocab, opt, name, inference_only=False):
        # Declare attributes
        self.vocab = vocab
        self.opt = opt
        self.name = name
        self.enc_name = 'encoder'
        self.dec_name = 'decoder'
        self.inference_only = inference_only
        self.float_type = tf.float32
        self.int_type = tf.int32

        with tf.name_scope(self.name):
            # Initialize input placeholders
            self.lr = self.inputs_subgraph()

            # Initialize encoder and decoder networks
            self.enc = Encoder(self.vocab, self.opt, self.enc_name)
            self.dec = Decoder(self.vocab, self.opt, self.dec_name)

            # Declare accessible values
            self.predicted_scores = None
            self.predicted_idx = None
            self.predicted_idx_eos = None
            self.last_prediction = None
            self.loss_avg = None
            self.loss_regularized = None
            self.train_op = None

    def inputs_subgraph(self):
        """ Creates the placeholders used to feed data to the model. """
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            # Read-in the current learning rate value (possibly subject to decay) from session inputs
            lr = tf.placeholder(dtype=self.float_type, name='learning_rate')
        return lr

    def sampled_loss_subgraph(self, label_idx):
        """ Calculates the sampled softmax candidate sampling loss used to train the SAE in a sequence-to-sequence
        manner. """
        with tf.name_scope('sampled_loss'), tf.device('/gpu:0'):
            # Flatten labels to two dimensions
            flat_labels = tf.reshape(label_idx, [-1, 1])
            # 'div' partition strategy assures consistency with use of softmax at inference time
            sampled_loss = tf.nn.sampled_softmax_loss(weights=self.enc.embedding_table,
                                                      biases=self.enc.output_embedding_biases,
                                                      labels=flat_labels,
                                                      inputs=self.dec.projected_rnn_outputs,
                                                      num_sampled=self.opt.samples,
                                                      num_classes=self.vocab.n_words, num_true=1,
                                                      sampled_values=self.opt.sampled_values,
                                                      remove_accidental_hits=self.opt.remove_accidental_hits,
                                                      partition_strategy='div', name='sampled_loss')
            # RNN predictions corresponding to padded input positions are masked out,
            # so as to not affect the incurred loss
            loss_mask = tf.cast(tf.not_equal(tf.reshape(flat_labels, [-1]), self.vocab.pad_id),
                                self.float_type)
            masked_loss = tf.multiply(sampled_loss, loss_mask, name='masked_sampled_loss')
            sampled_loss_avg = tf.reduce_mean(masked_loss, name='sampled_loss_avg')
        return sampled_loss_avg

    def cross_ent_loss_subgraph(self, label_idx):
        """ Calculates the cross-entropy loss used to evaluate the model during validation steps. """
        with tf.name_scope('cross_entropy_loss'), tf.device('/gpu:0'):
            labels_flat = tf.reshape(label_idx, [-1])
            cross_ent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_flat,
                                                                            logits=self.dec.logits,
                                                                            name='cross_entropy_loss')
            # Masking is applied as for sampled softmax loss calculation
            loss_mask = tf.cast(tf.not_equal(labels_flat, self.vocab.pad_id), self.float_type)
            masked_loss = tf.multiply(cross_ent_loss, loss_mask, name='masked_cross_ent_loss')
            cross_ent_loss_avg = tf.reduce_mean(masked_loss, name='cross_ent_loss_avg')
        return cross_ent_loss_avg

    def optimization_subgraph(self, loss_avg):
        """ Defines the optimization procedure for the entire model. """
        with tf.variable_scope('optimization'), tf.device('/gpu:0'):
            # Variable tracking the optimization steps
            global_step = tf.get_variable(shape=[], name='global_step', dtype=self.int_type,
                                          initializer=tf.constant_initializer(0, dtype=self.int_type),
                                          trainable=False)
            # All trainable variables are optimized jointly
            t_vars = tf.trainable_variables()
            # Apply L2 regularization by imposing Gaussian priors on model's parameters
            loss_l2 = tf.add_n([tf.nn.l2_loss(var) for var in t_vars if len(var.shape) > 1]) * self.opt.l2_beta
            loss_regularized = tf.add(loss_avg, loss_l2, name='loss_regularized')
            # Calculate gradients for backpropagation with respect to the regularized loss
            grads = tf.gradients(loss_regularized, t_vars)
            # Define optimization OP
            optimizer = tf.train.AdamOptimizer(self.lr, name='optimizer')
            train_op = optimizer.apply_gradients(zip(grads, t_vars), global_step=global_step, name='train_op')
        return loss_regularized, grads, train_op

    def prediction_subgraph(self):
        """ Defines the generative process taking place during inference. """
        # Sub-graph contains some level of redundancy, as the decoder performs projection steps already
        with tf.variable_scope('prediction'), tf.device('/gpu:0'):
            # Isolate highest-probability RNN predictions for every time-step
            predicted_scores = tf.reshape(tf.nn.softmax(self.dec.logits, -1, name='predictions'),
                                          [self.dec.batch_length, -1, self.vocab.n_words])
            predicted_idx = tf.cast(tf.argmax(predicted_scores, axis=-1), dtype=self.int_type)
            # Predictions at last step only are used for beam-search prediction
            last_prediction = predicted_scores[:, -1, :]

            # Mask output past the <EOS> boundary, optionally also masking the <EOS> tag itself
            # First, identify <EOS> positions within each predicted sequence
            check_eos = tf.cast(tf.equal(predicted_idx, self.vocab.eos_id), dtype=self.int_type)
            # Flatten to one dimension, as required by tf.where()
            check_range = tf.range(tf.shape(check_eos)[-1], 0, -1, dtype=self.int_type)
            range_eos = tf.multiply(check_eos, check_range)
            eos_idx = tf.cast(tf.argmax(range_eos, axis=-1), dtype=self.int_type)

            # Assemble outputs masks
            max_sent_lens = tf.cast(tf.fill(tf.shape(eos_idx), tf.shape(predicted_idx)[1]), dtype=self.int_type)
            no_eos = tf.where(tf.equal(eos_idx, 0), max_sent_lens, eos_idx)
            # no_eos masks <EOS> tags
            shift_non_final = tf.cast(tf.not_equal(no_eos, max_sent_lens), dtype=self.int_type)
            final_eos = tf.add(no_eos, shift_non_final)
            # final_eos preserves sentence-final <EOS> tags
            no_eos_mask = tf.sequence_mask(no_eos, tf.shape(predicted_idx)[1], dtype=self.int_type)
            final_eos_mask = tf.sequence_mask(final_eos, tf.shape(predicted_idx)[1], dtype=self.int_type)
            # Apply masks to the RNN predictions
            pre_pad = tf.subtract(predicted_idx, self.vocab.pad_id)
            predicted_idx = tf.add(tf.multiply(pre_pad, no_eos_mask), self.vocab.pad_id)
            predicted_idx_eos = tf.add(tf.multiply(pre_pad, final_eos_mask), self.vocab.pad_id)
        return predicted_scores, predicted_idx, predicted_idx_eos, last_prediction

    def enc_call(self, enc_input_idx):
        """ Passes input data through the encoder module of the SAE only, to allow for a more efficient computation of
        adversarial loss within the IDGAN system. """
        self.enc(enc_input_idx)

    def dec_call(self, dec_input_idx, label_idx):
        """ Passes input data through the decoder. """
        self.dec(dec_input_idx, self.enc)

        # Decodes encoded sequences
        self.predicted_scores, self.predicted_idx, self.predicted_idx_eos, self.last_prediction = \
            self.prediction_subgraph()

        if not self.inference_only:
            # Only accessed when reconstruction loss is calculated
            if self.opt.use_reconstruction_objective:
                assert (label_idx is not None), \
                    'Training with the reconstruction objective requires target labels to not be None.'
                if self.opt.is_train:
                    # Calculate sampled softmax reconstruction loss
                    self.loss_avg = self.sampled_loss_subgraph(label_idx)
                else:
                    # Calculate full softmax plus cross-entropy reconstruction loss during validation
                    self.loss_avg = self.cross_ent_loss_subgraph(label_idx)

    def __call__(self, enc_input_idx, dec_input_idx=None, label_idx=None, reuse=False, encode_only=False):
        """ Passes input data through the full sequence autoencoder graph for the computation of reconstruction loss,
        information density associated with decoded sequences and information density reduction between encoded and
        decoded sequences."""
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # Traverse the encoder
            self.enc_call(enc_input_idx)
            if not encode_only:
                # Traverse the decoder
                self.dec_call(dec_input_idx, label_idx)

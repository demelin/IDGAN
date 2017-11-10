""" Interface object defined for the cognitive language model. Covers generation (both greedy and with beam-search),
as well as the calculation of various measurements related to information density, such as surprisal, entropy reduction,
and UID divergence (i.e. the distance between true UID and the observed information density distribution).
Main references:
[Frank, 2012]: onlinelibrary.wiley.com/doi/10.1111/tops.12025/full,
[Hale, 2006]: onlinelibrary.wiley.com/doi/10.1207/s15516709cog0000_64/full """

import numpy as np
import tensorflow as tf

from cognitive_language_model.src.codebase.batching import index_sentence, index_sentence_batch


class CogLMInterface(object):
    """ An interface for the IDGAN-internal language model, used to generate text greedily or through beam-search
    as well as for the estimation of ID-relevant measurements. """

    def __init__(self, model, vocab, session, opt):
        self.model = model
        self.vocab = vocab
        self.session = session
        self.opt = opt

    def infer_step(self, input_array, state=None, from_previous=False):
        """ Performs a single inference step. """
        if from_previous:
            assert (state is not None), 'Prefix state required when generating from a prefix!'
        if state is None:
            # Initialize a 'clean' RNN cell state as an array of zeros
            state = np.zeros((self.opt.num_layers, 2, self.opt.batch_size, self.opt.hidden_dims))
        # Variable values passed to the model graph
        feed_dict = {
            self.model.input_idx: input_array,
            self.model.rnn_state: state,
            self.model.static_keep_prob: 1.0,
            self.model.rnn_keep_prob: 1.0,
        }
        # OPs called within the model graph
        final_state, final_prediction = self.session.run(
            [self.model.final_state, self.model.final_prediction], feed_dict=feed_dict)
        return final_state, final_prediction

    def calculate_probabilities(self, input_array):
        """ Calculates word-wise probabilities for the input string or batch of strings;
        input should not be padded. """
        probabilities = list()
        step_state = None

        # Obtain a predictive probability distribution from the language model for a single time-step
        # Note: Obtained probability predictions are for word tokens at the current time step, so the input to the LM
        # is the sentence prefix up to and including the previous time-step
        for i in range(input_array.shape[1]):
            if i == 0:
                step_state, step_prediction = self.infer_step(
                    np.expand_dims(np.array([self.vocab.eos_id] * input_array.shape[0], dtype=np.int32), 1),
                    state=None, from_previous=False)
            else:
                step_state, step_prediction = self.infer_step(input_array[:, [i - 1]], state=step_state,
                                                              from_previous=True)
            # Isolate probability scores predicted for the input tokens at the current time-step
            step_input = input_array[:, [i]]
            flat_index = np.expand_dims(np.arange(
                step_prediction.shape[0]) * step_prediction.shape[1], 1) + step_input
            step_probs = np.take(step_prediction, flat_index)
            probabilities.append(np.reshape(step_probs, step_probs.shape))
        return probabilities

    def initialize_beam(self, prefix):
        """ Initializes a beam for beam-search assisted generation and entropy reduction estimation. """

        def _from_none():
            """ Initializes a beam from scratch. """
            eos_slice = np.array([[self.vocab.eos_id]] * self.opt.batch_size, dtype=np.int32)
            # Pick a random initial word to diversify generated sentences
            initial_word = np.random.randint(0, self.vocab.n_words, [self.opt.batch_size, 1])
            beam_prefix = np.concatenate([eos_slice, initial_word], 1)
            return beam_prefix

        def _from_string(string_prefix):
            """ Initializes a beam from an existing string prefix. """
            idx_prefix = index_sentence(string_prefix, self.vocab, self.opt)
            beam_prefix = np.concatenate(
                [np.array([[self.vocab.eos_id]] * self.opt.batch_size, dtype=np.int32), idx_prefix], 1)
            return beam_prefix

        beam = list()
        # Initialize beam entries; the number is determined by the specified beam width
        # Beam entries are tuples of the form (generated sequence, sequence probability)
        for _ in range(self.opt.beam_width):
            if prefix is None:
                beam_init = _from_none()
            else:
                beam_init = _from_string(prefix)
            beam_tpl = (np.array([[1.0]] * self.opt.batch_size, dtype=np.float32), beam_init)
            beam.append(beam_tpl)

        return beam

    def generate(self, prefix=None, print_results=False):
        """ Generates sentences generated using beam search; output probabilities are tracked for successive
        beam pruning performed to retain only most probable output sequences; beam search implementation is
        lightly inspired by evolutionary processes. """
        # Initialize the generation beam
        beam = self.initialize_beam(prefix)
        # Track sentences ending in <EOS>;
        # generation is concluded once all beam items contain a sentence-final <EOS>
        # Initialize containers and tracking variables
        finalists = list()
        next_beam = list()
        beam_width = self.opt.beam_width
        # Counteracts the model's tendency towards quick termination
        min_length = 10
        gen_step = 0

        # Generate output sequences with beam search
        while len(finalists) < self.opt.beam_width:
            # Uncomment the desired beam candidate tracking method
            # candidates = list()
            # Discrete 'populations' within the beam guarantee a more varied output
            # Each population is seeded with a different random initial word (see infer_step())
            candidates = [[] for _ in range(beam_width)]

            # Populate beam until all beam-items end with <EOS> or the maximum defined generation length is reached
            for beam_id in range(beam_width):
                step_input = beam[beam_id][1]
                _, step_prediction = self.infer_step(step_input, state=None, from_previous=False)  # read prefix
                # Select the top-n predicted continuations for each beam item (where n corresponds to beam width)
                step_prediction = tf.convert_to_tensor(step_prediction, dtype=tf.float32)
                best_scores, best_idx = tf.nn.top_k(step_prediction, k=beam_width + 2)

                # Construct new beam items by extending previous ones with new information obtained from the model
                for item_id in range(beam_width + 2):
                    item_score = best_scores.eval()[:, [item_id]]
                    item_idx = best_idx.eval()[:, [item_id]]
                    if item_idx == self.vocab.unk_id:
                        continue
                    if gen_step < min_length and item_idx == self.vocab.eos_id:
                        continue
                    else:
                        candidate = (np.multiply(beam[beam_id][0], item_score),
                                     np.concatenate([beam[beam_id][1], item_idx], 1))
                        # candidates.append(candidate)
                        candidates[beam_id].append(candidate)

            # Uncomment the desired pruning strategy
            # Strategy 1: Retain n-best candidates; yields better output quality
            # sorted(candidates, key=lambda x: x[0], reverse=True)
            # Strategy 2: Retain random candidates; yields slightly greater output diversity
            # random.shuffle(candidates)
            # beam = candidates[: beam_width]
            # Strategy 3: Output speciation; yields substantially greater output diversity
            candidates = [sorted(c_set, key=lambda x: x[0], reverse=True) for c_set in candidates]
            if prefix is None:
                beam = [c_set[0] for c_set in candidates]
            else:
                if gen_step == 0:
                    beam = [candidates[i][i] for i in range(len(candidates))]
                else:
                    beam = [candidates[i][0] for i in range(len(candidates))]

            # Check if any of the beam items extended during the current step terminate in <EOS>
            # If so, constrain the active beam width, retaining the completed sequence
            for beam_item in beam:
                if beam_item[1][:, -1][0] == self.vocab.eos_id:
                    finalists.append(beam_item)
                    beam_width -= 1
                else:
                    next_beam.append(beam_item)
            beam = next_beam
            next_beam = list()

            # Terminate generation once all candidates have reached maximum length (input length + length slack)
            # Unless all beam items have ended with <EOS> previously
            if self.opt.max_gen_len is not None:
                if len(beam) > 0 and beam[0][1].shape[1] >= self.opt.max_gen_len:
                    finalists += beam
                    break
            # Update tracker variable
            gen_step += 1

        # Translate final beam predictions into sequences of word tokens
        sequence_tuples = list()
        for beam_item in finalists:
            idx_to_words = [self.vocab.index_to_word[beam_item[1][0][step]] for step in range(beam_item[1].shape[1])]
            sequence = ' '.join(idx_to_words)
            sequence_tuples.append((sequence, beam_item[0][0][0]))
        if print_results:
            print('Generation with beam search of size {:d} yielded following sentences:'.format(self.opt.beam_width))
            for stpl in sequence_tuples:
                print('{:s} | Probability={:.10f}'.format(stpl[0], stpl[1]))
        else:
            return sequence_tuples

    def check_eos(self, top_scores, top_idx, last=False):
        """ Helper function for entropy reduction estimation, checks for and discards premature <EOS> predictions. """
        # Checks most recent ER-beam item continuations for <EOS> tags
        non_eos = tf.not_equal(top_idx, self.vocab.eos_id)
        num_kept = tf.count_nonzero(non_eos)
        if (num_kept.eval() > self.opt.er_width * self.opt.batch_size) or (last is True):
            # If no match is found, prune the beam by discarding the least-likely continuation
            top_scores = top_scores.eval()[:, : -1]
            top_idx = top_idx.eval()[:, : -1]
        else:
            # If <EOS> is among the predicted continuations, prune the beam by discarding the item terminating in <EOS>
            masked_scores = tf.boolean_mask(top_scores, non_eos).eval()
            masked_idx = tf.boolean_mask(top_idx, non_eos).eval()
            # Reshape back into an input-like format
            top_scores = np.reshape(masked_scores, [top_scores.eval().shape[0], -1])
            top_idx = np.reshape(masked_idx, [top_idx.eval().shape[0], -1])
        return top_scores, top_idx

    def simplified_entropy(self, prefix):
        """ Calculates the simplified entropy at the current point in sentence processing
        in accordance with Frank, 2012. """
        # Deviation from cited publication: At the initial time-step, only top-n candidate continuations are considered,
        # rather than every single vocabulary item
        # Initialize entropy beam, format = (sequence probability, sequence of word indices, <EOS> mask)
        entropy_beam = self.initialize_beam(prefix)
        extensions = list()
        # Generate most likely sentence continuations up to a specified ER look-ahead distance
        for step in range(self.opt.er_lookahead):
            if step > 0:
                entropy_beam = extensions
                extensions = list()
            for beam_item in entropy_beam:
                # Construct new ER beam items by extending previous ones with new information obtained from the model
                step_input = beam_item[1]
                _, step_prediction = self.infer_step(step_input, state=None, from_previous=False)
                # Select the top-n predicted continuations for each beam item (where n corresponds to ER beam width)
                step_prediction = tf.convert_to_tensor(step_prediction, dtype=tf.float32)
                top_scores, top_idx = tf.nn.top_k(step_prediction, k=self.opt.er_width)
                for s in range(self.opt.er_width):
                    # Check for <EOS> predictions
                    candidate_scores = top_scores.eval()[:, [s]]
                    candidate_idx = top_idx.eval()[:, [s]]
                    # Apply <EOS> mask to pad out completed sequences kept within the ER beam
                    candidate_idx[beam_item[2] == 0] = self.vocab.pad_id
                    candidate_scores[beam_item[2] == 0] = 1.0
                    # Update <EOS> mask to include the latest LM prediction
                    new_mask = np.not_equal(candidate_idx, self.vocab.eos_id)
                    # Construct a new beam item by extending the predicted sequence and updating the probability score
                    candidate = (np.multiply(beam_item[0], candidate_scores),
                                 np.concatenate([beam_item[1], candidate_idx], 1),
                                 new_mask)
                    extensions.append(candidate)
        # Calculate simplified entropy according to definition in Frank, 2012
        simplified_entropy = - np.sum([np.multiply(beam_item[0], np.log(beam_item[0]))
                                       for beam_item in entropy_beam], axis=0)
        return simplified_entropy

    def entropy_reduction(self, input_array, length_mask):
        """ Calculates the total simplified entropy reduction associated with the input string in accordance with
        the methodology outlined in Frank, 2012. """
        er_cache = list()
        # Initial sentence entropy; exact implementation unclear from referenced publication
        prior_entropy = self.simplified_entropy(None)
        for i in range(input_array.shape[1]):
            step_entropy = self.simplified_entropy(input_array[:, 0: i + 1])
            # Mask values at <PAD> positions, to omit them from the entropy reduction calculation
            step_mask = length_mask[:, [i]]
            step_entropy *= step_mask
            # Calculate approximate entropy reduction at the current time-step
            step_er = prior_entropy - step_entropy
            er_cache.append(step_er)
            prior_entropy = step_entropy

        # Calculate approximate reduction for the full sentence
        er_array = np.concatenate(er_cache, 1)
        # Exclude negative values, as suggested in Hale, 2006
        er_array[er_array < 0.0] = 0.0
        if er_array.shape[0] > 1:
            sentence_er = np.sum(er_array, axis=1, keepdims=True)
        else:
            sentence_er = np.sum(er_array, keepdims=True)
        return sentence_er, er_array

    def ready_input(self, input_data):
        """ Formats input sentences to be compatible with the language model implementation. """
        inputs = list()
        if isinstance(input_data, list):
            # In case input is a list of sentences (e.g. sampled from a corpus)
            batched_data, length_mask = index_sentence_batch(input_data, self.vocab, self.opt)
            inputs += [batched_data, length_mask]
        elif isinstance(input_data, str):
            # In case input is a single string (e.g. provided by user at test time)
            batched_data = index_sentence(input_data, self.vocab, self.opt)
            inputs.append(batched_data)
        else:
            raise ValueError('Please provide a string or a list of strings as input to the interface.')
        return inputs

    def get_probability(self, input_data):
        """ Returns the probability of the input data; takes either a single sentence string or a list of sentence
        strings as input, as do other getters. """
        # Format input and get word-wise probability predictions from the langauge model
        inputs = self.ready_input(input_data)
        probabilities = self.calculate_probabilities(inputs[0])
        prob_array = np.concatenate(probabilities, 1)
        # Mask predictions at <PAD> and <EOS> positions
        if len(inputs) > 1:
            masked_array = prob_array * inputs[1]
            # Mask replaces predicted values with 1.0, thus negating their contribution
            # when computing probability products or sums of log-probability scores
            masked_array[masked_array == 0.0] = 1.0
        else:
            masked_array = np.copy(prob_array)
        total_prob = np.prod(masked_array, axis=1, keepdims=True)
        return total_prob, prob_array, masked_array

    def get_log_probability(self, input_data):
        """ Returns the word-wise and mean log-probability for the input sentences. """
        _, _, masked_array = self.get_probability(input_data)
        log_prob_array = np.log2(masked_array)
        total_log_prob = np.sum(log_prob_array, axis=1, keepdims=True)
        return total_log_prob, log_prob_array, masked_array

    def get_surprisal(self, input_data):
        """ Returns the word-wise and mean surprisal and UID divergence scores for the input sentences. """
        _, _, masked_array = self.get_probability(input_data)
        # Computes per-word, total, and normalized surprisal
        surprisal_array = np.log2(1 / masked_array)
        total_surprisal = np.sum(surprisal_array, axis=1, keepdims=True)
        normalized_surprisal = total_surprisal / np.sum(np.not_equal(masked_array, 1.0), axis=1, keepdims=True)

        # Computes per-word, total, and normalized UID divergence
        shifted = np.concatenate((np.zeros([surprisal_array.shape[0], 1], dtype=np.float32),
                                  surprisal_array[:, :-1]), axis=1)
        uiddiv_array = np.abs(surprisal_array - shifted)
        total_uiddiv = np.sum(uiddiv_array, axis=1, keepdims=True)  # low values are closer to UID
        normalized_uiddiv = total_uiddiv / np.sum(np.not_equal(masked_array, 1.0), axis=1, keepdims=True)
        return total_surprisal, surprisal_array, normalized_surprisal, total_uiddiv, uiddiv_array, normalized_uiddiv

    def get_entropy_reduction(self, input_data):
        """ Returns an entropy reduction estimate for the input sentences. """
        inputs = self.ready_input(input_data)
        if len(inputs) > 1:
            length_mask = inputs[1]
        else:
            length_mask = np.ones(inputs[0].shape, dtype=np.int32)
        total_er, er_array = self.entropy_reduction(inputs[0], length_mask=length_mask)
        normalized_er = total_er / np.sum(np.not_equal(length_mask, 0), axis=1, keepdims=True)
        return total_er, er_array, normalized_er

    def get_cognitive_load(self, input_data):
        """ Returns the cognitive load assigned to the input sentences as a weighted sum of mean surprisal and
        entropy reduction scores. """
        # Obtain surprisal and ER estimates
        total_surprisal, surprisal_array, normalized_surprisal, _, _, _ = self.get_surprisal(input_data)
        total_er, er_array, normalized_er = self.get_entropy_reduction(input_data)
        # Calculate CL
        total_cl = total_surprisal - (self.opt.cl_weight * total_er)
        cl_array = surprisal_array - (self.opt.cl_weight * er_array)
        normalized_cl = normalized_surprisal - (self.opt.cl_weight * normalized_er)
        return total_cl, cl_array, normalized_cl

    def get_sequence_perplexity(self, input_data):
        """ Returns the sentence-wise model perplexity score assigned to the input sentence by the trained LM. """
        total_log_prob, _, masked_array = self.get_log_probability(input_data)
        sequence_lengths = np.sum(np.not_equal(masked_array, 1.0), axis=1, keepdims=True)
        return np.power(2, -(np.divide(total_log_prob, sequence_lengths)))

    def get_model_perplexity(self, input_data):
        """ Returns the model perplexity score for the full test corpus. """
        corpus_log_prob = np.zeros([1, 1], dtype=np.float32)
        corpus_num_words = 0
        # Iterate over contents of the test set
        for i in range(len(input_data)):
            try:
                batch = input_data[i: i + self.opt.batch_size]
            except IndexError:
                break
            total_log_prob, _, masked_array = self.get_log_probability(batch)
            sequence_lengths = np.sum(np.not_equal(masked_array, 1.0), axis=1, keepdims=True)
            # Accumulate sentence-wise log probabilities
            corpus_log_prob += np.sum(total_log_prob)
            corpus_num_words += np.sum(sequence_lengths)
        # Return the model perplexity score for the test set
        return np.power(2, -(corpus_log_prob / corpus_num_words))

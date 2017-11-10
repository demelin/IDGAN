""" Interface object defined for the sequence autoencoder model. Incorporates methods used in generation through
encoding-decoding. So far, greedy generation and generation with beam-search are covered; however, beam-search
generation is limited use for the reconstruction of input sequences. """

import numpy as np
import tensorflow as tf

from cognitive_language_model.src.codebase.batching import index_sentence


class SeqAEInterface(object):
    """ An interface for the sequence autoencoder, used to generate text either greedily or via beam search. """

    def __init__(self, model, vocab, session, opt):
        self.model = model
        self.vocab = vocab
        self.session = session
        self.opt = opt

    def infer_step(self, enc_input, dec_input, sampling_bias):
        """ Performs a single inference step. """
        # Variable values passed to the model graph
        feed_dict = {
            self.model.encoder.input_idx: enc_input,
            self.model.encoder.static_keep_prob: 1.0,
            self.model.encoder.rnn_keep_prob: 1.0,
            self.model.decoder.input_idx: dec_input,
            self.model.decoder.static_keep_prob: 1.0,
            self.model.decoder.rnn_keep_prob: 1.0,
            self.model.decoder.sampling_bias: sampling_bias
        }
        # OPs called within the model graph
        ops = [self.model.predicted_scores, self.model.predicted_idx_eos, self.model.last_prediction]
        # OP output is returned as numpy arrays
        predicted_scores, predicted_idx_eos, last_prediction = self.session.run(ops, feed_dict=feed_dict)
        return predicted_scores, predicted_idx_eos, last_prediction

    def greedy_generation(self, enc_input, dec_input, sampling_bias=0.0):
        """ Encode-decodes sentences greedily; decoder production is conditioned on sentence encodings generated by
        the encoder. """
        # Setting sampling_bias to 0.0 results in decoder always receiving its output from previous time-step
        # as input during current time-step
        _, batch_idx, _ = self.infer_step(enc_input, dec_input, sampling_bias)  # 0.0 == always from previous
        # Convert predicted word indices to word tokens
        batch_idx = [np.squeeze(array).tolist() for array in np.split(batch_idx, batch_idx.shape[0], axis=0)]
        # Assemble output sequences from predictions; truncate output after the sentence-final <EOS> tag
        batch_boundaries = [idx_list.index(self.vocab.eos_id) if self.vocab.eos_id in idx_list else len(idx_list)
                            for idx_list in batch_idx]
        batch_sentences = [[self.vocab.index_to_word[idx] for idx in batch_idx[i][:batch_boundaries[i]]]
                           for i in range(len(batch_idx))]
        batch_sentences = [' '.join(word_list) + '.' for word_list in batch_sentences]
        return batch_sentences

    def initialize_beam(self, prefix):
        """ Initializes a beam for generation with beam search. """

        def _from_none():
            """ Initializes a beam from scratch. """
            beam_prefix = np.array([[self.vocab.go_id]] * self.opt.batch_size, dtype=np.int32)
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

    def beam_generation(self, enc_input, prefix=None, print_results=False, sampling_bias=1.0):
        """ Generates sentences generated using beam search; output probabilities are tracked for successive
        beam pruning performed to retain only most probable output sequences. """
        # Initialize the generation beam
        beam = self.initialize_beam(prefix)
        # Track sentences ending in <EOS>;
        # generation is concluded once all beam items contain a sentence-final <EOS>
        # Initialize containers and tracking variables
        finalists = list()
        next_beam = list()
        beam_width = self.opt.beam_width
        gen_step = 0

        # Generate output sequences with beam search
        while len(finalists) < self.opt.beam_width:
            # Uncomment the desired beam candidate tracking method
            candidates = list()
            # Discrete 'populations' within the beam guarantee a more varied output
            # Each population is seeded with a different random initial word (see infer_step())
            # candidates = [[] for _ in range(beam_width)]

            # Populate beam until all beam-items end with <EOS> or the maximum defined generation length is reached
            for beam_id in range(beam_width):
                # Beam item sequence generated during the previous step serves as input during the current step
                dec_input = beam[beam_id][1]
                _, _, step_prediction = self.infer_step(enc_input, dec_input, sampling_bias)
                # Select the top-n predicted continuations for each beam item (where n corresponds to beam width)
                step_prediction = tf.convert_to_tensor(step_prediction, dtype=tf.float32)
                best_scores, best_idx = tf.nn.top_k(step_prediction, k=beam_width)

                # Construct new beam items by extending previous ones with new information obtained from the model
                for item_id in range(beam_width):
                    item_score = best_scores.eval()[:, [item_id]]
                    item_idx = best_idx.eval()[:, [item_id]]
                    candidate = (np.multiply(beam[beam_id][0], item_score),
                                 np.concatenate([beam[beam_id][1], item_idx], 1))
                    candidates.append(candidate)
                    # candidates[beam_id].append(candidate)

            # Uncomment the desired pruning strategy
            # Strategy 1: Retain n-best candidates; yields better output quality
            sorted(candidates, key=lambda x: x[0], reverse=True)
            # Strategy 2: Retain random candidates; yields slightly greater output diversity
            # random.shuffle(candidates)
            beam = candidates[: beam_width]
            # Strategy 3: Output speciation; yields substantially greater output diversity
            # candidates = [sorted(c_set, key=lambda x: x[0], reverse=True) for c_set in candidates]
            # if prefix is None:
            #     beam = [c_set[0] for c_set in candidates]
            # else:
            #     if gen_step == 0:
            #         beam = [candidates[i][i] for i in range(len(candidates))]
            #     else:
            #        beam = [candidates[i][0] for i in range(len(candidates))]

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
            if len(beam) > 0 and beam[0][1].shape[-1] >= enc_input.shape[-1] + self.opt.length_slack:
                finalists += beam
                break
            # Update tracker variable
            gen_step += 1

        # Translate final beam predictions into sequences of word tokens
        sequence_tuples = list()
        for beam_item in finalists:
            beam_item_idx = [beam_item[1][0][step] for step in range(beam_item[1].shape[1])]
            if self.vocab.eos_id in beam_item_idx:
                beam_item_boundary = beam_item_idx.index(self.vocab.eos_id)
            else:
                beam_item_boundary = len(beam_item_idx)
            idx_to_words = [self.vocab.index_to_word[idx] for idx in beam_item_idx[:beam_item_boundary]]
            # Omit <GO> and <EOS> tags for a cleaner output
            sequence = ' '.join(idx_to_words[1: -1]) + '.'
            sequence_tuples.append((sequence, beam_item[0][0][0]))

        # Predictions are either printed or passed on to a downstream method/ function
        if print_results:
            print('Generation with beam search of size {:d} yielded following sentences:'.format(self.opt.beam_width))
            for stpl in sequence_tuples:
                print('{:s} | Probability={:.10f}'.format(stpl[0], stpl[1]))
            print('\n')
        else:
            return sequence_tuples

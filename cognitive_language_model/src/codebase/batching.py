""" Generates batches fed to the sequence autoencoder during training, validation, and inference steps.
Includes a procedure for bucketing, to increase the efficiency with which the data is processed by the
recurrent neural network. """

import random
import string
import numpy as np


def corpus_to_idx(input_data, vocab, mark_borders):
    """ Converts a corpus of natural language sentences into a single array of corresponding word indices. """

    def _is_num(w):
        """ Checks if the word should be replaced by <NUM>. """
        symbols = list(w)
        for s in symbols:
            if s in string.digits:
                return '<NUM>'
        return w

    # Replace numerical expressions with <NUM>
    all_words = [[_is_num(word) for word in s.split()] for s in input_data]
    # Index words while optionally filtering low-occurring words
    all_idx = [[vocab.word_to_index[word] if word in vocab.word_to_index.keys() else vocab.unk_id for word in lst]
               for lst in all_words]
    # Optionally mark the ending of each sentence with <EOS> (simultaneously marks the beginning of subsequent sentence)
    if mark_borders:
        all_idx = [lst + [vocab.eos_id] for lst in all_idx]
    # Flatten list and convert to numpy array
    all_idx = [idx for lst in all_idx for idx in lst]
    return np.array(all_idx, dtype=np.int32)


def transform_to_batches(input_data, vocab, mark_borders, batch_size):
    """ Transforms the input data into an array of word indices of batch-size compatible length. """
    # Convert corpus to 2D index tensor
    index_array = corpus_to_idx(input_data, vocab, mark_borders)
    # Determine the maximum number of batches
    num_batches = index_array.shape[0] // batch_size
    # Trim remainders
    index_array = index_array[0: num_batches * batch_size]
    # Reshape array into two dimensions
    index_array = index_array.reshape((batch_size, -1))
    return index_array


def index_sentence(input_sentence, vocab, opt):
    """ Indexes a single sentence from raw user or model input; called during inference only. """
    sentence_array = transform_to_batches([input_sentence], vocab, opt.mark_borders, 1)
    return sentence_array


def index_sentence_batch(input_batch, vocab, opt):
    """ Indexes a list of sentences from raw user or model input; called during inference only. """
    # Index sentences
    sentence_arrays = [index_sentence(sent, vocab, opt) for sent in input_batch]
    sentence_lengths = [array.shape[1] for array in sentence_arrays]
    # Apply padding up to the length of the longest sequence present within the batch
    max_len = max(sentence_lengths)
    padded_arrays = [np.concatenate(
        [array, np.expand_dims(np.array([vocab.pad_id] * (max_len - array.shape[1]), dtype=np.int32), 0)], 1)
                     if array.shape[1] < max_len else array for array in sentence_arrays]
    # Concatenate sentence arrays into a single batch array
    batch_array = np.concatenate(padded_arrays, 0)
    # Generate a binary length mask based on the sentence length list
    length_mask = np.concatenate([np.expand_dims(np.array([1.0] * l + [0.0] * (max_len - l), dtype=np.int32), 0)
                                  for l in sentence_lengths], 0)
    return batch_array, length_mask


class DataServer(object):
    """ Iterates through a specified data source, i.e. a list of buckets containing sentences of similar length,
    at training and inference time; produces batches of shape [batch_size, num_steps]. """

    def __init__(self, data, vocab, opt):
        self.data = data
        self.vocab = vocab
        self.opt = opt
        # Shuffle data
        if self.opt.shuffle:
            random.shuffle(self.data)
        self.pointer = 0
        # Transform corpus into a numpy array of appropriate size
        self.corpus_array = transform_to_batches(self.data, self.vocab, self.opt.mark_borders, self.opt.batch_size)
        self.corpus_len = self.corpus_array.shape[1]  # number of batches

    def __iter__(self):
        """ Returns an iterator object """
        return self

    def __next__(self):
        """ Returns the next batch in a format compatible with TensorFlow graphs. """
        # Extract a single batch from the corpus array at the current pointer position
        batch_len = min(self.opt.num_steps, self.corpus_len - self.pointer - 1)  # -1 is the label offset
        data_array = self.corpus_array[:, self.pointer: self.pointer + batch_len]
        label_array = self.corpus_array[:, self.pointer + 1: self.pointer + 1 + batch_len]
        # Update pointer position
        self.pointer += self.opt.num_steps
        # Terminate iteration once corpus has been exhausted
        if self.pointer >= self.corpus_len - 1:
            raise StopIteration
        return data_array, label_array

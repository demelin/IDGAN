""" Generates batches fed to the sequence autoencoder during training, validation, and inference steps.
Includes a procedure for bucketing, to increase the efficiency with which the data is processed by the
recurrent neural network. """

import random
import string
import numpy as np


def perform_bucketing(opt, labeled_pair_list):
    """ Groups the corpus content into the specified number of buckets of similar size;
    the desired number of buckets needs to be specified, whereas sentence lengths included in each bucket are inferred
    automatically. """
    # Obtain sentence lengths
    sentence_pair_lens = [(len(pair[0].split()), len(pair[1].split())) for pair in labeled_pair_list[0]]

    # Calculate average bucket depth (created buckets try to stay close to this optimum)
    # The length of a sentence pair is equal to the length of the longest of its elements
    buckets = [[0, 0] for _ in range(opt.num_buckets)]
    avg_bucket = len(labeled_pair_list[0]) // opt.num_buckets
    max_lens = [max(pair[0], pair[1]) for pair in sentence_pair_lens]
    len_counts = [(sent_len, max_lens.count(sent_len)) for sent_len in set(max_lens)]
    len_counts.sort(key=lambda x: x[0])

    bucket_pointer = 0
    len_pointer = 0

    # Determine bucket boundaries
    while bucket_pointer < opt.num_buckets and len_pointer < len(len_counts):
        target_bucket = buckets[bucket_pointer]
        # Set lower limit on the length of the current bucket's contents
        target_bucket[0] = len_counts[len_pointer][0]
        bucket_load = 0
        while True:
            try:
                len_count_pair = len_counts[len_pointer]
                deficit = avg_bucket - bucket_load
                surplus = (bucket_load + len_count_pair[1]) - avg_bucket
                if deficit >= surplus or bucket_pointer == opt.num_buckets - 1:
                    bucket_load += len_count_pair[1]
                    # Set upper limit
                    target_bucket[1] = len_count_pair[0]
                    len_pointer += 1
                else:
                    bucket_pointer += 1
                    break
            except IndexError:
                break

    # Populate buckets
    bucketed = [([], []) for _ in range(opt.num_buckets)]
    for k in range(len(labeled_pair_list[0])):
        pair_len = max(sentence_pair_lens[k][0], sentence_pair_lens[k][1])
        for l in range(len(buckets)):
            if buckets[l][0] <= pair_len <= buckets[l][1]:
                bucketed[l][0].append(labeled_pair_list[0][k])
                bucketed[l][1].append(labeled_pair_list[1][k])
    return buckets, bucketed


class DataServer(object):
    """ Iterates through a specified data source, i.e. a list of buckets containing sentences of similar length,
    at training and inference time; produces batches of shape [batch_size, num_steps]. """

    def __init__(self, data, vocab, opt):
        self.data = data
        self.vocab = vocab
        self.opt = opt
        self.pair_id = 0
        self.buckets = None
        # Bucket the input data
        if self.opt.num_buckets > 1:
            self.buckets, self.data = perform_bucketing(self.opt, self.data)
            self.bucket_id = 0

        # Convert data from sequences of word tokes to sequences of word indices
        if self.opt.num_buckets > 1:
            indexed_data = list()
            for bucket_tpl in self.data:
                indexed_bucket = ([(self.sent_to_idx(pair[0]), self.sent_to_idx(pair[1]))
                                  for pair in bucket_tpl[0]], bucket_tpl[1])
                indexed_data.append(indexed_bucket)
            self.data = indexed_data
        if self.opt.num_buckets == 0:
            self.data = (
                [(self.sent_to_idx(pair[0]), self.sent_to_idx(pair[1])) for pair in self.data[0]], self.data[1])

        # Shuffle data to negate possible ordering effects when training the model
        if self.opt.shuffle:
            if self.opt.num_buckets > 1:
                # Shuffle within buckets
                for i in range(len(self.data)):
                    zipped = list(zip(*self.data[i]))
                    random.shuffle(zipped)
                    self.data[i] = list(zip(*zipped))
                # Shuffle buckets, also
                bucket_all = list(zip(self.buckets, self.data))
                random.shuffle(bucket_all)
                self.buckets, self.data = zip(*bucket_all)
            else:
                zipped = list(zip(*self.data))
                random.shuffle(zipped)
                self.data = list(zip(*zipped))

    def sent_to_idx(self, sent):
        """ Transforms a sequence of words to the corresponding sequence of indices. """

        def _is_num(w):
            """ Checks if the word should be replaced by <NUM>. """
            symbols = list(w)
            for s in symbols:
                if s in string.digits:
                    return '<NUM>'
            return w

        # Replace numerical expressions with <NUM>
        sent_words = [_is_num(word) for word in sent.split()]
        # Index words replacing low-frequency tokens with <UNK>
        idx_list = [self.vocab.word_to_index[word] if word in self.vocab.word_to_index.keys() else self.vocab.unk_id for
                    word in sent_words]
        return idx_list

    def apply_padding(self, batch_list):
        """ Pads the index sequences within a batch list to the highest list-internal sequence length. """
        max_len = max([len(idx_seq) for idx_seq in batch_list])
        padded = [idx_seq + [self.vocab.pad_id] * (max_len - len(idx_seq)) for idx_seq in batch_list]
        return padded

    def __iter__(self):
        """ Returns an iterator object. """
        return self

    def bucketed_next(self):
        """ Samples the next batch from the current bucket. """
        # Initialize batch containers
        s1_batch = list()
        s2_batch = list()
        label_batch = list()
        if self.bucket_id < self.opt.num_buckets:
            # Fill individual batches by iterating over bucket contents
            while len(s1_batch) < self.opt.batch_size:
                try:
                    s1 = self.data[self.bucket_id][0][self.pair_id][0]
                    s2 = self.data[self.bucket_id][0][self.pair_id][1]
                    label = [float(self.data[self.bucket_id][1][self.pair_id])]
                    s1_batch.append(s1)
                    s2_batch.append(s2)
                    label_batch.append(label)
                    self.pair_id += 1
                except IndexError:
                    # Finish batch prematurely if current bucket has been exhausted, i.e. no mixed-bucket batches
                    self.pair_id = 0
                    self.bucket_id += 1
                    break
            # Check if bucket is empty, to prevent empty batches from being generated
            try:
                if self.pair_id == len(self.data[self.bucket_id][0]):
                    self.bucket_id += 1
            except IndexError:
                pass
        else:
            raise IndexError
        return s1_batch, s2_batch, label_batch

    def unbucketed_next(self):
        """ Samples the next batch from the un-bucketed corpus. """
        # Initialize batch containers
        s1_batch = list()
        s2_batch = list()
        label_batch = list()
        # Fill individual batches by iterating over the entire data source
        if self.pair_id < self.get_length():
            while len(s1_batch) < self.opt.batch_size:
                try:
                    s1 = self.data[0][self.pair_id][0]
                    s2 = self.data[0][self.pair_id][1]
                    label = [float(self.data[1][self.pair_id])]
                    s1_batch.append(s1)
                    s2_batch.append(s2)
                    label_batch.append(label)
                    self.pair_id += 1
                except IndexError:
                    break
        else:
            raise IndexError
        return s1_batch, s2_batch, label_batch

    def __next__(self):
        """ Returns the next batch from within the iterator source. """
        # Stop iteration once data source has been exhausted
        empty = True
        while empty:
            try:
                if self.opt.num_buckets > 1:
                    s1_batch, s2_batch, label_batch = self.bucketed_next()
                else:
                    s1_batch, s2_batch, label_batch = self.unbucketed_next()
                # was '>='
                if len(s1_batch) > 0:
                    empty = False
            except IndexError:
                raise StopIteration

        # Optionally apply padding
        if self.opt.pad:
            s1_batch = self.apply_padding(s1_batch)
            s2_batch = self.apply_padding(s2_batch)
            label_batch = self.apply_padding(label_batch)
        # Convert batch lists to numpy arrays
        s1_array = np.array(s1_batch, dtype=np.int32)
        s2_array = np.array(s2_batch, dtype=np.int32)
        label_array = np.array(label_batch, dtype=np.float32)
        return (s1_array, s2_array), label_array

    def get_length(self):
        """ Reports the lengths of the corpus in sentence pairs. """
        if self.opt.num_buckets > 1:
            return sum([len(bucket[0]) for bucket in self.data])
        else:
            return len(self.data[0])

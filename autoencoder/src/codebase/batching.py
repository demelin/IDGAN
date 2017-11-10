""" Generates batches fed to the sequence autoencoder during training, validation, and inference steps.
Includes a procedure for bucketing, to increase the efficiency with which the data is processed by the
recurrent neural network. """

import random
import string
import numpy as np


def perform_bucketing(opt, corpus_sentences):
    """ Groups the corpus content into the specified number of buckets of similar size;
    the desired number of buckets needs to be specified, whereas sentence lengths included in each bucket are inferred
    automatically. """
    # Obtain sentence lengths
    sentence_lens = [len(s.split(' ')) for s in corpus_sentences]

    # Calculate average bucket depth (created buckets try to stay close to this optimum)
    buckets = [[0, 0] for _ in range(opt.num_buckets)]
    avg_bucket = len(corpus_sentences) // opt.num_buckets
    len_counts = [(sl, sentence_lens.count(sl)) for sl in set(sentence_lens)]
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
    bucketed = [list() for _ in range(opt.num_buckets)]
    for k in range(len(corpus_sentences)):
        for l in range(opt.num_buckets):
            if buckets[l][0] <= sentence_lens[k] <= buckets[l][1]:
                bucketed[l].append(corpus_sentences[k])

    return buckets, bucketed


class DataServer(object):
    """ Iterates through a specified data source, i.e. a list of buckets containing sentences of similar length,
    at training and inference time; produces batches of shape [batch_size, num_steps]. """

    def __init__(self, data, vocab, opt):
        self.data = data
        assert type(self.data) == list, \
            'DataServer object expects the input data to be provided as a list of strings.'
        self.vocab = vocab
        self.opt = opt
        self.sent_id = 0
        self.buckets = None
        # Bucket the input data
        if self.opt.num_buckets > 1:
            self.buckets, self.data = perform_bucketing(self.opt, self.data)
            self.bucket_id = 0

        # Convert data from sequences of word tokes to sequences of word indices
        if self.opt.num_buckets > 1:
            self.data = [[self.sent_to_idx(sent) for sent in bucket] for bucket in self.data]
        else:
            self.data = [self.sent_to_idx(sent) for sent in self.data]

        # Shuffle data to negate possible ordering effects when training the model
        if self.opt.shuffle:
            if self.opt.num_buckets > 1:
                # Shuffle within buckets
                for i in range(len(self.data)):
                    random.shuffle(self.data[i])
                # Shuffle buckets, also
                bucket_all = list(zip(self.buckets, self.data))
                random.shuffle(bucket_all)
                self.buckets, self.data = zip(*bucket_all)
            else:
                random.shuffle(self.data)

    def sent_to_idx(self, sent):
        """ Transforms a sequence of words to the corresponding sequence of word indices. """

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
        # Optionally mark sentence boundaries (i.e. '<GO> w1, w2, ... <EOS>')
        if self.opt.mark_borders:
            idx_list = [self.vocab.go_id] + idx_list + [self.vocab.eos_id]
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
        label_batch = list()
        enc_input_batch = list()
        dec_input_batch = list()
        if self.bucket_id < self.opt.num_buckets:
            # Fill individual batches by iterating over bucket contents
            while len(enc_input_batch) < self.opt.batch_size:
                try:
                    indexed_sent = self.data[self.bucket_id][self.sent_id]
                    label_item = indexed_sent[1:]
                    enc_input_item = indexed_sent[1:]
                    # Reverse the input to the encoder, see arxiv.org/pdf/1703.03906.pdf
                    enc_input_item.reverse()
                    dec_input_item = indexed_sent[:-1]
                    label_batch.append(label_item)
                    enc_input_batch.append(enc_input_item)
                    dec_input_batch.append(dec_input_item)
                    self.sent_id += 1
                except IndexError:
                    # Finish batch prematurely if current bucket has been exhausted, i.e. no mixed-bucket batches
                    self.sent_id = 0
                    self.bucket_id += 1
                    break
            # Check if bucket is empty, to prevent empty batches from being generated
            try:
                if self.sent_id == len(self.data[self.bucket_id]):
                    self.bucket_id += 1
            except IndexError:
                pass
        else:
            raise IndexError
        return label_batch, enc_input_batch, dec_input_batch

    def unbucketed_next(self):
        """ Samples the next batch from the un-bucketed corpus. """
        # Initialize batch containers
        label_batch = list()
        enc_input_batch = list()
        dec_input_batch = list()
        # Fill individual batches by iterating over the entire data source
        if self.sent_id < self.get_length():
            while len(enc_input_batch) < self.opt.batch_size:
                try:
                    indexed_sent = self.data[self.sent_id]
                    label_item = indexed_sent[1:]
                    enc_input_item = indexed_sent[1:]
                    # Reverse the input to the encoder, see arxiv.org/pdf/1703.03906.pdf
                    enc_input_item.reverse()
                    dec_input_item = indexed_sent[:-1]
                    label_batch.append(label_item)
                    enc_input_batch.append(enc_input_item)
                    dec_input_batch.append(dec_input_item)
                    self.sent_id += 1
                except IndexError:
                    break
        else:
            raise IndexError
        return label_batch, enc_input_batch, dec_input_batch

    def __next__(self):
        """ Returns the next batch in a format compatible with TensorFlow graphs. """
        # Stop iteration once data source has been exhausted
        empty = False
        while not empty:
            try:
                if self.opt.num_buckets > 1:
                    label_batch, enc_input_batch, dec_input_batch = self.bucketed_next()
                else:
                    label_batch, enc_input_batch, dec_input_batch = self.unbucketed_next()
                if len(enc_input_batch) > 0:
                    empty = True
            except IndexError:
                raise StopIteration
        # Apply padding to the obtained batch
        if self.opt.pad:
            label_batch = self.apply_padding(label_batch)
            enc_input_batch = self.apply_padding(enc_input_batch)
            dec_input_batch = self.apply_padding(dec_input_batch)
        # Convert batch lists to numpy arrays
        label_array = np.array(label_batch, dtype=np.int32)
        enc_input_array = np.array(enc_input_batch, dtype=np.int32)
        dec_input_array = np.array(dec_input_batch, dtype=np.int32)
        return label_array, enc_input_array, dec_input_array

    def get_length(self):
        """ Reports the lengths of the corpus in sentences. """
        if self.opt.num_buckets > 1:
            return sum([len(bucket) for bucket in self.data])
        else:
            return len(self.data)


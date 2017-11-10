""" Defines the vocabulary object and extraction procedure used by all of IDGAN's subsystems.
Some of the code has been adopted from the pyTorch NLP tutorial and adjusted as needed. """

import re
import string
import pandas as pd


class Indexer(object):
    """ Translates words to their respective indices and vice versa;
        holds the vocabulary, frequency counts, and (optionally frequency-sorted) word indices of the target corpus. """

    def __init__(self, opt, name, zipf_sort=True):
        self.opt = opt
        self.name = name
        self.zipf_sort = zipf_sort

        self.word_to_freq = dict()
        self.word_to_count = dict()
        self.word_to_index = dict()
        self.index_to_word = dict()

        # n_words does not include the pruned low-frequency entries, includes special tags
        self.n_words = 0
        self.n_sentences = 0
        self.n_unk = 0
        self.n_num = 0

        # Surfaces special tag indices for quick retrieval
        self.go_id = None
        self.eos_id = None
        self.unk_id = None
        self.num_id = None
        self.pad_id = None

        self.observed_msl = None

    def add_sentence(self, sentence):
        """ Adds contents of a sentence to the object-internal index dictionary. """
        self.n_sentences += 1
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        """ Adds words to count dictionary. """

        def _is_num(w):
            """ Checks if the word should be replaced by <NUM>. """
            symbols = list(w)
            for s in symbols:
                if s in string.digits:
                    return '<NUM>'
            return w

        # Replace numerical expressions with <NUM>
        word = _is_num(word)
        # Add word to frequency dictionary
        if word not in self.word_to_freq:
            self.word_to_freq[word] = 1
        else:
            self.word_to_freq[word] += 1
        # Track the number of <UNK> tokens
        if self.word_to_freq[word] < self.opt.freq_bound:
            self.n_unk += 1
        # Once the frequency of any word type has been found to surpass the specified threshold,
        # subtract its previous occurrences from the <UNK> count
        elif self.word_to_freq[word] == self.opt.freq_bound:
            self.n_unk -= (self.opt.freq_bound - 1)

    def set_observed_msl(self, value):
        """ Setter method for the maximum sentence length encountered in the corpus. """
        self.observed_msl = value

    def filter_low_frequency(self, count_tuples):
        """ Filters out low frequency entries so as to reduce network parameter size
        (e.g. dimensionality of embedding tables). """
        count_tuples = [ctpl for ctpl in count_tuples if ctpl[1] >= self.opt.freq_bound]
        return count_tuples

    def map_ids(self):
        """ Assigns word indices to vocabulary entries, optionally sorting the vocabulary by word frequency
        (descending) first. """
        # Convert dictionary constructed during corpus read-in into a list of (word, word count) tuples
        count_tuples = list(self.word_to_freq.items())
        # Optionally filter out tuples containing low-frequency word types
        if self.opt.freq_bound > 0:
            count_tuples = self.filter_low_frequency(count_tuples)
        # Add special tokens to the list
        count_tuples += [('<GO>', self.n_sentences), ('<EOS>', self.n_sentences), ('<UNK>', self.n_unk), ('<PAD>', 1)]

        # Optionally sort by frequency, should word indices be distributed according to Zipf's law
        if self.zipf_sort:
            count_tuples = sorted(count_tuples, key=lambda x: x[1], reverse=True)

        # Generate dictionaries mapping vocabulary entries to corresponding indices and counts
        self.word_to_count = dict(count_tuples)
        self.word_to_index = {ctpl[0]: count_tuples.index(ctpl) for ctpl in count_tuples}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}

        # Update tag indices and final vocabulary size
        self.go_id = self.word_to_index['<GO>']
        self.eos_id = self.word_to_index['<EOS>']
        self.num_id = self.word_to_index['<NUM>']
        self.unk_id = self.word_to_index['<UNK>']
        self.pad_id = self.word_to_index['<PAD>']
        self.n_words += len(self.word_to_count)


def prepare_data(opt, corpus_location, corpus_name, zipf_sort=False, generate_vocab=False):
    """ Converts a corpus comprised of sentence similarity pairs into a list of tuples of the form
    (sent_a, sent_b, sim_score), used to train the content similarity estimator used in conjunction with
    the reinforcement learning extension of IDGAN. """
    # Read in the source corpus
    df_sim = pd.read_table(corpus_location, header=None, names=['sentence_A', 'sentence_B', 'relatedness_score'],
                           skip_blank_lines=True)
    # Initialize a list containing individual sentences pairs and corresponding similarity labels
    sim_data = [[], []]
    # Initialize a list containing raw sentences for the construction of the vocabulary object
    sim_sents = list()
    # Additionally, track sentence lengths
    sent_lens = list()
    for i in range(len(df_sim['relatedness_score'])):
        # Isolate the similarity pair within each dataframe row
        sent_a = df_sim.iloc[i, 0].strip()
        sent_b = df_sim.iloc[i, 1].strip()
        sent_list = [sent_a, sent_b]
        # Filter out punctuation
        exceptions = ''
        punctuation = string.punctuation + '–’‘'
        to_filter = ''.join([p if p not in exceptions else '' for p in punctuation])
        filter_regex = re.compile('[{:s}]'.format(re.escape(to_filter)))
        # Optionally lower word tokens
        if opt.lower:
            sent_list = [filter_regex.sub('', sent.lower()) for sent in sent_list]
        else:
            sent_list = [filter_regex.sub('', sent) for sent in sent_list]
        sent_a, sent_b = sent_list
        # Truncate similarity label
        label = '{:.4f}'.format(float(df_sim.iloc[i, 2]))
        # Extend collections
        sim_data[0].append((sent_a, sent_b))
        sim_data[1].append(label)
        sim_sents += [sent_a, sent_b]
        sent_lens += [len(sent_a.split()), len(sent_b.split())]
    out = sim_data

    # Optionally generate a vocabulary object
    if generate_vocab:
        observed_msl = max(sent_lens)
        vocab = Indexer(opt, corpus_name, zipf_sort=zipf_sort)
        vocab.set_observed_msl(observed_msl)
        print('Assembling index dictionary ...')
        for i in range(len(sim_sents)):
            vocab.add_sentence(sim_sents[i])
        vocab.map_ids()
        # Report rudimentary corpus statistics
        print('Registered {:d} unique words for the {:s} corpus.'.format(vocab.n_words, vocab.name))
        out = [vocab, sim_data]
    return out

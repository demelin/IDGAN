""" Defines the vocabulary object and extraction procedure used by all of IDGAN's subsystems.
Some of the code has been adopted from the pyTorch NLP tutorial and adjusted as needed. """

import re
import string
import codecs


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


def read_text_file(text_file, max_sent_len=None, sent_select='truncate', lower=False):
    """ Processes a raw text file into a list of sentences. """
    # Define punctuation filter (exceptions may be specified, if desired)
    # see: stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
    exceptions = ''
    punctuation = string.punctuation + '–’‘'
    to_filter = ''.join([p if p not in exceptions else '' for p in punctuation])
    filter_regex = re.compile('[{:s}]'.format(re.escape(to_filter)))
    # Read in specified text corpus line-by-line, so as to avoid running out of memory
    observed_msl = 0
    filtered = list()
    print('Reading in {:s} ...'.format(text_file))
    with codecs.open(text_file, 'r', encoding='utf8') as in_file:
        for line in in_file:
            line = line.strip()
            word_list = line.split()
            sent_len = len(word_list)
            # Optionally exclude ('drop') or truncate sentences above a predetermined length
            if max_sent_len:
                if sent_select == 'drop':
                    if sent_len > max_sent_len:
                        continue
                elif sent_select == 'truncate':
                    line = ' '.join(word_list[:max_sent_len])
                else:
                    raise ValueError('sent_select may equal either \'truncate\' or \'drop\'.')
            else:
                # If no max_sent_len is specified, get the maximum observed sentence length for subsequent padding
                if sent_len > observed_msl:
                    observed_msl = sent_len
            if lower:
                line = line.lower()
            # Apply punctuation filter
            line = filter_regex.sub('', line)
            filtered.append(line)
    print('Read in {:d} sentences.'.format(len(filtered)))
    return filtered, observed_msl


def prepare_data(opt, corpus_source, corpus_name, zipf_sort=False, generate_vocab=False):
    """ Reads in the specified corpus and returns the corresponding vocabulary object. """
    # Obtain the pre-processed list of corpus sentences
    corpus_sents, observed_msl = read_text_file(corpus_source, opt.max_sent_len, opt.sent_select, opt.lower)
    out = corpus_sents

    # Optionally create a vocabulary object
    if generate_vocab:
        corpus_vocab = Indexer(opt, corpus_name, zipf_sort=zipf_sort)
        corpus_vocab.set_observed_msl(observed_msl)
        print('Assembling dictionary ...')
        for i in range(len(corpus_sents)):
            corpus_vocab.add_sentence(corpus_sents[i])
        print('Mapping and sorting dictionary indices ... ')
        corpus_vocab.map_ids()
        # Report rudimentary corpus statistics
        print('Registered {:d} unique words for the {:s} corpus.'.format(corpus_vocab.n_words, corpus_vocab.name))
        out = [corpus_vocab, corpus_sents]
    return out

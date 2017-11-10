""" Defines a set of methods for preparing, extending, and merging human-annotated and synthetic data used in the
training of the sentence similarity classifier.
Main references:
[Mueller, 2016]: mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf,
[Zhang, 2015]: arxiv.org/pdf/1509.01626.pdf """

import os
import re
import random
import codecs
import string
import numpy as np
import pandas as pd

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

from pywsd.lesk import simple_lesk, cosine_lesk, adapted_lesk
import kenlm


class SickExtender(object):
    """ Generates synthetic data from corpora consisting of individual sentences, such as the SICK corpus, by replacing
    random words in each sentence with one of their synonyms as denoted by WordNet synsets; extensions are, as expected,
    reasonably noisy. """

    def __init__(self, sick_path, target_directory, lm_path=None, wsd_algorithm='cosine', sampling_parameter=0.5,
                 min_substitutions=2, num_candidates=5):
        # Declare corpus paths and parameters governing the extension process
        self.sick_path = sick_path
        self.target_directory = target_directory
        self.lm_path = lm_path
        self.wsd_algorithm = wsd_algorithm
        self.sampling_parameter = sampling_parameter
        self.min_substitutions = min_substitutions
        self.num_candidates = num_candidates
        self.filtered_path = os.path.join(self.target_directory, 'filtered_sick.txt')
        self.noscore_path = os.path.join(self.target_directory, 'noscore_sick.txt')
        # Clean the SICK corpus before the commencement of the extension procedure
        # Cleaned corpus is also used to train an n-gram language model used in selecting best corpus extensions from
        # a candidate pool
        if not os.path.exists(self.filtered_path) or not os.path.exists(self.noscore_path):
            self.filter_sick()
        if self.lm_path is None:
            raise ValueError('No language model provided! Use the noscore_sick corpus to train a .klm LM, first.')
        else:
            self.language_model = kenlm.LanguageModel(self.lm_path)

    def create_extension(self):
        """ Replaces randomly selected words within each line of the cleaned SICK corpus with their WordNet synonyms;
        replacement operations are limited to nouns, verbs, adjectives, and adverbs, i.e. the set of POS tags used
        within the WordNet ontology."""
        # Track extension progress in corpus lines already processed
        counter = 0
        # Designate path pointing to the extended corpus
        target_path = os.path.join(self.target_directory, 'extended_sick.txt')
        # Generate naive paraphrases via synonym replacement
        print('Commencing with the creation of the synthetic SICK examples.')
        with open(self.filtered_path, 'r') as rf:
            with open(target_path, 'w') as wf:
                for line in rf:
                    # Obtain line-specific word tokens and POS tags;
                    # sentences == [sent1, sent2]
                    sentences, sim_score = self.line_prep(line)
                    new_line = list()
                    for sentence in sentences:
                        # Store original sentence content for later use in paraphrase construction
                        tokens = sentence[1]
                        # Identify the WordNet synset most likely to match each word token within the sentence
                        disambiguation = self.disambiguate_synset(sentence)
                        # Replace a random number of sentence words with a randomly selected synonym drawn from its
                        # assigned synset
                        candidate_list = self.replace_with_synonyms(disambiguation)
                        if candidate_list is None:
                            continue
                        paraphrase = self.pick_candidate(tokens, candidate_list)
                        new_line.append(paraphrase)
                    # If nothing could be replaced in either sentence, do not include pair in the extension set
                    if len(new_line) < 2:
                        continue
                    # Concatenate the source corpus and extended corpus
                    wf.write(line)
                    wf.write(new_line[0] + '\t' + new_line[1] + '\t' + sim_score)
                    # Report on the progress periodically
                    counter += 1
                    if counter % 50 == 0 and counter != 0:
                        print('Current progress: Line {:d}.'.format(counter))

        # Report completion of the extension procedure
        print('The extension sentences for the SICK corpus has been successfully generated.\n'
              'It can be found under {:s}.\n'
              'Total amount of new sentence pairs: {:d}.'.format(target_path, counter))

    def filter_sick(self):
        """ Convert the SICK corpus into a cleaner format, where each line contains the two compared sentences
        and their relatedness score. """
        # Convert the corpus into a pandas dataframe, then isolate sentences and similarity score
        df_origin = pd.read_table(self.sick_path)
        df_classify = df_origin.loc[:, ['sentence_A', 'sentence_B', 'relatedness_score']]
        df_noscore = df_origin.loc[:, ['sentence_A', 'sentence_B']]
        df_noscore = df_noscore.stack()

        # Write the filtered corpus to file
        df_classify.to_csv(self.filtered_path, sep='\t', index=False, header=False)
        print('Filtered corpus saved to {:s}.'.format(self.filtered_path))

        # Write a score-less corpus variant to file, used to train the KenLM language model
        df_noscore.to_csv(self.noscore_path, index=False, header=False)
        print('Score-free corpus saved to {:s}.'.format(self.noscore_path))

    def line_prep(self, line):
        """ Tokenizes and tags a single line from the SICK corpus; prerequisite for synonym look-up in WordNet. """
        # Split line into its components (sentences and score)
        s1, s2, sim_score = line.split('\t')
        # Tokenize sentences
        s1_tokens = word_tokenize(s1)
        s2_tokens = word_tokenize(s2)
        # Assign part of speech tags to word tokens
        s1_penn_pos = pos_tag(s1_tokens)
        s2_penn_pos = pos_tag(s2_tokens)
        # Translate NLTK POS tags into corresponding WordNet POS tags;
        # also memorize word index within source sentences for latter use in paraphrase generation
        s1_wn_pos = list()
        s2_wn_pos = list()
        # Each tuple has the following format: (word token, WordNet POS tag, word token position in source sentence)
        for idx, item in enumerate(s1_penn_pos):
            if self.get_wordnet_pos(item[1]) != 'OTHER':
                s1_wn_pos.append((item[0], self.get_wordnet_pos(item[1]), s1_penn_pos.index(item)))
        for idx, item in enumerate(s2_penn_pos):
            if self.get_wordnet_pos(item[1]) != 'OTHER':
                s2_wn_pos.append((item[0], self.get_wordnet_pos(item[1]), s2_penn_pos.index(item)))

        # Return sentence parses obtained through the above procedure, plus original tokens and similarity score
        return [(s1_wn_pos, s1_tokens), (s2_wn_pos, s2_tokens)], sim_score

    def disambiguate_synset(self, sentence_plus_lemmas):
        """ Picks the most likely synset for a word given its sentence context. Utilizes the 'Cosine Lesk' word sense
        disambiguation algorithm, as implemented as part of the pywsd library. """
        # Select the word sense disambiguation algorithm; 'cosine Lesk' seems to work best
        if self.wsd_algorithm == 'simple':
            wsd_function = simple_lesk
        elif self.wsd_algorithm == 'cosine':
            wsd_function = cosine_lesk
        elif self.wsd_algorithm == 'adapted':
            wsd_function = adapted_lesk
        else:
            raise ValueError('Please specify the word sense disambiguation algorithm:\n '
                             '\'simple\' for \'Simple Lesk\'\n'
                             '\'cosine\' for \'Cosine Lesk\'\n'
                             '\'adapted\' for \'Adapted/Extended Lesk\'')
        lemmas, context = sentence_plus_lemmas
        context = ' '.join(context)
        disambiguated = list()
        # Disambiguate the meaning of individual word tokens comprising the input sentence
        for lemma in lemmas:
            try:
                selection = wsd_function(context, lemma[0], pos=lemma[1])
            # Catch exception thrown by 'simple Lesk' algorithm, in case lemma does not match any synsets
            except IndexError:
                selection = None
            disambiguated.append((lemma[0], selection, lemma[2]))
        return disambiguated

    def replace_with_synonyms(self, disambiguated_lemmas):
        """ Samples candidate words for replacement following a geometric distribution and performs the replacement
        operation. """
        all_synonyms = list()
        # Collect WordNet synonyms for each lemma within the sentence-specific list
        for idx, lemma in enumerate(disambiguated_lemmas):
            if lemma[1] is not None:
                if len(lemma[1].lemma_names()) > 1:
                    # Mild pre-processing
                    synonyms_per_word = ([' '.join(s.split('_')) for s in lemma[1].lemma_names()], idx)
                    all_synonyms.append(synonyms_per_word)

        # If the sentence cannot be modified, skip it, thus excluding it from the extension set
        if len(all_synonyms) == 0:
            return None

        # Model a geometric distribution, following the sampling strategy described in Zhang, 2015
        lower_bound = max(min(self.min_substitutions, len(all_synonyms)), 1)
        choices = np.array([i for i in range(lower_bound, len(all_synonyms) + 1)], dtype=np.float32)
        weights = np.array([self.sampling_parameter ** j for j in choices], dtype=np.float32)
        prob_dist = np.array([k / sum(weights) for k in weights], dtype=np.float32)
        # Sample word tokens to be replaced by synonyms
        outputs = list()
        no_subs = [(l[0], l[2]) for l in disambiguated_lemmas]
        replacement_picks = np.random.choice(choices, size=self.num_candidates, replace=True, p=prob_dist)
        for l in range(self.num_candidates):
            syn_list = all_synonyms[:]
            candidate = no_subs[:]
            num_replacements = int(replacement_picks[l])
            # Perform replacement
            for _ in range(num_replacements):
                # Randomly pick the word to be replaced
                m = np.random.randint(0, len(syn_list))
                # Randomly pick the synonym to replace the word with
                n = np.random.randint(0, len(syn_list[m][0]))
                candidate[syn_list[m][1]] = (syn_list[m][0][n], disambiguated_lemmas[syn_list[m][1]][2])
                # Remove the synonym set the replacement was drawn from
                del (syn_list[m])
            # Collect paraphrase candidates
            outputs.append(candidate)
        return outputs

    def pick_candidate(self, tokens, candidate_list):
        """ Picks the most probable paraphrase candidate according to a trained language model. """
        best_paraphrase = None
        best_nll = 0
        # Construct and rate paraphrases
        for candidate in candidate_list:
            for replacement in candidate:
                tokens[replacement[1]] = replacement[0]
            paraphrase = ' '.join(tokens)
            score = self.language_model.score(paraphrase)
            # Store the paraphrase which has been assigned highest likelihood by the language model;
            # high likelihood should hopefully translate into better output quality (w.r.t. well-formedness)
            if abs(score) > best_nll:
                best_nll = score
                best_paraphrase = paraphrase
        # Return best scored paraphrase
        return best_paraphrase

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """ Converts Penn Tree Bank POS tags into corresponding WordNet POS tags for synonym look-up.
        See stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python. """
        if treebank_tag.startswith('J') or treebank_tag.startswith('A'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return 'OTHER'


def shuffle_fuse(semeval_path, extended_sick_path, target_path):
    """ Combines the SemEval and the extended SICK corpus into single file;
    converts the SICK similarity score to lie within the [0: 1] range. """
    # Read in and optionally pre-process the contents of both corpora (files are relatively small)
    with open(semeval_path, 'r') as semeval_file:
        semeval_lines = semeval_file.readlines()
        semeval_lines = [line.lower().strip() for line in semeval_lines]
    with open(extended_sick_path, 'r') as sick_file:
        sick_lines = sick_file.readlines()
    # Convert SICK scores
    clean_lines = list()
    for line in sick_lines:
        line = line.lower().strip()
        line_list = line.split('\t')
        line_list[-1] = '{:.4f}'.format(float(line_list[-1]))
        clean_lines.append('\t'.join(line_list))
    # Merge and shuffle the adjusted corpus contents
    fused_lines = semeval_lines + clean_lines
    random.shuffle(fused_lines)
    # Write result to file
    with open(target_path, 'w') as out_file:
        for line in fused_lines:
            out_file.write(line + '\n')


def generate_fake_corpus(data_dir, similarity_dir, source_name, target_name, shuffle_mixtures=False, mode='corrupt'):
    """ Samples the 90k Europarl corpus to generate artificial sentence similarity pairs;
    ~5% identity pairs (100% similarity), ~5% crossed pairs (0% similarity), rest random mixtures. """
    # Declare locations of the source corpus and the created fake similarity corpus
    source_path = os.path.join(data_dir, '{:s}.txt'.format(source_name))
    target_path = os.path.join(similarity_dir, '{:s}.txt'.format(target_name))
    buffer = list()
    line_count = 0

    def _process_string(sentence):
        """ Strips and lowers the input sentence, removing all punctuation (unless exceptions are specified). """
        exceptions = ''
        punctuation = string.punctuation + '–’‘'
        to_filter = ''.join([p if p not in exceptions else '' for p in punctuation])
        filter_regex = re.compile('[{:s}]'.format(re.escape(to_filter)))
        return filter_regex.sub('', sentence.strip().lower())

    def _splice_buffer(in_buffer):
        """ Generates sentence similarity pairs from a buffered list of sentences by splicing together
        sentence fragments; buffering is used to circumvent possible memory limitations. """
        # Pad buffer, if current number of buffer contents is uneven
        if len(in_buffer) % 2 != 0:
            in_buffer.append(random.choice(in_buffer[:-1]))
        out_buffer = list()
        # Remove periods to make output more uniform
        buffered = [s[:-1] if s[-1] == '.' else s for s in in_buffer]
        # Create randomized sentence mixtures
        for j in range(0, len(in_buffer), 2):
            # Pre-process sampled sentence pair
            source_pair = [buffered[j].strip().split(), buffered[j + 1].strip().split()]
            pair_lengths = [len(sent) for sent in source_pair]
            # Decide whether to create mixtures
            mixture_flip = np.random.uniform(0.0, 1.0)
            if mixture_flip < 0.1:
                # Decide whether to generate identity or crossed pairs
                identity_flip = np.random.uniform(0.0, 1.0)
                if identity_flip < 0.5:
                    # Identity pair (pair contains two copies of the same sentence)
                    for k in range(2):
                        out_buffer.append('{:s}\t{:s}\t{:.4f}\n'
                                          .format(' '.join(source_pair[k]), ' '.join(source_pair[k]), 1.0))
                else:
                    # Crossed pair (pair contains two independently sampled, possibly similar, sentences)
                    for k in range(2):
                        out_buffer.append('{:s}\t{:s}\t{:.4f}\n'
                                          .format(' '.join(source_pair[k]), ' '.join(source_pair[abs(k - 1)]), 0.0))
            else:
                # Generate mixtures via the 'splicing' strategy
                # Decide on how to split the sampled sentences to obtain the mixture components
                random_fractions = [np.random.uniform(0.0, 1.0) for _ in range(2)]
                kept_lengths = [int(random_fractions[k] * pair_lengths[k]) for k in range(2)]
                kept_fractions = [kept_lengths[k] / pair_lengths[k] for k in range(2)]
                for k in range(2):
                    # Determine the length of the sentence fragments to be mixed and the length of the
                    # resulting mixture string
                    prefix_bound = kept_lengths[k]
                    suffix_bound = pair_lengths[abs(k-1)] - kept_lengths[abs(k-1)]
                    mixture_len = prefix_bound + suffix_bound
                    # Combine sentence fragments into new sentence strings
                    prefix_k = source_pair[k][: prefix_bound]
                    suffix_k = source_pair[abs(k-1)][-suffix_bound:]
                    mixture_list = prefix_k + suffix_k
                    if shuffle_mixtures:
                        random.shuffle(mixture_list)
                    mixture = ' '.join(mixture_list)
                    # prefix_coverage: Fraction of the 'donor' sentence found in the mixture string
                    # prefix_overlap: Fraction of the mixture string corresponding to the 'donor' sentence
                    # Mixture score is calculated as the mean of the two overlap measures to capture the
                    # bi-directionality of the similarity relationship
                    prefix_coverage = kept_fractions[k]
                    prefix_overlap = prefix_bound / mixture_len
                    mixture_weight = (prefix_coverage + prefix_overlap) / 2
                    # Fill buffer with newly generated, artificial similarity pairs
                    out_buffer.append('{:s}\t{:s}\t{:.4f}\n'.format(' '.join(source_pair[k]), mixture, mixture_weight))
        return out_buffer

    def _corrupt_buffer(in_buffer):
        """ Generates sentence similarity pairs from a buffered list of sentences by randomly corrupting sentence pairs
        via a cross-over operation; buffering is used to circumvent possible memory limitations. """
        # Pad buffer, if current number of buffer contents is uneven
        if len(in_buffer) % 2 != 0:
            in_buffer.append(random.choice(in_buffer[:-1]))
        out_buffer = list()
        # Remove periods to make output more uniform
        buffered = [s[:-1] if s[-1] == '.' else s for s in in_buffer]
        # Create randomized sentence mixtures
        for j in range(0, len(in_buffer), 2):
            # Pre-process sampled sentence pair
            source_pair = [buffered[j].strip().split(), buffered[j + 1].strip().split()]
            pair_lengths = [len(sent) for sent in source_pair]
            # Decide whether to create mixtures
            mixture_flip = np.random.uniform(0.0, 1.0)
            if mixture_flip < 0.1:
                # Decide whether to generate identity or crossed pairs
                identity_flip = np.random.uniform(0.0, 1.0)
                if identity_flip < 0.5:
                    # Identity pair (pair contains two copies of the same sentence)
                    for k in range(2):
                        out_buffer.append('{:s}\t{:s}\t{:.4f}\n'
                                          .format(' '.join(source_pair[k]), ' '.join(source_pair[k]), 1.0))
                else:
                    # Crossed pair (pair contains two independently sampled, possibly similar, sentences)
                    for k in range(2):
                        out_buffer.append('{:s}\t{:s}\t{:.4f}\n'
                                          .format(' '.join(source_pair[k]), ' '.join(source_pair[abs(k - 1)]), 0.0))
            else:
                # Generate mixtures via the 'corruption' strategy
                # Decide on how many replacements to perform to obtain sentence mixtures
                random_fractions = [np.random.uniform(0.0, 1.0) for _ in range(2)]
                corrupted_counts = [int(random_fractions[k] * pair_lengths[k]) for k in range(2)]
                corrupted_fractions = [corrupted_counts[k] / pair_lengths[k] for k in range(2)]
                for k in range(2):
                    # Randomly pick word indices within each sentence to perform replacements at
                    source_range = [_ for _ in range(pair_lengths[k])]
                    donor_range = [_ for _ in range(pair_lengths[abs(k-1)])]
                    corrupted_idx = np.random.choice(source_range, corrupted_counts[k], replace=False).tolist()
                    corrupted_source = list()
                    for idx in source_range:
                        # Try to replace words in the 'source' sentence with word from the 'donor' sentence located
                        # at the same position (as denoted by the index)
                        if idx in corrupted_idx:
                            try:
                                corrupted_source.append(source_pair[abs(k - 1)][idx])
                            except IndexError:
                                # Otherwise, pick a random replacement from the 'donor' sentence
                                corrupted_source.append(source_pair[abs(k - 1)][np.random.choice(donor_range, 1)[0]])
                        else:
                            # At positions where no replacement is performed, copy words from the 'source'
                            corrupted_source.append(source_pair[k][idx])
                    if shuffle_mixtures:
                        # Optionally shuffle mixtures (more randomness, even less coherent mixtures)
                        random.shuffle(corrupted_source)
                    mixture = ' '.join(corrupted_source)
                    # Similarity score is equal to the preserved fraction of the 'source' sentence,
                    # i.e. 1.0 if no corruption took place, 0.5 if half of sentence positions have been corrupted
                    mixture_weight = 1.0 - corrupted_fractions[k]
                    # Fill buffer with newly generated, artificial similarity pairs
                    out_buffer.append('{:s}\t{:s}\t{:.4f}\n'.format(' '.join(source_pair[k]), mixture, mixture_weight))
        return out_buffer

    # Choose synthesis function for the construction of the fake similarity corpus
    if mode == 'corrupt':
        syn_func = _corrupt_buffer
    else:
        syn_func = _splice_buffer
    print('Creating fake similarity pairs from {:s} ...'.format(target_name))
    with codecs.open(source_path, 'r', encoding='utf8') as in_file:
        # Read in Europarl sentences and sort by length, to obtain more accurate scores via synthesis operations
        in_lines = in_file.readlines()
        in_lines = sorted(in_lines, key=lambda x: len(x.split()), reverse=False)
        out_lines = list()
        with open(target_path, 'w') as out_file:
            for line_count, line in enumerate(in_lines):
                # Fill up buffer, create fake similarity pairs
                # i.e. create mixture strings and compare them against sentences from which they had been derived
                if len(buffer) == 10000:
                    print('Processed {:d} sentences'.format(line_count))
                    buffer = syn_func(buffer)
                    out_lines += buffer
                    # Empty buffer before continuing the loop
                    buffer = list()
                buffer.append(_process_string(line))
            # Flush the remainders still in buffer after exhausting the source corpus
            buffer = syn_func(buffer)
            for fake_pair in buffer:
                out_lines.append(fake_pair)
            # Write synthetic similarity pairs to file
            random.shuffle(out_lines)
            for out_line in out_lines:
                out_file.write(out_line)
    return line_count


def extend_true_corpora(similarity_dir, component_dir, filtered_name, extended_name, semeval_name, full_name,
                        train_name, valid_name, test_name, split_fractions):
    """ A wrapper function for extending and combining human-annotated data used in pre-training the sentence
    similarity classifier. """
    # Declare paths pointing to source corpora
    filtered_path = os.path.join(component_dir, '{:s}.txt'.format(filtered_name))
    extended_path = os.path.join(component_dir, '{:s}.txt'.format(extended_name))
    semeval_path = os.path.join(component_dir, '{:s}.txt'.format(semeval_name))
    full_path = os.path.join(component_dir, '{:s}.txt'.format(full_name))
    # Declare paths pointing to locations of the generated corpora extensions
    train_path = os.path.join(similarity_dir, '{:s}.txt'.format(train_name))
    valid_path = os.path.join(similarity_dir, '{:s}.txt'.format(valid_name))
    test_path = os.path.join(similarity_dir, '{:s}.txt'.format(test_name))
    # Initialize the extender object tasked with creating synthetic SICK similarity pairs
    extender = SickExtender(os.path.join(component_dir, 'sick.txt'), component_dir,
                            lm_path=os.path.join(component_dir, 'sick_lm.klm'))

    # If not done previously, filter and/ or extend the SICK corpus
    # and combine the result with the SemEval corpus
    if not os.path.exists(filtered_path):
        print('Filtering the SICK corpus ...')
        extender.filter_sick()
    if not os.path.exists(extended_path):
        print('Extending the SICK corpus ...')
        extender.create_extension()
    if not os.path.exists(full_path):
        print('Merging SICK and SemEval sentence similarity corpora ...')
        shuffle_fuse(semeval_path, extended_path, full_path)
    # Split the extended, combined, human-annotated similarity corpus into the training/ validation/ and test sets
    if not os.path.exists(train_path):
        print('Splitting the joint similarity corpus\n'
              'Training: {:.2f}% of source | Validation: {:.2f}% of source | Test: {:.2f}% of source'
              .format(split_fractions[0] * 100, split_fractions[1] * 100, split_fractions[2] * 100))
        with codecs.open(full_path, 'r', encoding='utf8') as out_file:
            source_lines = out_file.readlines()
            random.shuffle(source_lines)
            source_size = len(source_lines)
            with open(train_path, 'w') as train_file:
                with open(valid_path, 'w') as valid_file:
                    with open(test_path, 'w') as test_file:
                        for line_count, line in enumerate(source_lines):
                            if line_count < int(source_size * split_fractions[0]):
                                train_file.write(line)
                            elif line_count < int(source_size * (sum(split_fractions[:-1]))):
                                valid_file.write(line)
                            else:
                                test_file.write(line)

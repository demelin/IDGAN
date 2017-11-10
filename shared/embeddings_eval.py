""" A set of word pair analogy test designed in accordance with the strategy outlined in arxiv.org/pdf/1301.3781.pdf
for the 90k Europarl corpus, with the goal of intrinsically evaluating the quality of word embeddings learned by the
cognitive LM during the ID-variant corpus construction phase. """

import os
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.python import pywrap_tensorflow

from shared.util import load_pickle


class BaseOptions(object):
    """ Options class used in the evaluation of learned word embeddings; a truncated variant of the LM base options
    sans repository creation. """
    def __init__(self):
        root = os.path.join(os.getcwd(), '..')
        self.root_dir = root
        self.local_dir = os.path.join(root, 'cognitive_language_model/src')
        self.data_dir = os.path.join(self.root_dir, 'data')
        self.save_dir = os.path.join(self.local_dir, 'checkpoints')
        self.log_dir = os.path.join(self.local_dir, 'logs')
        self.out_dir = os.path.join(self.local_dir, 'out')


base_opt = BaseOptions()
# Specify the target path for the embedding dictionary constructed by make_embedding_dict()
embedding_dict_pkl = os.path.join(base_opt.root_dir, 'data/europarl/europarl_v7_train_embedding_dict.pkl')


def make_embedding_dict(opt, dict_pkl):
    """ Creates a dictionary with vocabulary items designated as keys and their embeddings as learned by an LM
    as values. """
    # Declare the source vocabulary path
    vocab_pkl = os.path.join(opt.root_dir, 'data/europarl/europarl_v7_train_vocab.pkl')
    # Declare the path to an LM checkpoint containing the learned embeddings
    embeddings_ckpt = os.path.join(opt.local_dir, 'checkpoints/best_cog_lm.ckpt')

    # Extract embeddings from the checkpoint file
    vocab = load_pickle(vocab_pkl)
    reader = pywrap_tensorflow.NewCheckpointReader(embeddings_ckpt)
    embedding_table = reader.get_tensor('embeddings/embedding_table')

    # Construct the embedding dictionary and pickle it for future access
    embedding_dict = {vocab.index_to_word[idx]: embedding_table[[idx], :] for idx in range(vocab.n_words)}
    with open(dict_pkl, 'wb') as in_file:
        pickle.dump(embedding_dict, in_file)
    print('Embedding dictionary created and pickled!')


def analogy_tests(opt, dict_pkl):
    """ Performs the semantic and syntactic analogy tests in accordance with arxiv.org/pdf/1301.3781.pdf
    on a selection of embeddings learned by the LM; as the tests had to be manually constructed, their scope is
    limited to questions referencing the ten most frequent word tokens and their paired counterparts
    per test question. """

    def _find_nearest(lookup_dict, source_pair, target_pair):
        """ Performs the analogy tests by iterating over the specified pairs. """
        # Input format
        # input pairs: ('france', 'paris'), ('germany', 'berlin')
        # corresponding variables: source[0], source[1], target[0], nearest

        # Keep track of the word vector closest to the predicted location of the analogy question answer and
        # the associated cosine distance score
        nearest = None
        max_similarity = 0.0

        # Identify the vector value denoting the relationship represented by the source pair
        relationship = lookup_dict[source_pair[1]] - lookup_dict[source_pair[0]]
        # Attempt to predict the second item in the target pair by applying the relationship vector to the first item
        predicted = relationship + lookup_dict[target_pair[0]]

        # Check which of the learned embeddings is closest to the predicted location within the embedding space
        for item in lookup_dict.items():
            similarity = cosine_similarity(predicted, item[1])[0][0]
            if similarity >= max_similarity:
                max_similarity = similarity
                nearest = item[0]
        return nearest

    # Load embedding dict
    embed_dict = load_pickle(dict_pkl)

    # Declare destination path for the test evaluation file
    out_path = os.path.join(opt.local_dir, 'out/embedding_tests.txt')

    # Manually define semantic tests
    capital = [('paris', 'france'), ('berlin', 'germany'), ('brussels', 'belgium'), ('vienna', 'austria'),
               ('copenhagen', 'denmark'), ('london', 'england'), ('athens', 'greece'),
               ('dublin', 'ireland'), ('amsterdam', 'netherlands'), ('lisbon', 'portugal')]
    currency = [('denmark', 'krone'), ('england', 'pound'), ('usa', 'dollar'), ('japan', 'yen'),
                ('germany', 'euro')]
    gender = [('mr', 'mrs'), ('sir', 'madam'), ('man', 'woman'), ('he', 'she'), ('king', 'queen'),
              ('father', 'mother'), ('boy', 'girl'), ('son', 'daughter')]

    # Manually define syntactic tests
    adverb = [('particular', 'particularly'), ('clear', 'clearly'), ('extreme', 'extremely'),
              ('final', 'finally'), ('absolute', 'absolutely'), ('simple', 'simply'), ('full', 'fully'),
              ('current', 'currently'), ('complete', 'completely'), ('quick', 'quickly')]
    opposite = [('possible', 'impossible'), ('necessary', 'unnecessary'), ('legal', 'illegal'),
                ('important', 'unimportant'), ('likely', 'unlikely'), ('clear', 'unclear'),
                ('realistic', 'unrealistic'), ('able', 'unable'), ('responsible', 'irresponsible')]
    comparative = [('great', 'greater'), ('long', 'longer'), ('early', 'earlier'), ('late', 'later'),
                   ('close', 'closer'), ('high', 'higher'), ('small', 'smaller'), ('few', 'fewer'),
                   ('large', 'larger'), ('broad', 'broader')]
    superlative = [('great', 'greatest'), ('long', 'longest'), ('early', 'earliest'), ('late', 'latest'),
                   ('close', 'closest'), ('high', 'highest'), ('small', 'smallest'), ('large', 'largest'),
                   ('broad', 'broadest'), ('poor', 'poorest')]
    participle = [('work', 'working'), ('make', 'making'), ('take', 'taking'), ('vote', 'voting'),
                  ('monitor', 'monitoring'), ('develop', 'developing'), ('read', 'reading'), ('say', 'saying'),
                  ('talk', 'talking'), ('sit', 'sitting')]
    nationality = [('france', 'french'), ('germany', 'german'), ('belgium', 'belgian'), ('austria', 'austrian'),
                   ('denmark', 'danish'), ('england', 'english'), ('greece', 'greek'), ('ireland', 'irish'),
                   ('netherlands', 'dutch'), ('portugal', 'portuguese')]
    past = [('working', 'worked'), ('making', 'made'), ('taking', 'took'), ('voting', 'voted'),
            ('monitoring', 'monitored'), ('developing', 'developed'), ('saying', 'said'),
            ('talking', 'talked'), ('sitting', 'sat'), ('wanting', 'wanted')]
    plurals = [('democracy', 'democracies'), ('nationality', 'nationalities'), ('president', 'presidents'),
               ('nation', 'nations'), ('country', 'countries'), ('committee', 'committees'), ('year', 'years'),
               ('citizen', 'citizens'), ('agenda', 'agendas'), ('month', 'months')]
    third = [('work', 'works'), ('make', 'makes'), ('take', 'takes'), ('vote', 'votes'),
             ('monitor', 'monitors'), ('say', 'says'), ('talk', 'talks'), ('want', 'wants'),
             ('cover', 'covers'), ('offer', 'offers')]

    # Iterate over all test items and write results to file
    all_tests = [capital, currency, gender, adverb, opposite, comparative, superlative, participle, nationality, past,
                 plurals, third]
    # Track fraction of correct predictions for intrinsic embedding quality estimation
    correct_predictions_total = list()
    questions_total = list()

    # Write output
    with open(out_path, 'w') as out_file:
        for test_set in all_tests:
            correct_set_predictions = 0
            for pair_a in test_set:
                source = pair_a
                correct_pair_predictions = -1
                for pair_b in test_set:
                    target = pair_b
                    prediction = _find_nearest(embed_dict, source, target)
                    if prediction == target[1]:
                        correct_pair_predictions += 1
                    log_entry = '{:s} is similar to {:s} as [{:s}] is similar to {:s}\n' \
                        .format(source[1], source[0], prediction, target[0])
                    out_file.write(log_entry)
                correct_set_predictions += correct_pair_predictions
                out_file.write('-' * 10 + '\n')

            # Compile output statistics per test set
            num_questions = len(test_set) ** 2 - len(test_set)
            questions_total.append(num_questions)
            correct_predictions_total.append(correct_set_predictions)
            out_file.write('\n')
            out_file.write('Number of non-identity questions asked {:d} | Correct answers: {:d} | Accuracy: {:.4f}\n'
                           .format(num_questions, correct_set_predictions, correct_set_predictions / num_questions))
            out_file.write('\n')
            out_file.write('=' * 10 + '\n')
            out_file.write('\n')
            print('Competed one of the test sets!')

        # Compile output statistics for the full collection of tests
        out_file.write('\n')
        out_file.write('Asked count: {}\n'.format(questions_total))
        out_file.write('Answered correctly count: {}\n'.format(correct_predictions_total))
        out_file.write('Semantic accuracy: {:.4f}\n'
                       .format(sum(correct_predictions_total[:3]) / sum(questions_total[:3])))
        out_file.write('Syntactic accuracy: {:.4f}\n'
                       .format(sum(correct_predictions_total[3:]) / sum(questions_total[3:])))
        out_file.write('Overall accuracy: {:.4f}\n'
                       .format(sum(correct_predictions_total) / sum(questions_total)))


# Call functions
if not os.path.exists(embedding_dict_pkl):
    make_embedding_dict(base_opt, embedding_dict_pkl)

analogy_tests(base_opt, embedding_dict_pkl)

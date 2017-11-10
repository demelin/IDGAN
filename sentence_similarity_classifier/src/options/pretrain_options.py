import os

from sentence_similarity_classifier.src.options.base_options import BaseOptions


class PreTrainOptions(BaseOptions):
    """ Default pre-training options for the sentence similarity classifier model.
    Also usable for quick evaluation on local machines. """

    def __init__(self):
        super(PreTrainOptions, self).__init__()
        # Data pre-processing
        self.save_dir = os.path.join(self.local_dir, 'checkpoints/pre_training')
        self.max_sent_len = None
        self.sent_select = 'truncate'
        # Word frequency filtering criterion is kept low due to small amount of training data
        self.freq_bound = 1
        self.shuffle = True
        self.lower = True
        self.pad = True
        self.num_buckets = 4

        # Model hyper-parameters
        self.static_keep_prob = 0.5
        self.rnn_keep_prob = 0.5

        # Loss and optimization hyper-parameters
        self.l2_beta = 1e-5
        self.grad_clip_norm = 6.0
        self.learning_rate = 0.0001

        # Session hyper-parameters
        self.is_train = True
        self.pre_train = True
        self.batch_size = 32
        self.num_epochs = 100
        self.report_freq = 200  # in steps
        self.summary_freq = 50  # in steps
        self.save_freq = None  # in epochs
        self.enable_early_stopping = True
        # 'Warm-up' phase
        self.start_early_stopping = 6  # in epochs
        self.annealing_step = 2  # in epochs
        self.annealing_factor = 0.9
        self.patience = 20  # in epochs

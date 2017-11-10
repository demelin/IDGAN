from sentence_similarity_classifier.src.options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    """ Default training options for the sentence similarity classifier model. """

    def __init__(self):
        super(TrainOptions, self).__init__()
        # Data pre-processing
        self.max_sent_len = None
        self.sent_select = 'truncate'
        self.freq_bound = 3
        self.shuffle = True
        self.lower = True
        self.pad = True
        self.num_buckets = 8

        # Model hyper-parameters
        self.static_keep_prob = 0.5
        self.rnn_keep_prob = 0.5

        # Loss and optimization hyper-parameters
        self.l2_beta = 1e-5
        self.grad_clip_norm = 12.0
        self.learning_rate = 0.0001

        # Session hyper-parameters
        self.is_train = True
        self.pre_train = False
        self.batch_size = 64
        # Training duration is shortened for fine-tuning / domain-adaptation
        self.num_epochs = 50
        self.report_freq = 200  # in steps
        self.summary_freq = 50  # in steps
        self.save_freq = None  # in epochs
        self.enable_early_stopping = True
        # 'Warm-up' phase
        self.start_early_stopping = 3  # in epochs
        self.annealing_step = 2  # in epochs
        self.annealing_factor = 0.9
        self.patience = 10  # in epoch

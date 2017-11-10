from cognitive_language_model.src.options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    """ Default training options for the cognitive language model. """

    def __init__(self):
        super(TrainOptions, self).__init__()
        # Data pre-processing
        self.max_sent_len = None
        self.sent_select = 'truncate'
        self.mark_borders = True
        self.freq_bound = 3
        self.shuffle = True
        self.lower = True
        self.pad = False

        # Model hyper-parameters
        self.static_keep_prob = 0.5
        self.rnn_keep_prob = 0.5

        # Loss and optimization hyper-parameters
        self.samples = 25
        self.sampled_values = None
        self.remove_accidental_hits = True
        self.l2_beta = 1e-5
        self.grad_clip_norm = 10.0
        # Higher starting rate encourages rapid initial exploration
        self.learning_rate = 0.001

        # Session hyper-parameters
        self.is_train = True
        # If is_local is set to True, training and testing is done on the toy corpus
        self.is_local = False
        self.batch_size = 32
        self.num_steps = 35
        self.num_epochs = 50
        self.report_freq = 500  # in steps
        self.summary_freq = 50  # in steps
        self.save_freq = None  # in epochs
        self.enable_early_stopping = True
        # 'Warm-up' phase
        self.start_early_stopping = 6  # in epochs
        self.annealing_step = 2  # in epochs
        self.annealing_factor = 0.9
        self.patience = 20  # in epochs

from cognitive_language_model.src.options.base_options import BaseOptions


class TrainOptionsSmall(BaseOptions):
    """ Reduced training parameters for quick model evaluation on local machines; for comprehensive comments
     refer to corresponding train_options. """

    def __init__(self):
        super(TrainOptionsSmall, self).__init__()
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
        self.grad_clip_norm = 8.0
        self.learning_rate = 0.001

        # Session hyper-parameters
        self.is_train = True
        self.is_local = True
        self.batch_size = 32
        self.num_steps = 35
        self.num_epochs = 100
        self.report_freq = 5
        self.summary_freq = 10
        self.save_freq = None
        self.enable_early_stopping = True
        self.start_early_stopping = 10
        self.annealing_step = 2
        self.annealing_factor = 0.9
        self.patience = 10

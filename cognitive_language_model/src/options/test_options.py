from cognitive_language_model.src.options.train_options import TrainOptions


class TestOptions(TrainOptions):
    """ Default testing options for the cognitive language model. """

    def __init__(self):
        super(TestOptions, self).__init__()
        # Data pre-processing
        self.max_gen_len = 60
        self.mark_borders = False
        self.shuffle = False
        self.pad = True

        # Session hyper-parameters
        self.is_train = False
        self.batch_size = 1
        self.num_samples = 10
        self.beam_width = 3
        self.er_lookahead = 4
        # onlinelibrary.wiley.com/doi/10.1111/tops.12025/full suggests er_width to be set to 40,
        # which has been found computationally too expensive for large-scale experiments
        self.er_width = 3
        self.cl_weight = 0.6

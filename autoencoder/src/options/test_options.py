from autoencoder.src.options.train_options import TrainOptions


class TestOptions(TrainOptions):
    """ Default testing options for the auto-encoder model. """

    def __init__(self):
        super(TestOptions, self).__init__()
        # Data pre-processing
        self.shuffle = False
        self.num_buckets = 1

        # Session hyper-parameters
        self.length_slack = 0
        self.is_train = False
        self.allow_dropout = False
        self.beam_width = 3
        self.num_samples = 10
        self.batch_size = 1

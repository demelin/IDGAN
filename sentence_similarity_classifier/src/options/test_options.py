from sentence_similarity_classifier.src.options.train_options import TrainOptions


class TestOptions(TrainOptions):
    """ Default testing options for the sentence similarity classifier model. """

    def __init__(self):
        super(TestOptions, self).__init__()
        # Session hyper-parameters
        self.is_train = False
        self.shuffle = False
        self.batch_size = 1
        self.num_buckets = 0

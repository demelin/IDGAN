""" Used to call the training / inference session functions defined for the sentence similarity classifier.
The uncommented line denotes the session functionality executed upon call. """

from sentence_similarity_classifier.src.codebase import session

# Pre-training pipeline:
# 1. Train SSC model from scratch on the extended human-annotated similarity corpus
# 2. Fine-tune on the synthetic Europarl similarity corpus
# 3. Evaluate performance on test set

if __name__ == '__main__':
    # Note: Specify pre-training in train options
    # session.train_session()
    # session.test_session()

    # Note: Specify fine-tuning in train options
    # session.train_session()
    session.test_session()

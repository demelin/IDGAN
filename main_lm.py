""" Used to call the training / inference session functions defined for the cognitive LM.
The uncommented line denotes the session functionality executed upon call. """

from cognitive_language_model.src.codebase import session

# Data preparation procedure:
# 1. Split full training corpus in train/ valid/ test sets
# 2. Train LM on the training set
# 3. Shrink full corpus according to model ppx scores obtained from the learned model
# 4. Split shrunk corpus in train/ valid/ test sets
# 5. Re-train model from scratch on the shrunk training set
# 6. Annotate shrunk corpus with ID scores from the re-learned model
# 7. Split shrunk corpus into low-ID and high-ID halves
# 8. Split each ID-variant corpus in train/ valid/ test sets used for SAE/ IDGAN training

if __name__ == '__main__':
    # session.train_session()
    # session.score_corpus()
    # session.shrink_scored_corpus()
    # session.shrunk_split()

    # Note: Manually rename data to match session script before executing next step
    # session.train_session()
    # session.annotate_corpus()
    # session.split_annotated_corpus()
    # session.low_split()
    # session.high_split()
    session.test_session()

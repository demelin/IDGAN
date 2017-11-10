import tensorflow as tf
from autoencoder.src.codebase.ae_function import SeqAE
from IDGAN.src.codebase.discriminator import EncodingsClassifier
from cognitive_language_model.src.codebase.lm_function import CogLM
from shared.util import padded_log


class IDGAN(object):
    """ The implementation of the Information Density Adversarial Generative Network (IDGAN), designed with the goal
    of reducing the information density (ID) of input sequences while preserving their content;
    in its present state, the system has been found unable to achieve this goal. """

    def __init__(self, opts, vocab, name):
        # Declare attributes
        self.opt, self.truth_opt, self.lm_opt, self.ssc_opt = opts
        self.vocab = vocab
        self.name = name
        self.int_type = tf.int32
        self.float_type = tf.float32

        # ID domain designations and corresponding class labels used throughout the code:
        # source == high-ID | class 0
        # target == low-ID | class 1
        self.gen = SeqAE(self.vocab, self.opt, 'translator_generator')  # high-ID -> low-ID
        self.truth = SeqAE(self.vocab, self.truth_opt, 'ground_truth', inference_only=True)  # low-ID -> low-ID

        # ID estimator
        self.lm = CogLM(self.vocab, self.lm_opt, 'generator_language_model')

        # Discriminator
        self.disc = EncodingsClassifier(self.opt, 'discriminator')

        with tf.name_scope(self.name):
            # Build computational sub-graphs
            # inputs | loss | optimization | prediction
            inputs, lrs, lambdas = self.inputs_subgraph()

            self.source_labels, self.source_enc_inputs, self.source_dec_inputs, \
                self.target_labels, self.target_enc_inputs, self.target_dec_inputs = inputs
            self.gen_lr, self.disc_lr = lrs
            self.adv_lambda, self.rec_lambda = lambdas

            self.adversarial_loss_fn = self.adversarial_loss_subgraph()
            self.information_density_fn = self.information_density_subgraph()
            self.gen_loss, self.partial_gen_losses, self.translation_id, self.input_id, self.id_reduction = \
                self.generator_loss_subgraph()
            self.disc_loss = self.discriminator_loss_subgraph()

            self.gen_grad_norm, self.gen_loss_reg, self.gen_train_op = self.gen_optimization_subgraph()
            self.disc_grad_norm, self.disc_loss_reg, self.disc_train_op = self.disc_optimization_subgraph()

            self.predicted_idx_eos, self.last_prediction = self.prediction_subgraph()

            self.gen_summaries, self.disc_summaries = self.summaries()

    def inputs_subgraph(self):
        """ Creates the placeholders used to feed data to the model. """
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            source_labels = tf.placeholder(shape=[None, None], dtype=self.int_type, name='source_labels')
            source_enc_inputs = tf.placeholder(shape=[None, None], dtype=self.int_type, name='source_enc_inputs')
            source_dec_inputs = tf.placeholder(shape=[None, None], dtype=self.int_type, name='source_dec_inputs')
            target_labels = tf.placeholder(shape=[None, None], dtype=self.int_type, name='target_labels')
            target_enc_inputs = tf.placeholder(shape=[None, None], dtype=self.int_type, name='target_enc_inputs')
            target_dec_inputs = tf.placeholder(shape=[None, None], dtype=self.int_type, name='target_dec_inputs')

            gen_lr = tf.placeholder(dtype=self.float_type, name='generator_learning_rate')
            disc_lr = tf.placeholder(dtype=self.float_type, name='discriminator_learning_rate')

            adv_lambda = tf.placeholder(dtype=self.float_type, name='adversarial_objective_lambda')
            rec_lambda = tf.placeholder(dtype=self.float_type, name='reconstruction_cost_lambda')
            return (source_labels, source_enc_inputs, source_dec_inputs, target_labels, target_enc_inputs,
                    target_dec_inputs), (gen_lr, disc_lr), (adv_lambda, rec_lambda)

    def adversarial_loss_subgraph(self):
        """ Defines the adversarial objective used within the IDGAN system. """
        with tf.name_scope('adv_loss'), tf.device('/gpu:0'):
            def _nll_loss(predictions, name):
                """ Helper function for calculating the negative log-likelihood GAN ('NLLGAN') loss, as suggested in
                the original GAN formulation; referred to as 'standard GAN' throughout the thesis manuscript. """
                return tf.reduce_mean(-padded_log(predictions), name=name)

            def _bce_loss(predictions, targets, name):
                """ Helper function for calculating the binary cross-entropy GAN ('BCEGAN') loss,
                as employed by DCGAN. """
                return -tf.reduce_mean(
                    tf.add(tf.multiply(targets, padded_log(predictions)),
                           tf.multiply(tf.subtract(1.0, targets),
                                       padded_log(tf.subtract(1.0, predictions)))), name=name)

            def _lse_loss(predictions, targets, name):
                """ Helper function for calculating the least squares GAN ('LSGAN') loss, in the manner utilized within
                LSGAN; Publication: arxiv.org/pdf/1611.04076.pdf. """
                return tf.reduce_mean(tf.square(tf.subtract(predictions, targets)), name=name)

            def _get_labels(input_data, is_true, smooth=True, flip=True):
                """ Generates the (smoothed, flipped) labels for the GAN criterion used in the training of the
                discriminator network. """
                labels_shape = [tf.shape(input_data)[0], 1]
                # Converts conditions to boolean tensors, so as to encapsulate label generation within the TF graph
                smooth = tf.convert_to_tensor(smooth)
                flip = tf.convert_to_tensor(flip)
                is_true = tf.convert_to_tensor(is_true)
                # Occasionally flip labels during discriminator training, see github.com/soumith/ganhacks
                flip_probability = tf.random_uniform([], 0, 1)
                is_true = tf.cond(tf.logical_and(flip, tf.greater(flip_probability, 0.1)), lambda: is_true,
                                  lambda: tf.logical_not(is_true))

                def _get_true():
                    """ Returns labels for samples from the target distribution; one-sided smoothing only. """
                    smooth_labels = tf.random_uniform(labels_shape, minval=0.7, maxval=1.2, dtype=self.float_type)
                    hard_labels = tf.ones(labels_shape, dtype=self.float_type)
                    return tf.cond(smooth, lambda: smooth_labels, lambda: hard_labels)

                def _get_fake():
                    """ Returns labels for samples from the source distribution. """
                    hard_labels = tf.zeros(labels_shape, dtype=self.float_type)
                    return hard_labels

                # Produced labels are (optionally) smoothed, as this has been found to stabilize GAN training
                return tf.cond(is_true, lambda: _get_true(), lambda: _get_fake())

            def adversarial_loss_fn(fake_encodings, true_encodings, discriminator, is_gen_step):
                """ Computes the adversarial loss for the current mini-batch pair. """
                # Generate class labels (fake/ true for disc training, inverted for gen training)
                fake_labels = _get_labels(fake_encodings, False, self.opt.smooth_labels, self.opt.flip_labels)
                true_labels = _get_labels(true_encodings, True, self.opt.smooth_labels, self.opt.flip_labels)
                gen_labels = _get_labels(fake_encodings, True, self.opt.smooth_labels, False)

                # Obtain class predictions for sentence encodings, as classified by disc
                initial_reuse = not is_gen_step  # gen loss is calculated first
                fake_predictions = discriminator(fake_encodings, reuse=initial_reuse)  # classify translator encodings
                true_predictions = discriminator(true_encodings, reuse=True)  # classify ground-truth encodings

                # Define loss names for more clarity in TensorBoard displays
                true_name = 'disc_loss_true'
                fake_name = 'disc_loss_fake'
                disc_name = 'discriminator_loss'
                gen_name = 'generator_loss'

                def _nllgan_loss(gen_step):
                    """ Calculates the adversarial NLLGAN loss. """
                    if gen_step:
                        # Generator loss
                        loss = _nll_loss(fake_predictions, gen_name)
                    else:
                        # Discriminator loss
                        loss = tf.add(_nll_loss(true_predictions, true_name),
                                      _nll_loss(1 - fake_predictions, fake_name), name=disc_name)
                    return loss

                def _wgan_loss(gen_step):
                    """ Calculates the adversarial WGAN loss using the Earth Mover's/ Wasserstein-1 distance;
                    Publication: arxiv.org/pdf/1701.07875.pdf and arxiv.org/pdf/1704.00028.pdf """
                    if gen_step:
                        loss = tf.reduce_mean(fake_predictions, name=gen_name)
                    else:
                        # Critic loss; In case compared mini-batches differ in size, truncate the longer batch to equal
                        # the length of the shorter
                        batch_least = tf.minimum(tf.shape(fake_predictions)[0], tf.shape(true_predictions)[0])
                        loss = tf.subtract(tf.reduce_mean(fake_predictions[: batch_least, :], name=fake_name),
                                           tf.reduce_mean(true_predictions[: batch_least, :], name=true_name),
                                           name=disc_name)
                    return loss

                def _wgangp_loss(gen_step):
                    """ Calculates the adversarial WGANGP loss (i.e. WGAN with gradient norm penalty)."""
                    if gen_step:
                        loss = -tf.reduce_mean(fake_predictions, name=gen_name)
                    else:
                        # Critic loss, same strategy as for standard WGAN
                        batch_least = tf.minimum(tf.shape(fake_predictions)[0], tf.shape(true_predictions)[0])
                        loss = tf.subtract(tf.reduce_mean(fake_predictions[: batch_least, :], name=fake_name),
                                           tf.reduce_mean(true_predictions[: batch_least, :], name=true_name),
                                           name=disc_name)
                        # Enforce gradient norm penalty
                        # Sample uniformly along straight lines drawn between points drawn from the source and target
                        # distributions; for EM distance estimation, gradients at each sampled point must have unit norm
                        # 1. Sample gradient checking positions
                        epsilon = tf.random_uniform([], 0.0, 1.0)
                        differences = tf.subtract(fake_encodings[: batch_least, :], true_encodings[: batch_least, :])
                        mix_encodings = tf.add(true_encodings[: batch_least, :], tf.multiply(epsilon, differences))
                        # 2. Calculate gradients at sampled positions
                        mix_gradients = tf.gradients(discriminator(mix_encodings, reuse=True), [mix_encodings])[0]
                        # 3. Penalize gradient norms exceeding 1.0, thereby enforcing the 1-Lipschitz contraint;
                        # Penalty term is minimized during the training of the critic, together with critic's loss
                        # Penalty term is made two-sided by taking the square of the norm divergence, and encourages
                        # the norm to go towards 1.0
                        gradient_penalty = tf.reduce_mean(tf.square(tf.subtract(mix_gradients, 1.0)))
                        gradient_penalty = tf.abs(gradient_penalty)
                        loss += tf.multiply(gradient_penalty, 10.0)

                    return loss

                def _bcegan_loss(gen_step):
                    """ Calculates the adversarial BCEGAN loss."""
                    if gen_step:
                        loss = _bce_loss(fake_predictions, gen_labels, gen_name)
                    else:
                        loss = tf.add(_bce_loss(true_predictions, true_labels, true_name),
                                      _bce_loss(fake_predictions, fake_labels, fake_name),
                                      name=disc_name)
                    return loss

                def _lsgan_loss(gen_step):
                    """ Calculates the adversarial LSGAN loss."""
                    if gen_step:
                        loss = _lse_loss(fake_predictions, gen_labels, gen_name)
                    else:
                        loss = tf.add(_lse_loss(true_predictions, true_labels, true_name),
                                      _lse_loss(fake_predictions, fake_labels, fake_name),
                                      name=disc_name)
                    return loss

                if self.opt.gan_type == 'NLLGAN':
                    res = _nllgan_loss(is_gen_step)
                elif self.opt.gan_type == 'WGAN':
                    res = _wgan_loss(is_gen_step)
                elif self.opt.gan_type == 'WGANGP':
                    res = _wgangp_loss(is_gen_step)
                elif self.opt.gan_type == 'BCEGAN':
                    res = _bcegan_loss(is_gen_step)
                elif self.opt.gan_type == 'LSGAN':
                    res = _lsgan_loss(is_gen_step)
                else:
                    # If the specified adversarial loss type is not implemented, default to NLLGAN
                    gan_types = ['NLLGAN', 'WGAN', 'WGANGP', 'BCEGAN', 'LSGAN']
                    print('Specified GAN variety for the computation of the adversarial loss is not supported\n'
                          'Available options: {}\nDefaulting to NLLGAN.'.format(gan_types))
                    res = _nllgan_loss(is_gen_step)
                return res

            return adversarial_loss_fn

    def information_density_subgraph(self):
        """ Calculates the information density, as estimated by surpisal scores, for the input sequences fed to the
        encoder net within the translator SAE and the output sequences generated by the corresponding decoder net;
        the so obtained surprisal scores have been found to be consistent with the non-TF surprisal calculation defined
        in the interface script used in the construction of ID-variant corpora. """
        with tf.name_scope('id_loss'), tf.device('/gpu:0'):
            def information_density_fn(inputs, targets, lm_reuse):
                """ Obtains the normalized, sequence-specific surprisal and/ or UID divergence scores
                from the ID estimator LM. """

                def _log2(value):
                    """ Log to the base of 2, required for surprisal calculation. """
                    numerator = tf.log(value)
                    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
                    return tf.div(numerator, denominator)

                # Obtain the predictive distributions for each time-step of the processed sequence from the LM
                length_mask, scores = self.lm(inputs, reuse=lm_reuse)

                # Mask predictions associated with <EOS> and <PAD> symbols from being included in surprisal computation
                mask = tf.sequence_mask(
                    tf.subtract(length_mask, tf.ones(tf.shape(length_mask), dtype=self.int_type)),
                    tf.shape(targets)[1], dtype=self.float_type)

                # Isolate the probability scores for word tokens comprising the evaluated sequences from the step-wise
                # probability distributions
                # Flatten sequence scores and adjust target labels accordingly
                flat_scores = tf.reshape(scores, [-1])
                flat_targets = tf.reshape(targets, [-1])
                flat_range = tf.range(
                    0, (tf.reduce_prod(tf.shape(flat_scores))), tf.shape(scores)[-1])
                range_targets = tf.add(flat_targets, flat_range)

                # dynamic_partition used a stf.gather() substitute to avoid costly gradient transforms
                indices = tf.reshape(range_targets, [-1, 1])
                updates = tf.ones(tf.shape(flat_targets), dtype=self.int_type)
                # Collect probabilities associated with individual words within the processed sentences
                num_partitions = 2
                partitions = tf.scatter_nd(indices, updates, tf.shape(flat_scores))
                partition_list = tf.dynamic_partition(flat_scores, partitions, num_partitions)
                flat_probabilities = partition_list[1]
                # Reshape back into a matrix of shape [batch_size, max_sentence_length]
                probabilities = tf.reshape(flat_probabilities, tf.shape(targets))

                # Calculate mean sentence surprisal score in accordance with the standard surprisal formulation
                surprisal = _log2(tf.div(1.0, probabilities))
                masked_surprisal = tf.multiply(surprisal, mask)
                target_lengths = tf.count_nonzero(masked_surprisal, axis=-1, keep_dims=True, dtype=self.float_type)
                target_id = tf.reduce_mean(
                    tf.divide(tf.reduce_sum(masked_surprisal, axis=-1, keep_dims=True), target_lengths))

                # Calculate absolute UID divergence -- unused, retained for possible application in future research
                # 1. UID posits that every word should ideally convey the same amount of information
                # 2. Information transmitted is estimated via word surprisal
                # 3. To enforce UID, reduce surprisal difference/ divergence between neighboring words
                # if self.opt.smooth_uid:
                #     zero_slice = tf.zeros([tf.shape(masked_surprisal)[0], 1], dtype=self.float_type)
                #     step_shifted = tf.concat([zero_slice, masked_surprisal[:, :-1]], axis=1)
                #     uid_div = tf.abs(tf.subtract(masked_surprisal, step_shifted))
                #     target_uid_div = tf.reduce_mean(
                #         tf.divide(tf.reduce_sum(uid_div, axis=-1, keep_dims=True), target_lengths))
                #     target_id = tf.add(target_id, target_uid_div) / 2

                return target_id

            return information_density_fn

    # Note: The generator loss is decoupled from the discriminator loss, as, depending on the employed adversarial
    # objective, the generator may require more updates than the discriminator and vice versa
    def generator_loss_subgraph(self):
        """ Defines the full generator loss as a weighted combination between the adversarial objective and the
        reconstruction objective. """
        with tf.name_scope('gen_loss'), tf.device('/gpu:0'):
            # Propagate input data through the translator SAE as well as the ground-truth SAE
            self.gen(self.source_enc_inputs, self.source_dec_inputs, label_idx=self.source_labels)
            self.truth(self.target_enc_inputs, self.target_dec_inputs)

            # Specify LM inputs and targets (by definition of LM-training, inputs should begin with the <EOS> tag)
            eos_slice = tf.fill([tf.shape(self.source_dec_inputs)[0], 1], self.vocab.eos_id)
            lm_input_truth = tf.concat([eos_slice, self.source_dec_inputs[:, 1:]], -1)
            lm_target_truth = self.source_labels
            # IDGAN input sequence forwarded to the LM for ID estimation
            lm_input_translation = tf.concat([eos_slice, self.gen.predicted_idx[:, : -1]], -1)
            # IDGAN output sequence, as produced the translator SAE's decoder, forwarded to the LM for ID estimation
            lm_target_translation = self.gen.predicted_idx_eos

            # Obtain sentence encodings to be used as inputs to the discriminator for the computation of the gen portion
            # of the adversarial objective; within IDGAN, sentence encodings are obtained by extracting the final hidden
            # states from the encoders of either SAE
            gen_encodings = self.gen.enc.h_state  # high-ID sentence encodings
            truth_encodings = self.truth.enc.h_state  # low-ID sentence encodings

            # Discriminator input is normalized to improve training stability; see github.com/soumith/ganhacks
            gen_encodings = tf.tanh(gen_encodings)
            truth_encodings = tf.tanh(truth_encodings)
            # Calculate the gen portion of the adversarial loss
            gen_adv = self.adversarial_loss_fn(gen_encodings, truth_encodings, self.disc, is_gen_step=True)

            # Obtain surprisal scores corresponding to IDGAN's input and output sequences (for validation only)
            input_id = self.information_density_fn(lm_input_truth, lm_target_truth, lm_reuse=False)
            translation_id = self.information_density_fn(lm_input_translation, lm_target_translation, lm_reuse=True)
            # Compute the ID differential (positive value denotes a reduction in ID within the output as compared
            # to the input, negative value indicates an increase)
            id_reduction = tf.subtract(input_id, translation_id)

            # Calculate the reconstruction loss as defined within the translator SAE graph
            gen_rec = self.gen.loss_avg

            # Combine partial gen objectives into a single scalar value denoting the full gen loss
            total_gen_loss = tf.add_n([tf.multiply(self.adv_lambda, gen_adv),
                                       tf.multiply(self.rec_lambda, gen_rec)], name='generator_loss')
            # Keep track of partial objectives for visualizations in TensorBoard / custom visualizations
            partial_gen_losses = [gen_adv, gen_rec]

        return total_gen_loss, partial_gen_losses, translation_id, input_id, id_reduction

    def discriminator_loss_subgraph(self):
        """ Defines the full discriminator loss, which is equal to the discriminator part of the full adversarial
        objective. """
        with tf.name_scope('disc_loss'), tf.device('/gpu:0'):
            # Propagate input data through the translator SAE as well as the ground-truth SAE
            self.gen(self.source_enc_inputs, encode_only=True, reuse=True)
            self.truth(self.target_enc_inputs, encode_only=True, reuse=True)

            # Obtain sentence encodings
            gen_encodings = self.gen.enc.h_state
            truth_encodings = self.truth.enc.h_state

            # Discriminator input is normalized
            gen_encodings = tf.tanh(gen_encodings)
            truth_encodings = tf.tanh(truth_encodings)
            # Calculate the disc portion of the adversarial loss
            disc_adv = self.adversarial_loss_fn(gen_encodings, truth_encodings, self.disc, is_gen_step=False)

        return disc_adv

    def gen_optimization_subgraph(self):
        """ Defines the optimization procedure for the generator. """
        with tf.variable_scope('gen_optimization'), tf.device('/gpu:0'):

            def _get_train_op(loss_variables, loss, optimizer, lr, name, clip_grads=False):
                """ Constructs a training op used in model optimization;
                generalizable to multiple generators used jointly in the GAN. """
                global_step = tf.get_variable(shape=[], name='{:s}_global_step'.format(name), dtype=self.int_type,
                                              initializer=tf.constant_initializer(0, dtype=self.int_type),
                                              trainable=False)
                # Apply l2 regularization
                loss_l2 = tf.add_n(
                    [tf.nn.l2_loss(var) for var in loss_variables if len(var.shape) > 1]) * self.opt.l2_beta

                loss_regularized = tf.add(loss, loss_l2, name='{:s}_loss_regularized'.format(name))

                grads = tf.gradients(loss_regularized, loss_variables)
                # Gradient norm is monitored, as low values (< 100), are indicative of stable GAN training
                grad_norm = tf.reduce_mean([tf.norm(grad) for grad in grads if grad is not None])

                if clip_grads:
                    # Optionally clip gradients to prevent the exploding gradients scenario within encoder LSTM-RNNs;
                    # unused in final experiments
                    grads, _ = tf.clip_by_global_norm(grads, self.opt.grad_clip_norm)

                optimizer = optimizer(lr, name='{:s}_optimizer'.format(name))
                train_op = optimizer.apply_gradients(zip(grads, loss_variables), global_step=global_step,
                                                     name='{:s}_train_op'.format(name))
                return grad_norm, loss_regularized, train_op

            # Restrict the set of optimized variables to gen parameters only
            t_vars = tf.trainable_variables()
            gen_vars = [var for var in t_vars if self.gen.name in var.name]
            if not self.opt.train_dec:
                # Optionally restrict the variable set further, to only train the encoder network of the translator SAE
                gen_vars = [var for var in gen_vars if 'encoder' in var.name]
            optimizer_function = tf.train.AdamOptimizer

            # Gen portion of the WGAN objective is optimized with RMSProp in the original publication
            if self.opt.gan_type == 'WGAN':
                optimizer_function = tf.train.RMSPropOptimizer

            # Define gen-specific training OPs to be called during graph execution / session
            gen_grad_norm, gen_loss_reg, gen_train_op = _get_train_op(
                gen_vars, self.gen_loss, optimizer_function, self.gen_lr, self.gen.name)

        return gen_grad_norm, gen_loss_reg, gen_train_op

    def disc_optimization_subgraph(self):
        """ Defines the optimization procedure for the discriminator. """
        with tf.variable_scope('disc_optimization'), tf.device('/gpu:0'):
            def _get_train_op(loss_variables, loss, optimizer, lr, name):
                """ Constructs a training op used in model optimization;
                generalizable to multiple discriminators, if desired. """
                global_step = tf.get_variable(shape=[], name='{:s}_global_step'.format(name), dtype=self.int_type,
                                              initializer=tf.constant_initializer(0, dtype=self.int_type),
                                              trainable=False)
                # Apply l2 regularization
                loss_l2 = tf.add_n(
                    [tf.nn.l2_loss(var) for var in loss_variables if len(var.shape) > 1]) * self.opt.l2_beta

                loss_regularized = tf.add(loss, loss_l2, name='{:s}_loss_regularized'.format(name))
                grads = tf.gradients(loss_regularized, loss_variables)

                # Gradient norm is monitored, as low values (> 100), are indicative of stable GAN training
                grad_norm = tf.reduce_mean([tf.norm(grad) for grad in grads if grad is not None])
                optimizer = optimizer(lr, name='{:s}_optimizer'.format(name))
                train_op = optimizer.apply_gradients(zip(grads, loss_variables), global_step=global_step,
                                                     name='{:s}_train_op'.format(name))
                return grad_norm, loss_regularized, train_op

            # Restrict the set of optimized variables to disc parameters only
            t_vars = tf.trainable_variables()
            disc_vars = [var for var in t_vars if self.disc.name in var.name]
            optimizer_function = tf.train.AdamOptimizer

            # 1. Critic portion of the WGAN objective is optimized with RMSProp in the original publication
            # Furthermore, 1-Lipschitz constraint is enforced by clipping critic's parameters to a fixed box
            # As a consequence, loss gradient magnitude will be upper-bounded by 1.0
            # 2. 'Critic' differs from disc in that, within the WGAN(-GP) setup, the classifier seeks to model a
            # function which captures the maximum distance between the source and target distributions, rather than
            # learning to classify the samples it receives in a discriminatory manner
            if self.opt.gan_type == 'WGAN':
                disc_vars = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in disc_vars]
                optimizer_function = tf.train.RMSPropOptimizer

            # Define disc-specific training OPs to be called during graph execution / session
            disc_grad_norm, disc_loss_reg, disc_train_op = _get_train_op(
                disc_vars, self.disc_loss, optimizer_function, self.disc_lr, self.disc.name)

        return disc_grad_norm, disc_loss_reg, disc_train_op

    def prediction_subgraph(self):
        """ Defines the generative process taking place during inference; generated sequences are extracted from
         translator SAE's decoder net. """
        with tf.variable_scope('prediction'), tf.device('/gpu:0'):
            return self.gen.predicted_idx_eos, self.gen.last_prediction

    def summaries(self):
        """ Defines and compiles the summaries tracking various model parameters and outputs. """
        with tf.name_scope('summaries'), tf.device('/cpu:0'):
            summary_names = ['gen_adversarial_loss', 'gen_reconstruction_loss', 'gen_translation_id',
                             'information_density_reduction', 'full_gen_loss', 'full_disc_loss']
            tracked_variables = self.partial_gen_losses + [self.translation_id, self.id_reduction, self.gen_loss,
                                                           self.disc_loss]

            summary_tpls = zip(tracked_variables, summary_names)
            summary_list = [tf.summary.scalar(name=tpl[1], tensor=tpl[0]) for tpl in summary_tpls]
            gen_summaries = tf.summary.merge(summary_list[:-1], name='gen_summaries')
            disc_summaries = tf.summary.merge([summary_list[-1]], name='disc_summaries')
        return gen_summaries, disc_summaries

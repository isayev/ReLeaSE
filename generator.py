import numpy as np
import random
from matplotlib import pyplot as plt

import lasagne
from lasagne.layers import DenseLayer, ConcatLayer, InputLayer, EmbeddingLayer
import agentnet
from agentnet.resolver import ProbabilisticResolver
from agentnet.memory import GRUMemoryLayer, StackAugmentation
from agentnet.agent import Recurrence


#theano imports
import lasagne
import theano
import theano.tensor as T
import sys
sys.setrecursionlimit(100000)
floatX = theano.config.floatX

class Generator(object):

    def __init__(self, G_inp, n_tokens, n_hid_1, stack_depth, stack_width, lr=0.01, seq_len=100):

        self.G_inp = G_inp
        self.n_tokens = n_tokens
        self.n_hid_1 = n_hid_1
        self.stack_depth = stack_depth
        self.stack_width = stack_width
        self.lr = lr
        self.seq_len = seq_len
        self.REPORT_RATE = 100

    def build_model(self):
        #it's size (theano-expression)
        batch_size = self.G_inp.shape[0]

        # input letter goes here
        output_shape = (None,)
        self.observation_layer = InputLayer(output_shape, name="obs_input")

        # embedding the input layer
        self.obs_embedding = EmbeddingLayer(self.observation_layer,
                                       input_size=self.n_tokens,
                                       output_size=self.n_tokens,
                                       name="input_embedding")

        # GRU n units
        # previous GRU state goes here
        self.prev_gru_layer = InputLayer((None, self.n_hid_1), name="prev_rnn_state")

        # previous stack goes here
        prev_stack_layer = InputLayer((None, self.stack_depth, self.stack_width))

        # Stack controls - push, pop and no-op
        self.stack_controls_layer = lasagne.layers.DenseLayer(self.prev_gru_layer, 3,
                                                         nonlinearity=lasagne.nonlinearities.softmax,
                                                         name="stack controls")
        # stack input
        stack_input_layer = lasagne.layers.DenseLayer(self.prev_gru_layer, self.stack_width,
                                                      nonlinearity=lasagne.nonlinearities.tanh,
                                                      name="stack input")
        # new stack state
        next_stack = StackAugmentation(stack_input_layer,
                                       prev_stack_layer,
                                       self.stack_controls_layer)

        # stack top (used for RNN)
        stack_top = lasagne.layers.SliceLayer(next_stack, 0, 1)
        
        # new GRU state
        self.gru = GRUMemoryLayer(self.n_hid_1,
                             ConcatLayer([self.obs_embedding, stack_top]),
                             self.prev_gru_layer)
        self.features = lasagne.layers.get_output(stack_top)

        ##Outputs

        # next letter probabilities

        probability_layer = lasagne.layers.DenseLayer(self.gru,
                                                      num_units=self.n_tokens,
                                                      nonlinearity=lasagne.nonlinearities.softmax,
                                                      name="policy_original")

        # resolver picks a particular letter in generation mode

        resolver = ProbabilisticResolver(probability_layer,
                                         assume_normalized=True,
                                         name="resolver")

        # verify that letter shape matches
        assert tuple(lasagne.layers.get_output_shape(resolver)) == tuple(output_shape)

        # define a dictionary that maps agent's "next memory" output to its previous memory input
        from collections import OrderedDict

        memory_dict = OrderedDict([
            (self.gru, self.prev_gru_layer),
            (next_stack, prev_stack_layer),
        ])

        # define an input layer that stores sequences
        sequences_input_layer = InputLayer((None, self.seq_len),
                                           input_var=self.G_inp,
                                           name="reference sequences"
                                           )

        # and another one that only pretends to, while actually uses the reference letters
        # we will use it for training.
        generator_passive = Recurrence(

            # we use out previously defined dictionary to update recurrent network parameters
            state_variables=memory_dict,

            # we feed in reference sequence into "prev letter" input layer, tick by tick along the axis=1.
            input_sequences={self.observation_layer: sequences_input_layer},

            # we track agent would-be actions and probabilities
            tracked_outputs=[resolver, probability_layer],

            n_steps=self.seq_len,

            # finally, we define an optional batch size param
            # (if omitted, it will be inferred from inputs or initial value providers if there are any)
            batch_size=batch_size,
        )

        # get lasagne layers for sequences
        # dict of RNN/stack sequences, (actions and probabilities output)
        self.recurrent_states, (wouldbe_letters_layer, self.probas_seq_layer,) = generator_passive.get_sequence_layers()
        self.probas_seq = lasagne.layers.get_output(self.probas_seq_layer)
        self.weights = lasagne.layers.get_all_params(self.probas_seq_layer, trainable=True, )
        # Recurrent state dictionary: maps state outputs to prev_state inputs
        feedback_dict = OrderedDict([
            (self.gru, self.prev_gru_layer),
            (next_stack, prev_stack_layer),
            (resolver, self.observation_layer)  # here we feed actions back as next observations
        ])

        # generation batch size
        self.gen_batch_size = T.scalar('generated batch size', 'int32')

        # define a lasagne recurrence that actually generates letters
        # it will be used to generate sample sequences.

        generator_active = Recurrence(
            state_variables=feedback_dict,
            tracked_outputs=[probability_layer],

            # note that now we do need to provide batch size since generator does not have any inputs
            batch_size=self.gen_batch_size,
            n_steps=self.seq_len
        )

        # get sequences as before

        action_seq_layer, gen_probabilities_layer = generator_active[resolver], generator_active[probability_layer]

        self.generated_action_seq = lasagne.layers.get_output(action_seq_layer)
        self.gen_probas_seq = lasagne.layers.get_output(gen_probabilities_layer)
        
        # take all predicitons but for last one (we don't have it's 'next')
        predicted_probas = self.probas_seq[:, :-1]

        # crop probabilities to avoid -Inf logarithms
        self.predicted_probas = T.maximum(predicted_probas, 1e-10)

        # correct answers
        references = self.G_inp[:, 1:]

        # familiar lasagne crossentropy
        model_loss = lasagne.objectives.categorical_crossentropy(
            predicted_probas.reshape([-1, self.n_tokens]),
            references.ravel()
        ).mean()

        # Regularizer for kicks
        from lasagne.regularization import regularize_network_params, l2
        reg_l2 = regularize_network_params(resolver, l2) * 10 ** -5
        loss = model_loss + reg_l2
        updates = lasagne.updates.adadelta(loss, self.weights, learning_rate=self.lr)
        self.train_fun = theano.function([self.G_inp], [loss], updates=updates)
        self.evaluation_fun = theano.function([self.G_inp], [loss, model_loss, reg_l2])
        self.get_sequences = theano.function([self.gen_batch_size], self.generated_action_seq)

    def get_random_mol(self, mols, token2idx, n):
        """Returns random molecule's SMILES."""
        idx = random.randint(0, n - 1)
        mol = mols[idx]
        return [token2idx[tok] for tok in mol]

    def generate_sequence(self, mols, token2idx, batch_size=10, crop_length=100):
        """
        Picking batch_size number of sequences from mols randomly
        """
        sequences = []
        for i in range(batch_size):
            seq = []
            while len(seq) < crop_length:
                seq += [0] + self.get_random_mol(mols, token2idx, len(mols))
            seq = seq[:crop_length]
            sequences.append(seq)
        return np.array(sequences, dtype='int32')


    def train_model(self, mols, alphabet, token2idx, n_epochs=100, cur_epoch=0, log_loss_full = []):
        for i in range(n_epochs):

            cur_epoch = cur_epoch + 1
            new_batch = self.generate_sequence(mols, token2idx)
            self.train_fun(new_batch)

            if i % 100 == 0:
                plt.plot(log_loss_full)
                plt.show()

            # Display metrics once in a while
            if i % self.REPORT_RATE == 0:

                loss_components = self.evaluation_fun(new_batch)
                log_loss_full.append(loss_components[0])
                print "iter:%i\tfull:%.5f\tllh:%.5f\treg:%.5f" % tuple([i] + map(float, loss_components))

                examples = self.get_sequences(3)
                for tid_line in examples:
                    print ''.join(map(alphabet.__getitem__, tid_line))
                    
        return log_loss_full
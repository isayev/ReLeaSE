import theano
import lasagne
from lasagne.regularization import regularize_network_params, l2

import time
import numpy as np


class OracleNetwork(object):
    def __init__(self, inp_seq, target, generated_action_seq, num_tokens, lstm_units = 100, dense_units = 100, seq_len = 100,
                 grad_clip = 100, learning_rate = 0.0001):
        self.inp_seq = inp_seq
        self.target = target
        self.seq_len = seq_len
        self.num_tokens = num_tokens
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.generated_action_seq = generated_action_seq
        self.grad_clip = grad_clip
        #self.generator = generator

    def build_model(self):
        self.l_obs = lasagne.layers.InputLayer(shape=(None, self.seq_len))

        # embedding the input layer
        l_emb = lasagne.layers.EmbeddingLayer(self.l_obs, input_size=self.num_tokens, output_size=100,
                                                   name="D1 input_embedding")

        # LSTMs
        l_rnn = lasagne.layers.LSTMLayer(l_emb, num_units=self.lstm_units,
                                         grad_clipping=self.grad_clip,
                                         nonlinearity=lasagne.nonlinearities.tanh)
        l_slice = lasagne.layers.SliceLayer(l_rnn, -1, 1)
        
        l_dense = lasagne.layers.DenseLayer(l_slice, num_units=self.dense_units,
                                            nonlinearity=lasagne.nonlinearities.rectify)
        self.features = lasagne.layers.get_output(l_dense, self.inp_seq)

        self.l_out = lasagne.layers.DenseLayer(l_dense, num_units=1, nonlinearity=lasagne.nonlinearities.identity)

        self.prediction = lasagne.layers.get_output(self.l_out, {self.l_obs: self.inp_seq}, deterministic=True)
        
        oracle = lasagne.layers.get_output(self.l_out, {self.l_obs: self.generated_action_seq}, deterministic=True)

        reg_l2 = regularize_network_params(self.l_out, l2) * 1e-4

        self.loss = lasagne.objectives.squared_error(self.target, self.prediction).mean() + reg_l2

        params = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adadelta(self.loss, params, learning_rate=self.learning_rate)

        self.train_fn = theano.function([self.inp_seq, self.target], self.loss, updates=updates)
        self.test_fn = theano.function([self.inp_seq, self.target], self.loss)
        self.pred_fn = theano.function([self.inp_seq], self.prediction)

        return oracle

    def iterate_minibatches(self, X, y, batchsize):
        n = X.shape[0]
        ind = np.random.permutation(n)
        for start_index in range(0, n, batchsize):
            X_batch = X[ind[start_index:start_index + batchsize], :]
            y_batch = y[ind[start_index:start_index + batchsize], :]
            yield (X_batch, y_batch)

    def oracle_train(self, X_train, y_train, X_test, y_test, batch_size, num_epochs, train_loss_log = [], val_loss_log = []):
        for epoch in range(num_epochs):
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(X_train, y_train, batch_size):
                inputs, targets = batch
                train_err_batch = self.train_fn(inputs.astype('int32'), targets.astype('float32'))
                train_err += train_err_batch
                train_batches += 1

            # And a full pass over the validation data:
            val_loss = 0
            val_batches = 0
            for batch in self.iterate_minibatches(X_test, y_test, batch_size):
                inputs, targets = batch
                val_loss += self.test_fn(inputs.astype('int32'), targets.astype('float32'))
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))

            train_loss_log.append(train_err / train_batches / batch_size)
            val_loss_log.append(val_loss / val_batches / batch_size)
            print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches / batch_size))
            # print("  train accuracy:\t\t{:.2f} %".format(
            #    train_acc / train_batches * 100))
            print("  validation loss:\t\t{:.2f}".format(
                val_loss / val_batches / batch_size))

        return train_loss_log, val_loss_log

    def reset_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        params = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adam(self.loss, params, learning_rate=learning_rate)

        self.train_fn = theano.function([self.inp_seq, self.target], self.loss, updates=updates)
        self.test_fn = theano.function([self.inp_seq, self.target], self.loss)
        self.pred_fn = theano.function([self.inp_seq], self.prediction)

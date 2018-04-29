from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable

import time
import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from sklearn.ensemble import RandomForestRegressor as RFR

torch.manual_seed(1)


class RecurrentQSAR(nn.Module):
    def __init__(self, input_dim, data):
        super(RecurrentQSAR, self).__init__()
        
        self.data = data
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=250,
                                      padding_idx=0)  # Output: (N, W, embedding_dim)
        self.lstm = nn.LSTM(input_size=250, hidden_size=100, bidirectional=True,
                            num_layers=1)  # input(seq_len, batch, input_size)
        self.linear = torch.nn.Linear(in_features=200, out_features=100)
        self.relu = torch.nn.LeakyReLU() #SELU()logp,  #LeakyReLU()
        self.batch_norm = torch.nn.BatchNorm1d(num_features=100)
        self.output = torch.nn.Linear(in_features=100, out_features=1)
        

    def forward(self, inp, dropout=False):
        embedded = self.embedding(inp).permute(1, 0, 2)
        output = embedded
        output, _ = self.lstm(output)
        output = output[-1, :, :]
        output = self.linear(output)
        output = self.relu(output)
        output = self.output(output)
        return output

    def step(self, x, y, criterion, optimizer):
        # Reset gradient
        optimizer.zero_grad()
        fx = self.forward(x)
        loss = criterion(fx, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), max_norm=100)
        optimizer.step()
        return loss.data

    def predict(self, x):
        output = self.forward(x, dropout=False)
        return output.data
    
    def batch_char_tensor(self, smiles, use_cuda=True):
        tensor = torch.zeros(len(smiles), len(smiles[0])).long()
        for i in range (len(smiles)):
            string = smiles[i]
            for c in range(len(string)):
                tensor[i, c] = self.data.all_characters.index(string[c])
        if use_cuda:
            return Variable(tensor.cuda())
        else:
            return Variable(tensor)

    def iterate_minibatches(self, X, y, batchsize=100, use_cuda=True):
        n = X.shape[0]
        ind = np.random.permutation(n)
        for start_index in range(0, n, batchsize):
            X_batch = self.batch_char_tensor(X[ind[start_index:start_index + batchsize]])
            y_batch = y[ind[start_index:start_index + batchsize]]
            if use_cuda:
                yield (X_batch, Variable(torch.from_numpy(y_batch).float().cuda()))
            else:
                yield (X_batch, Variable(torch.from_numpy(y_batch).float()))

    def fit(self, criterion, optimizer, trX, trY, train_loss_log=[], num_epochs=100, batch_size=100):
        for epoch in range(num_epochs):
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(trX, trY, batch_size):
                inputs, targets = batch
                train_err_batch = self.step(inputs, targets, criterion, optimizer)
                train_err += train_err_batch.cpu().numpy().mean()
                train_batches += 1
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            train_loss_log.append(train_err / train_batches / batch_size)
            print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches / batch_size))
            # print("  train accuracy:\t\t{:.2f} %".format(
            #    train_acc / train_batches * 100))

        return train_loss_log

    def validate(self, teX, teY, batch_size=100, val_loss_log=[]):
        # Full pass over the validation data:
        val_loss = 0
        val_batches = 0
        for batch in self.iterate_minibatches(teX, teY, batch_size):
            inputs, targets = batch
            pred = self.predict(inputs)
            val_loss += ((pred - targets.data) ** 2).cpu().numpy().mean()
            val_batches += 1
        val_loss_log.append(val_loss / val_batches / batch_size)
        print("  validation loss:\t\t{:.6f}".format(
            val_loss / val_batches / batch_size))

        return val_loss_log
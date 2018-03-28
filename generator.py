import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class StackAugmentedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, stack_width, stack_depth,
                 use_cuda=True, n_layers=1):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.stack_width = stack_width
        self.stack_depth = stack_depth
        self.use_cuda = use_cuda
        self.n_layers = n_layers

        self.stack_controls_layer = nn.Linear(in_features=self.hidden_size * 2,
                                              out_features=3)

        self.stack_input_layer = nn.Linear(in_features=self.hidden_size * 2,
                                           out_features=self.stack_width)

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size + stack_width, hidden_size, n_layers,
                           bidirectional=True)
        self.decoder = nn.Linear(hidden_size * 2, output_size)
        super(StackAugmentedRNN, self).__init__()


    def forward(self, inp, hidden, cell, stack):
        inp = self.encoder(inp.view(1, -1))
        hidden_2_stack = torch.cat((hidden[0], hidden[1]), dim=1)
        stack_controls = self.stack_controls_layer(hidden_2_stack)
        stack_controls = F.softmax(stack_controls)
        stack_input = self.stack_input_layer(hidden_2_stack.unsqueeze(0))
        stack_input = F.tanh(stack_input)
        stack = self.stack_augmentation(stack_input.permute(1, 0, 2),
                                       stack, stack_controls)
        stack_top = stack[:, 0, :].unsqueeze(0)
        inp = torch.cat((inp, stack_top), dim=2)
        output, (hidden, cell) = self.rnn(inp.view(1, 1, -1), (hidden, cell))
        output = self.decoder(output.view(1, -1))
        return output, hidden, cell, stack

    def stack_augmentation(self, input_val, prev_stack, controls):
        batch_size = prev_stack.size(0)

        controls = controls.view(-1, 3, 1, 1)
        zeros_at_the_bottom = torch.zeros(batch_size, 1, self.stack_width)
        if self.use_cuda:
            zeros_at_the_bottom = Variable(zeros_at_the_bottom.cuda())
        else:
            zeros_at_the_bottom = Variable(zeros_at_the_bottom)
        a_push, a_pop, a_no_op = controls[:, 0], controls[:, 1], controls[:, 2]
        stack_down = torch.cat((prev_stack[:, 1:], zeros_at_the_bottom), dim=1)
        stack_up = torch.cat((input_val, prev_stack[:, :-1]), dim=1)
        new_stack = a_no_op * prev_stack + a_push * stack_up + a_pop * stack_down
        return new_stack

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers * 2, 1, self.hidden_size).cuda())

    def init_cell(self):
        return Variable(torch.zeros(self.n_layers * 2, 1, self.hidden_size).cuda())

    def init_stack(self):
        result = torch.zeros(1, self.stack_depth, self.stack_width)
        if self.use_cuda:
            return Variable(result.cuda())
        else:
            return Variable(result)


class Utils():
    def __init__(self):


    def random_chunk():
        index = random.randint(0, file_len)
        return file[index]


    # Turn string into list of longs
    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_characters.index(string[c])
        return Variable(tensor.cuda())


    def random_training_set(self):
        chunk = random_chunk()
        inp = char_tensor(chunk[:-1])
        target = char_tensor(chunk[1:])
        return inp, target


    def evaluate(self, prime_str='<', end_token='>', predict_len=100, temperature=0.8):
        hidden = decoder.init_hidden()
        cell = decoder.init_cell()
        stack = decoder.initStack()
        prime_input = char_tensor(prime_str)
        predicted = prime_str

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str)):
            _, hidden, cell, stack = decoder(prime_input[p], hidden, cell, stack)
        inp = prime_input[-1]

        for p in range(predict_len):
            output, hidden, cell, stack = decoder(inp, hidden, cell, stack)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_char = all_characters[top_i]
            predicted += predicted_char
            inp = char_tensor(predicted_char)
            if predicted_char == '>':
                break

        return predicted


    def time_since(since):
        s = time.time() - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def train(self, inp, target, decoder, criterion, decoder_optimizer):
        hidden = decoder.init_hidden()
        cell = decoder.init_cell()
        stack = decoder.initStack()
        decoder.zero_grad()
        loss = 0
        for c in range(len(inp)):
            output, hidden, cell, stack = decoder(inp[c], hidden, cell, stack)
            loss += criterion(output, target[c])

        loss.backward()
        decoder_optimizer.step()

        return loss.data[0] / len(inp)


    def main(self, file, n_epochs=3000, checkpoint=None):
        chunk_len = 1

        tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
                  '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
                  '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

        all_characters = ''.join(tokens)
        n_characters = len(all_characters)
        file_len = len(file)

        n_epochs = 100000
        print_every = 100
        plot_every = 10
        hidden_size = 500
        stack_width = 100
        stack_depth = 100
        n_layers = 1
        lr = 0.01
        all_losses = []

        decoder = StackAugmentedRNN(n_characters, hidden_size, n_characters, stack_width,
                                    stack_depth, n_layers)
        decoder = decoder.cuda()
        decoder_optimizer = torch.optim.Adadelta(decoder.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        if checkpoint is not None:
            weights = torch.load(checkpoint)
            decoder.load_state_dict(weights)

        start = time.time()
        loss_avg = 0

        for epoch in range(1, n_epochs + 1):
            loss = train(*random_training_set(), decoder, criterion, decoder_optimizer)
            loss_avg += loss

            if epoch % print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
                print(evaluate('<', 100), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0

class Generator(StackAugmentedRNN):
    def __init__(self, use_cuda=None, optimizer=None, lr=0.01):
        self.n_characters = 45
        self.hidden_size = 500
        self.stack_width = 100
        self.stack_depth = 100
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        #if model_type == 'StackRNN':
        #    self.decoder = StackAugmentedRNN(self.n_characters, self.hidden_size, self.n_characters,
        #                                     self.stack_width=100, self.stack_depth=100)
        #elif model_type == 'VanillaLSTM':
        #    raise NotImplementedError('Implementation in progress')
        #else:
        #    raise ValueError('Invalid value for argument "model_type": should be either "StackRNN" or "VanillaLSTM"')
        if self.use_cuda:
            self.decoder = self.decoder.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adadelta(self.decoder.parameters(), lr=lr)
        super(Generator, self).__init__()


    #def forward(self):
    #    raise NotImplementedError('Function must be overridden')

    #def train(self):
    #   raise NotImplementedError('Function must be overridden')


    def train(inp, target, decoder, criterion, decoder_optimizer):
        hidden = decoder.init_hidden()
        cell = decoder.init_cell()
        stack = decoder.initStack()
        decoder.zero_grad()
        loss = 0
        for c in range(len(inp)):
            output, hidden, cell, stack = decoder(inp[c], hidden, cell, stack)
            loss += criterion(output, target[c])

        loss.backward()
        decoder_optimizer.step()

        return loss.data[0] / len(inp)


    def main(self, file, n_epochs=3000, checkpoint=None):
        chunk_len = 1

        tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
                  '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
                  '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

        all_characters = ''.join(tokens)
        n_characters = len(all_characters)
        file_len = len(file)

        n_epochs = 100000
        print_every = 100
        plot_every = 10
        hidden_size = 500
        stack_width = 100
        stack_depth = 100
        n_layers = 1
        lr = 0.01
        all_losses = []

        decoder = StackAugmentedRNN(n_characters, hidden_size, n_characters, stack_width,
                                    stack_depth, n_layers)
        decoder = decoder.cuda()
        decoder_optimizer = torch.optim.Adadelta(decoder.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        if checkpoint is not None:
            weights = torch.load(checkpoint)
            decoder.load_state_dict(weights)

        start = time.time()
        loss_avg = 0

        for epoch in range(1, n_epochs + 1):
            loss = train(*random_training_set(), decoder, criterion, decoder_optimizer)
            loss_avg += loss

            if epoch % print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
                print(evaluate('<', 100), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0

















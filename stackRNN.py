import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class StackAugmentedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, stack_width, stack_depth,
                 use_cuda=None, n_layers=1, optimizer=None, lr=0.01):
        super(StackAugmentedRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.stack_width = stack_width
        self.stack_depth = stack_depth

        self.use_cuda = use_cuda
        if self.use_cuda is None:
            cd

        self.n_layers = n_layers

        self.stack_controls_layer = nn.Linear(in_features=self.hidden_size * 2,
                                              out_features=3)

        self.stack_input_layer = nn.Linear(in_features=self.hidden_size * 2,
                                           out_features=self.stack_width)

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size + stack_width, hidden_size, n_layers,
                           bidirectional=True)
        self.decoder = nn.Linear(hidden_size * 2, output_size)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adadelta(self.decoder.parameters(), lr=lr)

    def change_lr(self, new_lr):
        # ADD CUSTOM OPTIMIZER
        """Sets generator optimizer learning rate to new_lr."""
        self.optimizer = torch.optim.Adadelta(self.parameters(), lr=new_lr)
        self.lr = new_lr

    def forward(self, inp, hidden, cell, stack):
        """Generator forward function."""
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
        if self.use_cuda:
            return Variable(torch.zeros(self.n_layers * 2, 1, self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(self.n_layers * 2, 1, self.hidden_size))

    def init_cell(self):
        if self.use_cuda:
            return Variable(torch.zeros(self.n_layers * 2, 1, self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(self.n_layers * 2, 1, self.hidden_size))

    def init_stack(self):
        result = torch.zeros(1, self.stack_depth, self.stack_width)
        if self.use_cuda:
            return Variable(result.cuda())
        else:
            return Variable(result)

    def train_step(self, inp, target):
        hidden = self.init_hidden()
        cell = self.init_cell()
        stack = self.initStack()
        self.zero_grad()
        loss = 0
        for c in range(len(inp)):
            output, hidden, cell, stack = self(inp[c], hidden, cell, stack)
            loss += self.criterion(output, target[c])

        loss.backward()
        self.optimizer.step()

        return loss.data[0] / len(inp)


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
            inp = self.char_tensor(predicted_char)
            if predicted_char == '>':
                break

        return predicted


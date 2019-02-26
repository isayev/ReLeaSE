"""
This class implements generative recurrent neural network with augmented memory
stack as proposed in https://arxiv.org/abs/1503.01007
There are options of using LSTM or GRU, as well as using the generator without
memory stack.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import time
from tqdm import trange

from utils import time_since

from smiles_enumerator import SmilesEnumerator


class StackAugmentedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_type='GRU',
                 n_layers=1, is_bidirectional=False, has_stack=False,
                 stack_width=None, stack_depth=None, use_cuda=None,
                 optimizer_instance=torch.optim.Adadelta, lr=0.01):
        """
        Constructor for the StackAugmentedRNN object.

        Parameters
        ----------
        input_size: int
            number of characters in the alphabet

        hidden_size: int
            size of the RNN layer(s)

        output_size: int
            again number of characters in the alphabet

        layer_type: str (default 'GRU')
            type of the RNN layer to be used. Could be either 'LSTM' or 'GRU'.

        n_layers: int (default 1)
            number of RNN layers

        is_bidirectional: bool (default False)
            parameter specifying if RNN is bidirectional

        has_stack: bool (default False)
            parameter specifying if augmented memory stack is used

        stack_width: int (default None)
            if has_stack is True then this parameter defines width of the
            augmented stack memory

        stack_depth: int (default None)
            if has_stack is True then this parameter define depth of the augmented
            stack memory. Hint: no need fo stack depth to be larger than the
            length of the longest sequence you plan to generate

        use_cuda: bool (default None)
            parameter specifying if GPU is used for computations. If left
            unspecified, GPU will be used if available

        optimizer_instance: torch.optim object (default torch.optim.Adadelta)
            optimizer to be used for training

        lr: float (default 0.01)
            learning rate for the optimizer

        """
        super(StackAugmentedRNN, self).__init__()
        
        if layer_type not in ['GRU', 'LSTM']:
            raise InvalidArgumentError('Layer type must be GRU or LSTM')
        self.layer_type = layer_type
        self.is_bidirectional = is_bidirectional
        if self.is_bidirectional:
            self.num_dir = 2
        else:
            self.num_dir = 1
        if layer_type == 'LSTM':
            self.has_cell = True
        else:
            self.has_cell = False
        self.has_stack = has_stack
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if self.has_stack:
            self.stack_width = stack_width
            self.stack_depth = stack_depth

        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

        self.n_layers = n_layers
        
        if self.has_stack:
            self.stack_controls_layer = nn.Linear(in_features=self.hidden_size *
                                                              self.num_dir,
                                                  out_features=3)

            self.stack_input_layer = nn.Linear(in_features=self.hidden_size *
                                                           self.num_dir,
                                               out_features=self.stack_width)

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.has_stack:
            rnn_input_size = hidden_size + stack_width
        else:
            rnn_input_size = hidden_size
        if self.layer_type == 'LSTM':
            self.rnn = nn.LSTM(rnn_input_size, hidden_size, n_layers,
                               bidirectional=self.is_bidirectional)
            self.decoder = nn.Linear(hidden_size * self.num_dir, output_size)
        elif self.layer_type == 'GRU':
            self.rnn = nn.GRU(rnn_input_size, hidden_size, n_layers,
                             bidirectional=self.is_bidirectional)
            self.decoder = nn.Linear(hidden_size * self.num_dir, output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        
        if self.use_cuda:
            self = self.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer_instance = optimizer_instance
        self.optimizer = self.optimizer_instance(self.parameters(), lr=lr,
                                                 weight_decay=0.00001)
  
    def load_model(self, path):
        """
        Loads pretrained parameters from the checkpoint into the model.

        Parameters
        ----------
        path: str
            path to the checkpoint file model will be loaded from.
        """
        weights = torch.load(path)
        self.load_state_dict(weights)

    def save_model(self, path):
        """
        Saves model parameters into the checkpoint file.

        Parameters
        ----------
        path: str
            path to the checkpoint file model will be saved to.
        """
        torch.save(self.state_dict(), path)

    def change_lr(self, new_lr):
        """
        Updates learning rate of the optimizer.

        Parameters
        ----------
        new_lr: float
            new learning rate value
        """
        self.optimizer = self.optimizer_instance(self.parameters(), lr=new_lr)
        self.lr = new_lr

    def forward(self, inp, hidden, stack):
        """
        Forward step of the model. Generates probability of the next character
        given the prefix.

        Parameters
        ----------
        inp: torch.tensor
            input tensor that contains prefix string indices

        hidden: torch.tensor or tuple(torch.tensor, torch.tensor)
            previous hidden state of the model. If layer_type is 'LSTM',
            then hidden is a tuple of hidden state and cell state, otherwise
            hidden is torch.tensor

        stack: torch.tensor
            previous state of the augmented memory stack

        Returns
        -------
        output: torch.tensor
            tensor with non-normalized probabilities of the next character

        next_hidden: torch.tensor or tuple(torch.tensor, torch.tensor)
            next hidden state of the model. If layer_type is 'LSTM',
            then next_hidden is a tuple of hidden state and cell state,
            otherwise next_hidden is torch.tensor

        next_stack: torch.tensor
            next state of the augmented memory stack
        """
        inp = self.encoder(inp.view(1, -1))
        if self.has_stack:
            if self.has_cell:
                hidden_ = hidden[0]
            else:
                hidden_ = hidden
            if self.is_bidirectional:
                hidden_2_stack = torch.cat((hidden_[0], hidden_[1]), dim=1)
            else:
                hidden_2_stack = hidden_.squeeze(0)
            stack_controls = self.stack_controls_layer(hidden_2_stack)
            stack_controls = F.softmax(stack_controls, dim=1)
            stack_input = self.stack_input_layer(hidden_2_stack.unsqueeze(0))
            stack_input = torch.tanh(stack_input)
            stack = self.stack_augmentation(stack_input.permute(1, 0, 2),
                                            stack, stack_controls)
            stack_top = stack[:, 0, :].unsqueeze(0)
            inp = torch.cat((inp, stack_top), dim=2)
        output, next_hidden = self.rnn(inp.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, next_hidden, stack

    def stack_augmentation(self, input_val, prev_stack, controls):
        """
        Augmentation of the tensor into the stack. For more details see
        https://arxiv.org/abs/1503.01007

        Parameters
        ----------
        input_val: torch.tensor
            tensor to be added to stack

        prev_stack: torch.tensor
            previous stack state

        controls: torch.tensor
            predicted probabilities for each operation in the stack, i.e
            PUSH, POP and NO_OP. Again, see https://arxiv.org/abs/1503.01007

        Returns
        -------
        new_stack: torch.tensor
            new stack state

        """
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
        """
        Initialization of the hidden state of RNN.

        Returns
        -------
        hidden: torch.tensor
            tensor filled with zeros of an appropriate size (taking into
            account number of RNN layers and directions)
        """
        if self.use_cuda:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size))

    def init_cell(self):
        """
        Initialization of the cell state of LSTM. Only used when layers_type is
        'LSTM'

        Returns
        -------
        cell: torch.tensor
            tensor filled with zeros of an appropriate size (taking into
            account number of RNN layers and directions)
        """
        if self.use_cuda:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size))

    def init_stack(self):
        """
        Initialization of the stack state. Only used when has_stack is True

        Returns
        -------
        stack: torch.tensor
            tensor filled with zeros
        """
        result = torch.zeros(1, self.stack_depth, self.stack_width)
        if self.use_cuda:
            return Variable(result.cuda())
        else:
            return Variable(result)

    def train_step(self, inp, target):
        """
        One train step, i.e. forward-backward and parameters update, for
        a single training example.

        Parameters
        ----------
        inp: torch.tensor
            tokenized training string from position 0 to position (seq_len - 1)

        target:
            tokenized training string from position 1 to position seq_len

        Returns
        -------
        loss: float
            mean value of the loss function (averaged through the sequence
            length)

        """
        hidden = self.init_hidden()
        if self.has_cell:
            cell = self.init_cell()
            hidden = (hidden, cell)
        if self.has_stack:
            stack = self.init_stack()
        else:
            stack = None
        self.optimizer.zero_grad()
        loss = 0
        for c in range(len(inp)):
            output, hidden, stack = self(inp[c], hidden, stack)
            loss += self.criterion(output, target[c].unsqueeze(0))

        loss.backward()
        self.optimizer.step()

        return loss.item() / len(inp)
    
    def evaluate(self, data, prime_str='<', end_token='>', predict_len=100):
        """
        Generates new string from the model distribution.

        Parameters
        ----------
        data: object of type GeneratorData
            stores information about the generator data format such alphabet, etc

        prime_str: str (default '<')
            prime string that will be used as prefix. Deafult value is just the
            START_TOKEN

        end_token: str (default '>')
            when end_token is sampled from the model distribution,
            the generation of a new example is finished

        predict_len: int (default 100)
            maximum length of the string to be generated. If the end_token is
            not sampled, the generation will be aborted when the length of the
            generated sequence is equal to predict_len

        Returns
        -------
        new_sample: str
            Newly generated sample from the model distribution.

        """
        hidden = self.init_hidden()
        if self.has_cell:
            cell = self.init_cell()
            hidden = (hidden, cell)
        if self.has_stack:
            stack = self.init_stack()
        else:
            stack = None
        prime_input = data.char_tensor(prime_str)
        new_sample = prime_str

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str)-1):
            _, hidden, stack = self.forward(prime_input[p], hidden, stack)
        inp = prime_input[-1]

        for p in range(predict_len):
            output, hidden, stack = self.forward(inp, hidden, stack)

            # Sample from the network as a multinomial distribution
            probs = torch.softmax(output, dim=1)
            top_i = torch.multinomial(probs.view(-1), 1)[0].cpu().numpy()

            # Add predicted character to string and use as next input
            predicted_char = data.all_characters[top_i]
            new_sample += predicted_char
            inp = data.char_tensor(predicted_char)
            if predicted_char == end_token:
                break

        return new_sample

    def fit(self, data, n_iterations, all_losses=[], print_every=100,
            plot_every=10, augment=False):
        """
        This methods fits the parameters of the model. Training is performed to
        minimize the cross-entropy loss when predicting the next character
        given the prefix.

        Parameters
        ----------
        data: object of type GeneratorData
            stores information about the generator data format such alphabet, etc

        n_iterations: int
            how many iterations of training will be performed

        all_losses: list (default [])
            list to store the values of the loss function

        print_every: int (default 100)
            feedback will be printed to std_out once every print_every
            iterations of training

        plot_every: int (default 10)
            value of the loss function will be appended to all_losses once every
            plot_every iterations of training

        augment: bool (default False)
            parameter specifying if SMILES enumeration will be used. For mode
            details on SMILES enumeration see https://arxiv.org/abs/1703.07076

        Returns
        -------
        all_losses: list
            list that stores the values of the loss function (learning curve)
        """
        start = time.time()
        loss_avg = 0

        if augment:
            smiles_augmentation = SmilesEnumerator()
        else:
            smiles_augmentation = None

        for epoch in trange(1, n_iterations + 1, desc='Training in progress...'):
            inp, target = data.random_training_set(smiles_augmentation)
            loss = self.train_step(inp, target)
            loss_avg += loss

            if epoch % print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch,
                                               epoch / n_iterations * 100, loss)
                      )
                print(self.evaluate(data=data, prime_str = '<',
                                    predict_len=100), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0
        return all_losses

"""
Unit tests for StackAugmentedRNN class
"""
import sys
sys.path.append('./')
import pytest
import torch
from stackRNN import StackAugmentedRNN
from data import GeneratorData

gen_data_path = './data/logP_labels.csv'
gen_data = GeneratorData(training_data_path=gen_data_path, delimiter=',',
                         cols_to_read=[1], keep_header=False)

hidden_size = 50
stack_width = 50
stack_depth = 10
lr = 0.001
optimizer_instance = torch.optim.Adadelta
use_cuda = True

def test_bidirectional_stack_gru():
    layer_type = 'GRU'
    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters,
                                     hidden_size=hidden_size,
                                     output_size=gen_data.n_characters,
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=True,
                                     has_stack=True,
                                     stack_width=stack_width,
                                     stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     optimizer_instance=optimizer_instance,
                                     lr=lr)
    my_generator = my_generator.cuda()

    losses = my_generator.fit(gen_data, 100)

    my_generator.evaluate(gen_data)

def test_unidirectional_stack_gru():
    layer_type = 'GRU'
    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters,
                                     hidden_size=hidden_size,
                                     output_size=gen_data.n_characters,
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False,
                                     has_stack=True,
                                     stack_width=stack_width,
                                     stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     optimizer_instance=optimizer_instance,
                                     lr=lr)
    my_generator = my_generator.cuda()

    losses = my_generator.fit(gen_data, 100)

    my_generator.evaluate(gen_data)

def test_unidirectional_gru_no_stack():
    layer_type = 'GRU'
    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters,
                                     hidden_size=hidden_size,
                                     output_size=gen_data.n_characters,
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False,
                                     has_stack=False,
                                     stack_width=stack_width,
                                     stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     optimizer_instance=optimizer_instance,
                                     lr=lr)
    my_generator = my_generator.cuda()

    losses = my_generator.fit(gen_data, 100)

    my_generator.evaluate(gen_data)

def test_bidirectional_gru_no_stack():
    layer_type = 'GRU'
    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters,
                                     hidden_size=hidden_size,
                                     output_size=gen_data.n_characters,
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=True,
                                     has_stack=False,
                                     stack_width=stack_width,
                                     stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     optimizer_instance=optimizer_instance,
                                     lr=lr)
    my_generator = my_generator.cuda()

    losses = my_generator.fit(gen_data, 100)

    my_generator.evaluate(gen_data)

def test_bidirectional_stack_lstm():
    layer_type = 'LSTM'
    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters,
                                     hidden_size=hidden_size,
                                     output_size=gen_data.n_characters,
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=True,
                                     has_stack=True,
                                     stack_width=stack_width,
                                     stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     optimizer_instance=optimizer_instance,
                                     lr=lr)
    my_generator = my_generator.cuda()

    losses = my_generator.fit(gen_data, 100)

    my_generator.evaluate(gen_data)

def test_unidirectional_stack_lstm():
    layer_type = 'LSTM'
    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters,
                                     hidden_size=hidden_size,
                                     output_size=gen_data.n_characters,
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False,
                                     has_stack=True,
                                     stack_width=stack_width,
                                     stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     optimizer_instance=optimizer_instance,
                                     lr=lr)
    my_generator = my_generator.cuda()

    losses = my_generator.fit(gen_data, 100)

    my_generator.evaluate(gen_data)

def test_unidirectional_lstm_no_stack():
    layer_type = 'LSTM'
    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters,
                                     hidden_size=hidden_size,
                                     output_size=gen_data.n_characters,
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False,
                                     has_stack=False,
                                     stack_width=stack_width,
                                     stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     optimizer_instance=optimizer_instance,
                                     lr=lr)
    my_generator = my_generator.cuda()

    losses = my_generator.fit(gen_data, 100)

    my_generator.evaluate(gen_data)

def test_bidirectional_lstm_no_stack():
    layer_type = 'LSTM'
    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters,
                                     hidden_size=hidden_size,
                                     output_size=gen_data.n_characters,
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=True,
                                     has_stack=False,
                                     stack_width=stack_width,
                                     stack_depth=stack_depth,
                                     use_cuda=use_cuda,
                                     optimizer_instance=optimizer_instance,
                                     lr=lr)
    my_generator = my_generator.cuda()

    losses = my_generator.fit(gen_data, 100)

    my_generator.evaluate(gen_data)

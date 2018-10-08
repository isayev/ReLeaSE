import torch

import numpy as np

from stackRNN import StackAugmentedRNN
from predictor import RandomForestQSAR
from data import GeneratorData
from data import PredictorData
from data import sanitize_smiles
from reinforcement import Reinforcement
from ReplayMemory import ReplayMemory


gen_data_path = '/data/masha/generative_model/chembl_22_clean_1576904_sorted_std_final.smi'
egfr_data_path = '/home/mariewelt/Notebooks/PyTorch/data/egfr_with_pubchem.smi'
use_cuda = True
hidden_size = 500
stack_width = 100
stack_depth = 100
lr = 0.01
gen_data = GeneratorData(training_data_path=gen_data_path, use_cuda=use_cuda)
tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
char2idx = {}
gen_data.load_dictionary(tokens, char2idx)
egfr_data = PredictorData(path=egfr_data_path)
egfr_data.binarize(threshold=7.0)

my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                 output_size=gen_data.n_characters, stack_width=stack_width,
                                 stack_depth=stack_depth, use_cuda=use_cuda, n_layers=1,
                                 optimizer='Adadelta', lr=lr)

if use_cuda:
    my_generator = my_generator.cuda()

#my_generator.load_model('/home/mariewelt/Notebooks/PyTorch/Model_checkpoints/generator/policy_gradient_egfr_max')
my_generator.load_model('/home/mariewelt/Notebooks/PyTorch/Model_checkpoints/generator/checkpoint_lstm')

egfr_predictor = RandomForestQSAR(n_estimators=100, n_ensemble=5)
egfr_predictor.load_model('/home/mariewelt/Notebooks/PyTorch/data/RF/EGFR_RF')

RL = Reinforcement(my_generator, egfr_predictor)
replay = ReplayMemory(capacity=10000)

for i in range(len(egfr_data.smiles)):
    if egfr_data.binary_labels[i] == 1.0:
        replay.push(egfr_data.smiles[i])

generated = []
for _ in range(replay.capacity):
    generated.append(my_generator.evaluate(gen_data))

sanitized = sanitize_smiles(generated)

for sm in sanitized:
    if sm is not None:
        replay.push(sm)

for _ in range(20):
    RL.policy_gradient_replay(gen_data, replay)

f = open('generated.smi', 'w')
for _ in range(10):
    generated = my_generator.evaluate(gen_data)
    if generated[-1] == '>':
        sanitized = sanitize_smiles([generated[1:-1]])[0]
        if sanitized is not None:
            f.writelines(sanitized + '\n')
f.close()

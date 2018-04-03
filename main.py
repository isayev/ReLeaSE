import torch

from stackRNN import StackAugmentedRNN
from predictor import RandomForestQSAR
from data import GeneratorData
from data import PredictorData
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
egfr_data = PredictorData(training_data_path=egfr_data_path)
egfr_data.binarize(threshold=7.0)
print('I am here')
my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                 output_size=gen_data.n_characters, stack_width=stack_width,
                                 stack_depth=stack_depth, use_cuda=use_cuda, n_layers=1,
                                 optimizer='Adadelta', lr=lr)
my_generator.load_model('Model_checkpoints/generator/policy_gradient_egfr_max')

egfr_predictor = RandomForestQSAR(n_estimators=100, n_ensemble=5)
egfr_predictor.load_model('/home/mariewelt/Notebooks/PyTorch/data/RF/EGFR_RF')
RL = Reinforcement(my_generator, egfr_predictor)
replay = ReplayMemory()

for i in range(len(egfr_data.smiles)):
    if egfr_data.binary_labels == 1.0:
        replay.push(egfr_data.smiles[i])

for _ in range(10):
    RL.policy_gradient_replay(gen_data, replay)

predicted = my_generator.evaluate(gen_data)
for sm in predicted:
    print(sm)










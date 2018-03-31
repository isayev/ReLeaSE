import torch

from stackRNN import StackAugmentedRNN
from predictor import  RandomForestQSAR
from data import Data


def main():
    gen_data_path = '/data/masha/generative_model/chembl_22_clean_1576904_sorted_std_final.smi'
    egfr_data_path = ''
    use_cuda = True
    hidden_size = 500
    stack_width = 100
    stack_depth = 100
    lr = 0.01
    gen_data = Data(training_data_path=gen_data_path, use_cuda=use_cuda)
    egfr_data = Data(training_data_path=egfr_data_path)
    my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                     output_size=gen_data.n_characters, stack_width=stack_width,
                                     stack_depth=stack_depth, use_cuda=use_cuda, n_layers=1,
                                     optimizer='Adadelta', lr=lr)

    egfr_predictor = RandomForestQSAR()


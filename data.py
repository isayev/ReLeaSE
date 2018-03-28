import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import time
import math
import numpy as np

from rdkit import Chem

CHEMBL_FILENAME = ' '


class Data(object):
    def __init__(self, training_data='ChEMBL', replay_data=None, replay_capacity=10000, use_cuda=None):
        super(Data, self).__init__()
        if training_data == 'ChEMBL':
            self.file, success = self.read_smi_file(CHEMBL_FILENAME, unique=True)
        else:
            self.file, success = self.read_smi_file(training_data, unique=True)
        assert success
        self.file_len = len(self.file)
        self.all_characters, self.char2idx, self.n_characters = self.tokenize(self.file)
        if replay_data is not None:
            self.replay_memory, success = self.read_smi_file(replay_data)
            assert success
        else:
            self.replay_memory = []
        self.replay_capacity = replay_capacity
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

    def push_to_replay_memory(self, smiles):
        """
        Pushes new SMILES to replay memory. Works as queue: first come, first go.
        Args:
            smiles (list): SMILES to be added into replay memory.
        If replay_memory exceeds capacity, old examples are pop out from queue.
        """
        new_smiles = self.canonize_smiles(smiles)
        new_smiles = list(set(new_smiles))
        for sm in new_smiles:
            if not np.isnan(sm):
                self.replay_memory.append(sm)
        if len(self.replay_memory > self.replay_capacity):
            self.replay_memory = self.replay_memory[-self.replay_capacity:]

    def sample_from_replay_memory(self, batch_size=1):
        """
        Samples random examples from replay memory.
        Args:
            batch_size (int): number of examples to sample from replay memory.
        Returns:
            sample_smiles (list): batch_size random SMILES stings from replay_memory.
        """
        return random.sample(self.replay_memory, batch_size)

    def random_chunk(self):
        """
        Samples random SMILES string from generator training data set.
        Returns:
            random_smiles (str).
        """
        index = random.randint(0, self.file_len)
        return self.file[index]

    def char_tensor(self, string):
        """
        Converts SMILES into tensor of indices wrapped into torch.autograd.Variable.
        Args:
            string (str): input SMILES string
        Returns:
            tokenized_string (torch.autograd.Variable(torch.tensor))
        """
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = self.all_characters.index(string[c])
        if self.use_cuda:
            return Variable(tensor.cuda())
        else:
            return Variable(tensor)

    def random_training_set(self):
        chunk = self.random_chunk()
        inp = self.char_tensor(chunk[:-1])
        target = self.char_tensor(chunk[1:])
        return inp, target

    @staticmethod
    def time_since(since):
        s = time.time() - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    @staticmethod
    def tokenize(smiles):
        """
        Returns list of unique tokens, token-2-index dictionary and number of unique tokens from the list of SMILES

        Args:
            smiles (list): list of SMILES strings to tokenize.

        Returns:
            tokens (list): list of unique tokens/SMILES alphabet.
            token2idx (dict): dictionary mapping token to its index.
            num_tokens (int): number of unique tokens.
        """
        tokens = list(set(''.join(smiles)))
        tokens = ''.join(tokens)
        token2idx = dict((token, i) for i, token in enumerate(tokens))
        num_tokens = len(tokens)
        return tokens, token2idx, num_tokens

    @staticmethod
    def read_smi_file(filename, unique=True, ):
        """
        Reads SMILES from file. File must contain one SMILES string per line
        with \n token in the end of the line.

        Args:
            filename (str): path to the file
            unique (bool): return only unique SMILES

        Returns:
            smiles (list): list of SMILES strings from specified file.
            success (bool): defines whether operation was successfully completed or not.

        If 'unique=True' this list contains only unique copies.
        """
        f = open(filename, 'r')
        molecules = []
        for line in f:
            molecules.append(line[:-1])
        if unique:
            molecules = list(set(molecules))
        else:
            molecules = list(molecules)
        f.close()
        return molecules, f.closed

    @staticmethod
    def save_smi_to_file(filename, smiles, unique=True):
        """
        Takes path to file and list of SMILES strings and writes SMILES to the specified file.

            Args:
                filename (str): path to the file
                smiles (list): list of SMILES strings
                unique (bool): parameter specifying whether to write only unique copies or not.

            Output:
                success (bool): defines whether operation was successfully completed or not.
           """
        if unique:
            smiles = list(set(smiles))
        else:
            smiles = list(smiles)
        f = open(filename, 'w')
        for mol in smiles:
            f.writelines([mol, '\n'])
        f.close()
        return f.closed

    @staticmethod
    def canonize_smiles(smiles, sanitize=True):
        """
        Takes list of SMILES strings and returns list of their canonical SMILES.
            Args:
                smiles (list): list of SMILES strings
                sanitize (bool): parameter specifying whether to sanitize SMILES or not.
                For definition of sanitized SMILES check http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol

            Output:
                new_smiles (list): list of canonical SMILES and NaNs if SMILES string is invalid or unsanitized
                (when 'sanitize = True')

            When 'sanitize = True' the function is analogous to: sanitize_smiles(smiles, canonize=True).
        """
        new_smiles = []
        for sm in smiles:
            try:
                new_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=sanitize)))
            except UserWarning(sm + ' can not be canonized: invalid SMILES string!'):
                new_smiles.append(np.nan)
        return new_smiles

    @staticmethod
    def sanitize_smiles(smiles, canonize=True):
        """
        Takes list of SMILES strings and returns list of their sanitized versions.
        For definition of sanitized SMILES check http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
            Args:
                smiles (list): list of SMILES strings
                canonize (bool): parameter specifying whether to return canonical SMILES or not.

            Output:
                new_smiles (list): list of SMILES and NaNs if SMILES string is invalid or unsanitized.
                If 'canonize = True', return list of canonical SMILES.

            When 'canonize = True' the function is analogous to: canonize_smiles(smiles, sanitize=True).
        """
        new_smiles = []
        for sm in smiles:
            try:
                if canonize:
                    new_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=True)))
                else:
                    new_smiles.append(sm)
            except UserWarning('Unsanitized SMILES string: ' + sm):
                new_smiles.append(np.nan)
        return new_smiles

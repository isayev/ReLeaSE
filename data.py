import torch
from torch.autograd import Variable

import random
import time
import math
import numpy as np
import csv
import warnings

from rdkit import Chem
from rdkit import DataStructs


class GeneratorData(object):
    def __init__(self, training_data_path, replay_data=None, replay_capacity=10000, use_cuda=None):
        super(GeneratorData, self).__init__()
        self.file, success = read_smi_file(training_data_path, unique=True)

        assert success
        self.file_len = len(self.file)
        self.all_characters, self.char2idx, self.n_characters = tokenize(self.file)
        if replay_data is not None:
            self.replay_memory, success = read_smi_file(replay_data)
            assert success
        else:
            self.replay_memory = []
        self.replay_capacity = replay_capacity
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

    def load_dictionary(self, tokens, char2idx):
        self.all_characters = tokens
        self.char2idx = char2idx
        self.n_characters = len(tokens)

    def push_to_replay_memory(self, smiles):
        """
        Pushes new SMILES to replay memory. Works as queue: first come, first go.
        Args:
            smiles (list): SMILES to be added into replay memory.
        If replay_memory exceeds capacity, old examples are pop out from queue.
        """
        new_smiles = canonize_smiles(smiles)
        new_smiles = list(set(new_smiles))
        for sm in new_smiles:
            if sm is not None:
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
        index = random.randint(0, self.file_len-1)
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

    def random_training_set(self, smiles_augmentation):
        chunk = self.random_chunk()
        if smiles_augmentation is not None:
            chunk = '<' + smiles_augmentation.randomize_smiles(chunk[1:-1]) + '>'
        inp = self.char_tensor(chunk[:-1])
        target = self.char_tensor(chunk[1:])
        return inp, target

    def read_sdf_file(self, path, fields_to_read):
        raise NotImplementedError
        
    def update_data(self, path):
        self.file, success = read_smi_file(path, unique=True)
        self.file_len = len(self.file)
        assert success


class PredictorData(object):
    def __init__(self, path, delimiter=',', cols=[0, 1], use_cuda=None):
        super(PredictorData, self).__init__()
        self.smiles, self.property = read_smiles_property_file(path, delimiter=delimiter, cols=cols)

        assert len(self.smiles) == len(self.property)
        self.all_characters, self.char2idx, self.n_characters = tokenize(self.smiles)
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        self.binary_labels = None

    def binarize(self, threshold):
        self.binary_labels = np.array(self.property >= threshold, dtype='int32')

    def load_dictionary(self, tokens, char2idx):
        self.all_characters = tokens
        self.char2idx = char2idx
        self.n_characters = len(tokens)


def get_fp(smiles):
    fp = []
    for mol in smiles:
        fp.append(mol2image(mol, n=2048))
    return fp


def mol2image(x, n=2048):
    try:
        m = Chem.MolFromSmiles(x)
        fp = Chem.RDKFingerprint(m, maxPath=4, fpSize=n)
        res = np.zeros(len(fp))
        DataStructs.ConvertToNumpyArray(fp, res)
        return res
    except:
        warnings.warn('Unable to calculate Fingerprint', UserWarning)
        return [np.nan]


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
        except: 
            warnings.warn('Unsanitized SMILES string: ' + sm, UserWarning)
            new_smiles.append('')
    return new_smiles


def canonize_smiles(smiles, sanitize=True):
    """
    Takes list of SMILES strings and returns list of their canonical SMILES.
        Args:
            smiles (list): list of SMILES strings
            sanitize (bool): parameter specifying whether to sanitize SMILES or not.
            For definition of sanitized SMILES check
            http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol

        Output:
            new_smiles (list): list of canonical SMILES and NaNs if SMILES string is invalid or unsanitized
            (when 'sanitize = True')

        When 'sanitize = True' the function is analogous to: sanitize_smiles(smiles, canonize=True).
    """
    new_smiles = []
    for sm in smiles:
        try:
            new_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=sanitize)))
        except:
            warnings.warn(sm + ' can not be canonized: invalid SMILES string!', UserWarning)
            new_smiles.append('')
    return new_smiles


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


def read_smi_file(filename, unique=True):
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


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def cross_validation_split(data, labels, n_folds=5, split='random', folds=None):
    if split not in ['random', 'fixed']:
        raise ValueError('Invalid value for argument \'split\': must be either \'random\' of \'fixed\'')
    n = len(data)
    assert n > 0
    if split == 'fixed' and folds is None:
        raise TypeError('Invalid type for argument \'folds\': found None, but must be list')
    if split == 'random' and folds is not None:
        warnings.warn('\'folds\' argument will be ignored: \'split\' set to random, '
                          'but \'folds\' argument is provided.', UserWarning)

    if split == 'random':
        fold_len = round(n / n_folds)
        folds = []
        for i in range(n_folds):
            folds = folds + [i]*fold_len
        if len(folds) > n:
            folds = folds[:n]
        if len(folds) < n:
            folds = folds + [i]*(n - len(folds))
        assert(len(folds) == n)
        ind = np.random.permutation(n)
        new_data = []
        new_labels = []
        for i in ind:
            new_data.append(data[i])
            new_labels.append(labels[i])
        data = new_data
        labels = new_labels
    
    cross_val_data = []
    cross_val_labels = []
    folds = np.array(folds)
    for f in range(n_folds):
        left = np.where(folds == f)[0].min()
        right = np.where(folds == f)[0].max()
        cross_val_data.append(data[left:right + 1])
        cross_val_labels.append(list(labels[left:right + 1]))

    return cross_val_data, cross_val_labels


def read_smiles_property_file(path, delimiter=',', cols = [0, 1], keep_header=False):
    reader = csv.reader(open(path, 'r'), delimiter=delimiter)
    data_full = np.array(list(reader))
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data_full) > start_position
    smiles = data_full[start_position:, cols[0]]
    labels = np.array(data_full[start_position:, cols[1]], dtype='float')
    assert len(smiles) == len(labels)
    return smiles, labels

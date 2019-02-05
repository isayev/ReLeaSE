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

from sklearn.model_selection import KFold, StratifiedKFold

from mordred import Calculator, descriptors


class GeneratorData(object):
    def __init__(self, training_data_path, replay_data=None, replay_capacity=10000, use_cuda=None):
        super(GeneratorData, self).__init__()
        self.file, success = read_smi_file(training_data_path, unique=True, 
                                          add_start_end_tokens=True)

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
    def __init__(self, path, delimiter=',', cols=[0, 1], get_features=None,
                 has_label=True, labels_start=1, **kwargs):
        super(PredictorData, self).__init__()
        data = read_object_property_file(path, delimiter, cols_to_read=cols)
        if has_label:
            self.objects = np.array(data[:labels_start]).reshape(-1)
            self.y = np.array(data[labels_start:], dtype='float32')
            self.y = self.y.reshape(-1, len(cols) - labels_start)
            if self.y.shape[1] == 1:
                self.y = self.y.reshape(-1)
        else:
            self.objects = np.array(data[:labels_start]).reshape(-1)
            self.y = [None]*len(self.object)
        assert len(self.objects) == len(self.y)
        if get_features is not None:
            self.x, processed_indices, invalid_indices = \
                get_features(self.objects, **kwargs)
            self.invalid_objects = self.objects[invalid_indices]
            self.objects = self.objects[processed_indices]
            self.invalid_y = self.y[invalid_indices]
            self.y = self.y[processed_indices]
        else:
            self.x = self.objects
            self.invalid_objects = None
            self.invalid_y = None
        self.binary_y = None

    def binarize(self, threshold):
        self.binary_y = np.array(self.y >= threshold, dtype='int32')


def get_fp(smiles):
    fp = []
    processed_indices = []
    invalid_indices = []
    for i in range(len(smiles)):
        mol = smiles[i]
        tmp = np.array(mol2image(mol, n=2048))
        if np.isnan(tmp[0]):
            invalid_indices.append(i)
        else:
            fp.append(tmp)
            processed_indices.append(i)
    return np.array(fp), processed_indices, invalid_indices


def get_desc(smiles, calc):
    desc = []
    processed_indices = []
    invalid_indices = []
    for i in range(len(smiles)):
        sm = smiles[i]
        try:
            mol = Chem.MolFromSmiles(sm)
            tmp = np.array(calc(mol))
            desc.append(tmp)
            processed_indices.append(i)
        except:
            invalid_indices.append(i)

    desc_array = np.array(desc)
    return desc_array, processed_indices, invalid_indices


def normalize_desc(desc_array, desc_mean=None):
    desc_array = np.array(desc_array).reshape(len(desc_array), -1)
    ind = np.zeros(desc_array.shape)
    for i in range(desc_array.shape[0]):
        for j in range(desc_array.shape[1]):
            try:
                if np.isfinite(desc_array[i, j]):
                    ind[i, j] = 1
            except:
                pass
    for i in range(desc_array.shape[0]):
        for j in range(desc_array.shape[1]):
            if ind[i, j] == 0:
                desc_array[i, j] = 0
    if desc_mean is None:
        desc_mean = np.mean(desc_array, axis=0)
    for i in range(desc_array.shape[0]):
        for j in range(desc_array.shape[1]):
            if ind[i, j] == 0:
                desc_array[i, j] = desc_mean[j]
    return desc_array, desc_mean


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


def sanitize_smiles(smiles, canonize=True, throw_warning=False):
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
            if throw_warning:
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


def read_smi_file(filename, unique=True, add_start_end_tokens=False):
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
        if add_start_end_tokens:
            molecules.append('<' + line[:-1] + '>')
        else:
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


def cross_validation_split(x, y, n_folds=5, split='random', folds=None):
    assert(len(x) == len(y))
    x = np.array(x)
    y = np.array(y)
    if split not in ['random', 'stratified', 'fixed']:
        raise ValueError('Invalid value for argument \'split\': '
                         'must be either \'random\', \'stratified\' '
                         'or \'fixed\'')
    if split == 'random':
        cv_split = KFold(n_splits=n_folds, shuffle=True)
        folds = list(cv_split.split(x, y))
    elif split == 'stratified':
        cv_split = StratifiedKFold(n_splits=n_folds, shuffle=True)
        folds = list(cv_split.split(x, y))
    elif split == 'fixed' and folds is None:
        raise TypeError(
            'Invalid type for argument \'folds\': found None, but must be list')
    cross_val_data = []
    cross_val_labels = []
    if len(folds) == n_folds:
        for fold in folds:
            cross_val_data.append(x[fold[1]])
            cross_val_labels.append(y[fold[1]])
    elif len(folds) == len(x) and np.max(folds) == n_folds:
        for f in range(n_folds):
            left = np.where(folds == f)[0].min()
            right = np.where(folds == f)[0].max()
            cross_val_data.append(x[left:right + 1])
            cross_val_labels.append(y[left:right + 1])

    return cross_val_data, cross_val_labels


def read_object_property_file(path, delimiter=',', cols_to_read=[0, 1],
                              keep_header=False):
    reader = csv.reader(open(path, 'r'), delimiter=delimiter)
    data_full = np.array(list(reader))
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data_full) > start_position
    data = [[] for _ in range(len(cols_to_read))]
    for i in range(len(cols_to_read)):
        col = cols_to_read[i]
        data[i] = data_full[start_position:, col]

    return data

import torch

import random
import numpy as np

from utils import read_smi_file, tokenize, read_object_property_file


class GeneratorData(object):
    """
    Docstring coming soon...
    """
    def __init__(self, training_data_path, tokens=None, start_token='<', 
                 end_token='>', pad_symbol=' ', max_len=120, use_cuda=None,
                 **kwargs):
        """
        Constructor for the GeneratorData object.

        Parameters
        ----------
        training_data_path: str
            path to file with training dataset. Training dataset must contain
            a column with training strings. The file also may contain other
            columns.

        tokens: list (default None)
            list of characters specifying the language alphabet. Of left
            unspecified, tokens will be extracted from data automatically.

        start_token: str (default '<')
            special character that will be added to the beginning of every
            sequence and encode the sequence start.

        end_token: str (default '>')
            special character that will be added to the end of every
            sequence and encode the sequence end.

        max_len: int (default 120)
            maximum allowed length of the sequences. All sequences longer than
            max_len will be excluded from the training data.

        use_cuda: bool (default None)
            parameter specifying if GPU is used for computations. If left
            unspecified, GPU will be used if available

        kwargs: additional positional arguments
            These include cols_to_read (list, default [0]) specifying which
            column in the file with training data contains training sequences
            and delimiter (str, default ',') that will be used to separate
            columns if there are multiple of them in the file.

        """
        super(GeneratorData, self).__init__()

        if 'cols_to_read' not in kwargs:
            kwargs['cols_to_read'] = []

        data = read_object_property_file(training_data_path,
                                                       **kwargs)
        self.start_token = start_token
        self.end_token = end_token
        self.pad_symbol = pad_symbol
        self.file = []
        for i in range(len(data)):
            if len(data[i]) <= max_len:
                self.file.append(self.start_token + data[i] + self.end_token) 
        self.file_len = len(self.file)
        self.all_characters, self.char2idx, \
        self.n_characters = tokenize(self.file, tokens)
        self.pad_symbol_idx = self.all_characters.index(self.pad_symbol)
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

    def load_dictionary(self, tokens, char2idx):
        self.all_characters = tokens
        self.char2idx = char2idx
        self.n_characters = len(tokens)

    def random_chunk(self, batch_size):
        """
        Samples random SMILES string from generator training data set.
        Returns:
            random_smiles (str).
        """
        index = np.random.randint(0, self.file_len-1, batch_size)
        return [self.file[i][:-1] for i in index], \
               [self.file[i][1:] for i in index]

    def seq2tensor(self, seqs, tokens, flip=True):
        tensor = np.zeros((len(seqs), len(seqs[0])))
        for i in range(len(seqs)):
            for j in range(len(seqs[i])):
                if seqs[i][j] in tokens:
                    tensor[i, j] = tokens.index(seqs[i][j])
                else:
                    tokens = tokens + [seqs[i][j]]
                    tensor[i, j] = tokens.index(seqs[i][j])
        if flip:
            tensor = np.flip(tensor, axis=1).copy()
        return tensor, tokens

    def pad_sequences(self, seqs, max_length=None, pad_symbol=' '):
        if max_length is None:
            max_length = -1
            for seq in seqs:
                max_length = max(max_length, len(seq))
        lengths = []
        for i in range(len(seqs)):
            cur_len = len(seqs[i])
            lengths.append(cur_len)
            seqs[i] = seqs[i] + pad_symbol * (max_length - cur_len)
        return seqs, lengths

    def random_training_set(self, batch_size):
        inp, target = self.random_chunk(batch_size)
        inp_padded, _ = self.pad_sequences(inp)
        inp_tensor, self.all_characters = self.seq2tensor(inp_padded,
                                                          tokens=self.all_characters,
                                                          flip=False)
        target_padded, _ = self.pad_sequences(target)
        target_tensor, self.all_characters = self.seq2tensor(target_padded,
                                                             tokens=self.all_characters,
                                                             flip=False)
        self.n_characters = len(self.all_characters)
        inp_tensor = torch.tensor(inp_tensor).long()
        target_tensor = torch.tensor(target_tensor).long()
        if self.use_cuda:
            inp_tensor = inp_tensor.cuda()
            target_tensor = target_tensor.cuda()
        return inp_tensor, target_tensor

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

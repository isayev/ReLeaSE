import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns


def tokenize(smiles):
    tokens = list(set(''.join(smiles)))
    tokens = ''.join(tokens)
    token2idx = dict((token, i) for i, token in enumerate(tokens))
    num_tokens = len(tokens)
    return tokens, token2idx, num_tokens

def oracle_data_preprocess(features_filename, labels_filename, seq_len = 100):
    reader = csv.reader(open(features_filename, 'r'), delimiter=',')
    data_full = np.array(list(reader))
    feature_names = data_full[0, 1:]
    features = np.array(data_full[1:, 1:], dtype='float')

    reader = csv.reader(open(labels_filename, 'r'), delimiter=',')
    data_full = np.array(list(reader))
    smiles_old = data_full[1:, 1]
    labels = np.array(data_full[1:, 2], dtype='float')

    smiles_len = []
    for mol in smiles_old:
        smiles_len.append(len(mol))

    idxs = np.where(np.array(smiles_len) <= seq_len)[0]

    smiles_old = smiles_old[idxs]
    labels = labels[idxs]
    smiles_len = np.array(smiles_len)[idxs]

    smiles = []
    for mol in smiles_old:
        if len(mol) < seq_len:
            mol_len = len(mol)
            mol = ''.join([mol, ' ' * (seq_len - mol_len)])
        smiles.append(mol)

    return features, feature_names, smiles, labels


def data_preprocessing(filename, pr=False, SEQ_LEN=100):
    with open(filename, 'r') as f:
        data = f.read()
    if pr: print 'File is closed:', f.closed
    if pr: print 'Here is how data looks like'
    if pr: print data[:100]
    data = []
    for line in open(filename, 'r'):
        data.append(line)
    if pr: print 'File is closed:', f.closed
    n = len(data)
    if pr: print 'Number of compound in dataset:', n
    k = 0
    names = []
    mols = []
    # alphabet = []
    for molecule in data:
        fl = False
        tmp_mol = []
        tmp_name = []
        for l in list(molecule):
            if l == '\t':
                fl = True
                continue
            elif l == '\n':
                fl = False
                # alphabet = alphabet + list(set(tmp_mol))
                mols.append(''.join(tmp_mol))
                names.append(''.join(tmp_name))
                tmp_mol = []
                tmp_name = []
            if fl:
                tmp_name.append(l)
            else:
                tmp_mol.append(l)
    tokens = list('|') + list(set(''.join(mols)))
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

    plt.rc('font', **font)
    
    if pr: print tokens
    tokens = ''.join(tokens)
    if pr: print 'Unique tokens:'
    if pr: print tokens
    if pr: print "Compounds' names"
    if pr: print names[:10]
    if pr: print "Compounds' SMILES"
    if pr: print mols[:10]
    if pr: print 'All names unique?', len(set(names)) == len(names)
    if pr:print 'All compound unique?', len(set(mols)) == len(mols)
    _, unique_ind = np.unique(mols, return_index=True)
    mols = np.array(mols)[np.sort(unique_ind)]
    names = np.array(names)[np.sort(unique_ind)]
    n = len(mols)
    if pr: print 'Number of unique compounds in dataset:', n
    smiles_len = []
    for mol in mols:
        smiles_len.append(len(mol))
    #if pr: plt.hist(smiles_len, normed=1, bins=30)
    #if pr: plt.xlabel("SMILES's length ")
    #if pr: plt.ylabel('Proportion of molecules')
    #if pr: plt.title("Distribution of SMILES's length")
    #if pr: plt.grid(True)
    #if pr: plt.savefig('1.png', dpi=600)
    idxs = np.where(np.array(smiles_len) <= SEQ_LEN)[0]
    mols = mols[idxs]
    names = names[idxs]
    smiles_len = np.array(smiles_len)[idxs]
    n = len(mols)
    if pr: print 'Number of molecules in dataset after truncation:', n
    if pr: plt.hist(smiles_len, normed=1, bins=30)
    if pr: plt.xlabel("Truncated SMILES's length ")
    if pr: plt.ylabel('Proportion of molecules')
    #if pr: plt.title("Distribution of truncated SMILES's length")
    #if pr: plt.grid(True)
    if pr: plt.savefig('2.png', dpi=600)
    return mols, names

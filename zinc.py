import sys
from rdkit import Chem
import numpy as np
import csv
import pickle
import glob, os

directories_1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
dirs = []
for d1 in directories_1:
    for d2 in directories_1:
        dirs.append(d1 + d2)

for d in dirs[:40]:
    zinc_inchi = []
    zinc_ids = []

    for file in glob.glob('/data/mariewelt/ZINC15/' + d + '/*.smi'):
        reader = csv.reader(open(file, 'r'), delimiter=' ')
        data_full = np.array(list(reader))
        if len(data_full) > 0:
            smiles = data_full[1:, 0]
            ids = data_full[1:, 1]
            zinc_inchi = zinc_inchi + [Chem.InchiToInchiKey(Chem.MolToInchi(Chem.MolFromSmiles(sm)))[:14] for sm in
                                       smiles]
            zinc_ids = zinc_ids + [int(x[4:]) for x in ids]

    zinc_inchi_hash = [hash(key) for key in zinc_inchi]
    assert len(zinc_inchi_hash) == len(zinc_ids)
    dic = {'zinc_ids': zinc_ids, 'inchi_hash': zinc_inchi_hash}
    pickle.dump(dic, open('/home/mariewelt/Notebooks/PyTorch/data/ZINC15_/' + d + '.pkl', 'wb'))

    print('Directory ' + d + ' is processed!')

from rdkit import Chem
import pickle


def hashing(list_of_files, where_to_write):

    counter = 0
    my_smiles = []
    for filename in list_of_files:
        f = open(filename, 'r')
        for line in f:
            my_smiles.append(line[:-1])
    f.close()

    inchi_keys = []
    new_smiles = []
    for sm in my_smiles:
        try:
            inchi_keys.append(Chem.InchiToInchiKey(Chem.MolToInchi(Chem.MolFromSmiles(sm)))[:14])
            new_smiles.append(sm)
        except:
            raise UserWarning('SMILES string ' + sm + ' can not be hashed')

    inchi_hash = [hash(key) for key in inchi_keys]

    dic = {'smiles': new_smiles, 'inchi_hash': inchi_hash}
    pickle.dump(dic, open(where_to_write + str(counter) + '.pkl', 'wb'))

    return f.closed()
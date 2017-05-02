'''
This file contains utilization functions for checkpointing the network
'''

import cPickle as pickle
import lasagne
import os

def load(output_layers, snapshot_path):
    if not os.path.exists(snapshot_path):
        raise Exception('File does not exist')
    elif type(output_layers) is not list:
        raise Exception('output_layers should be list, even if single layer')
    else:
        dic = pickle.load(open(snapshot_path, 'rb'))
        if type(dic) is list:
            params = dic
            training_curve = []
        else:
            params = dic['weights']
            training_curve = dic['training_curve']
        n_net_params = len(lasagne.layers.get_all_params(output_layers))
        lasagne.layers.set_all_param_values(output_layers, params[:n_net_params])
        if len(params) > n_net_params:
            print 'Warning: file contains more parameters than network. Last parameters have been left out.'
    print 'Snapshot ' + snapshot_path + ' loaded'
    return training_curve
    
def save(output_layers, snapshot_path, training_curve=[], force_overwrite=False):
    if not force_overwrite and os.path.exists(snapshot_path):
        raise Exception('File exists: set force_overwrite=True to overwrite')
    elif type(output_layers) is not list:
        raise Exception('output_layers should be list, even if single layer')
    else: 
        pickle.dump({'weights': lasagne.layers.get_all_param_values(output_layers),
                     'training_curve': training_curve},
            open(snapshot_path, 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)
    print 'Snapshot saved as ' + snapshot_path
    

import os
import numpy as np

def load_st_dataset(dataset):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'ILIREGIONS':
        # adapt this path to wherever you actually put the file
        data_path = os.path.join(base_dir, 'data', 'ILIREGIONS.npy')
        data = np.load(data_path)                  # shape (T, N, F), no ['data'] here
    elif dataset == 'ILISTATES':
        # adapt this path to wherever you actually put the file
        data_path = os.path.join(base_dir, 'data', 'ILISTATES.npy')
        data = np.load(data_path)   
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

from __future__ import print_function

import hdf5storage
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import scipy.io as sio
from tensorflow import keras

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

print('Loading training data...')
mat1 = hdf5storage.loadmat('training_set_raw.mat')
arr1 = np.array(mat1['features'])
labels1 = np.array(mat1['labels'][:,3])
tr_labels = keras.utils.to_categorical(labels1, num_classes=None)

transformer = QuantileTransformer(output_distribution='uniform')
output1 = transformer.fit_transform(arr1)
del mat1
del arr1
features_training, labels_training = unison_shuffled_copies(output1, tr_labels)
print('Training data ready.')
sio.savemat('D:/data/Raw/6040/training.mat', mdict={'features_training': features_training, 'labels_training': labels_training}, do_compression=True)

print('Loading test data...')
mat2 = hdf5storage.loadmat('test_set_raw.mat')
arr2 = np.array(mat2['features'])
labels2 = np.array(mat2['labels'][:,3])
te_labels = keras.utils.to_categorical(labels2, num_classes=None)
output2 = transformer.transform(arr2)
del mat2
del arr2
features_test, labels_test = unison_shuffled_copies(output2, te_labels)

print('Test data ready.')
sio.savemat('D:/data/Raw/6040/test.mat', mdict={'features_test': features_test, 'labels_test': labels_test}, do_compression=True)

print('Writing output file...')

print('Done!')

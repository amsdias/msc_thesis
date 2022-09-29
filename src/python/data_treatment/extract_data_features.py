"""
    File name: classical_feature_extraction.py
    Author: Diego Cabrera
    Date created: 12/12/2018
    Modified by: Angelo Dias
    Date last modified: 23/06/2018
    Python Version: 3.4
"""

import numpy as np
import pywt
from multiprocessing import Pool
import re
import scipy.io as sio
import os
from os.path import join
import pickle
import scipy.stats
import random

__author__ = "Diego Cabrera"
__copyright__ = "Copyright 2018, The GIDTEC Fault Diagnosis Project"
__credits__ = ["Diego Cabrera"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Diego Cabrera"
__email__ = "dcabrera@ups.edu.ec"
__status__ = "Prototype"

def signal2wp_energy(signal, wavelet, max_level):
    """
    Compute the last level energy of the wavelet packet transform for a signal
    :param signal: time-series
    :type signal: numpy array
    :param wavelet: name of wavelet family
    :type wavelet: string
    :param max_level: level of decomposition
    :type max_level: int
    :return: energies of the last wavelet coefficients
    :rtype: numpy array
    """
    wp_tree = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=max_level)
    level = wp_tree.get_level(max_level, order='freq')
    energy_coef = np.zeros((len(level),))

    for i, node in enumerate(level):
        energy_coef[i] = np.sqrt(np.sum(node.data ** 2)) / node.data.shape[0]

    return energy_coef


def rectified_average(signal):
    """
    Compute rectified average feature
    :param signal: time-series
    :type signal: numpy array
    :return: rectified average of signal
    :rtype: float
    """
    return np.mean(np.abs(signal))


def statistic_features(signal):
    """
    Compute a group of statistical features
    :param signal: time-series
    :type signal: numpy array
    :return: group of statistical feature from the signal
    :rtype: tuple
    """
    mean = np.mean(signal)
    rms = np.sqrt(np.mean(np.square(signal)))
    std_dev = np.std(signal)
    kurtosis = scipy.stats.kurtosis(signal)
    peak = np.max(signal)
    crest = peak / rms
    r_mean = rectified_average(signal)
    form = rms / r_mean
    impulse = peak / r_mean
    variance = std_dev ** 2
    minimum = np.min(signal)
    return mean, rms, std_dev, kurtosis, peak, crest, r_mean, form, impulse, variance, minimum


def band_features(spectrum):
    """
    Compute statistical features from a portion of the spectrum
    :param spectrum: portion of spectrum
    :type spectrum: numpy array
    :return: set of features
    :rtype: tuple
    """
    mean = np.mean(spectrum)
    rms = np.sqrt(np.mean(np.square(spectrum)))
    peak = np.max(spectrum)
    power = np.mean(np.square(spectrum))
    energy = np.sum(np.square(spectrum))
    return mean, rms, peak, power, energy


class Data_set:
    def __init__(self, path, iter,
                 sensor_number=0,
                 time_steps=16384,
                 increment=16384,
                 name_acq='Acc1',
                 wavelet_list=['db7', 'sym3', 'coif4', 'bior6.8', 'rbio6.8'],
                 max_level=6,
                 sample_frequency=50000,
                 n_band=89):
        """
        Class for feature extraction process from a GIDTEC dataset
        :param path: path to the raw signals dataset
        :type path: string
        :param sensor_number: sensor to be processed
        :type sensor_number: int
        :param time_steps: length of sub-signal
        :type time_steps: int
        :param increment: displacement of window
        :type increment: int
        :param name_acq: name of data structure found in each .mat file
        :type name_acq: string
        :param wavelet_list: list of wavelet families to be used
        :type wavelet_list: list
        :param max_level: level of wavelet packet decomposition
        :type max_level: int
        :param sample_frequency: sample frequency of the signals
        :type sample_frequency: int
        :param n_band: number of bands to be cut the spectrum
        :type n_band: int
        """
        self.name_acq = name_acq
        self.sensor_number = sensor_number
        self.time_steps = time_steps
        self.increment = increment
        self.wavelet_list = wavelet_list
        self.max_level = max_level
        self.sample_frequency = sample_frequency
        self.n_band = n_band
        self.features, self.labels = self.processing_dataset(path, iter)

    def processing_dataset(self, path, iter):
        """
        Method to process the dataset in parallel
        :param path: path to raw signals dataset
        :type path: string
        :return: features dataset
        :rtype: numpy array
        """
        files = []
        labels = []
        for dir in os.listdir(path):
            folder = join(path, dir)
            list_files = [join(folder, file) for file in os.listdir(folder)]
            labels.extend(os.listdir(folder))
            files.extend(list_files)

        features = []
        labels = []
        for x in range(iter):
            print('\n######################')
            print(x)
            print('######################\n')
            p = Pool()
            iterador = p.imap(self.feature_extraction_file, files)
            for iteracion in iterador:
                feature, label = iteracion
                features.extend(feature)
                labels.extend(label)
            p.close()

        print('Finish')
        p.close()
        return np.array(features).squeeze(), np.array(labels)

    def feature_extraction_file(self, file):
        """
        Method to process a raw signal file
        :param file: name of raw signal .mat file
        :type file: string
        :return: group of features for the signal in .mat file
        :rtype: numpy array
        """
        label = re.split('R|F|L|P|.mat', file)[-5:-1]
        print(file)
        data = sio.loadmat(file)
        signal = data['data'][self.name_acq][0][0][self.sensor_number]
        ini1 = random.randint(0, 233615)
        ini2 = random.randint(250000, 483615)
        ini3 = random.randint(500000, 733615)
        ini4 = random.randint(750000, 983615)
        idx = []
        ini = 0
        idx.append(ini1)
        idx.append(ini2)
        idx.append(ini3)
        idx.append(ini4)
        group_features = []
        group_label = []
        for x in idx:
            features = []
            sub_signal = signal[x:self.time_steps + x]
            ini += self.increment
                # Time
            time_features = statistic_features(sub_signal)
            features.extend(time_features)
                # Frequency
            freq_sub_signal = np.abs(np.fft.rfft(sub_signal))
            freq_features = statistic_features(freq_sub_signal)
            features.extend(freq_features)
                # Frequency bands
            size_band = int(freq_sub_signal.shape[0] / self.n_band)
            i = 0
            while i + size_band <= freq_sub_signal.shape[0]:
                band_freq_features = band_features(freq_sub_signal[i:i + size_band])
                features.extend(band_freq_features)
                i += size_band
                # Time-frequency
            for i, wavelet in enumerate(self.wavelet_list):
                wavelet_features = signal2wp_energy(sub_signal, wavelet, self.max_level)
                features.extend(wavelet_features)

            group_features.append(features)
            group_label.append(label)
        return np.array(group_features), group_label


if __name__ == "__main__":    
    import numpy as np

    # Path to raw dataset
    path1 = 'D:/data/Raw/6040/training'
    path2 = 'D:/data/Raw/6040/test'
    # Feature extraction process
    dataset1 = Data_set(path1, 80)
    print('Feature dataset size:',dataset1.features.shape)
    print('Label dataset size:',dataset1.labels.shape)

    # Saving for matlab
    sio.savemat('D:/data/Raw/6040/training_set_raw.mat', mdict={'features': dataset1.features, 'labels': dataset1.labels},
                do_compression=True)


    dataset2 = Data_set(path2, 20)
    print('Feature dataset size:',dataset2.features.shape)
    print('Label dataset size:',dataset2.labels.shape)

    # Saving for matlab
    sio.savemat('D:/data/Raw/6040/test_set_raw.mat', mdict={'features': dataset2.features, 'labels': dataset2.labels},
                do_compression=True)

import cv2
import pywt
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

import torch

normal = ['N', 'L', 'R']
ignore = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', "'", '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@']


class Preprocess:
    def __init__(self, signals, annotations, peaks, split_pct, wavelet_name, channel, mit_dir, pickle_path=None):
        self.__signals = signals
        self.__annotations = annotations
        self.__peaks = peaks

        self.__split_pct = split_pct
        self.__wavelet_name = wavelet_name
        self.__channel = channel

        self.__mit_dir = mit_dir
        self.__pickle_path = pickle_path
        self.__beats, self.__mean_len = self.__segment(self.__signals, self.__peaks, self.__annotations)

        self.__normal_beats = self.__beats[0]
        self.__abnormal_beats = self.__beats[1]
        self.__len_split = len(self.__normal_beats) - len(self.__abnormal_beats)

        self.__normal_scalograms, self.__abnormal_scalograms = self.__extract_scalograms(self.__beats,
                                                                                         self.__wavelet_name,
                                                                                         self.__mean_len)

    @property
    def get_normal_beats(self):
        return self.__normal_beats

    @property
    def get_abnormal_beats(self):
        return self.__abnormal_beats

    @property
    def get_normal_scalograms(self):
        return self.__normal_scalograms

    @property
    def get_abnormal_scalograms(self):
        return self.__abnormal_scalograms

    @staticmethod
    def __segment(signals, peaks, annotations):

        iterable = zip(peaks[:-2],
                       peaks[1:-1],
                       peaks[2:],
                       annotations[1:-1])

        normal_beats = []
        abnormal_beats = []
        signals_len = []
        for previous_idx, idx, next_idx, label in iterable:
            diff_1 = (idx - previous_idx) // 2
            diff_2 = (next_idx - idx) // 2

            if label in ignore:
                continue

            beat = (signals[idx - diff_1: idx + diff_2], label)
            if label in normal:
                normal_beats.append(beat)
            else:
                abnormal_beats.append(beat)
            signals_len.append(len(beat[0]))

        mean_len = sum(signals_len) / len(signals_len)
        return (normal_beats, abnormal_beats), mean_len

    def __extract_scalograms(self, beats, wavelet_name, wavelet_y_axis):
        normal_beats, abnormal_beats = beats

        def __load_pickle(path):
            with open(path, "rb") as f:
                var = pickle.load(f)
            return var

        def __save_pickle(var, name):
            base_path = Path("dataset/scalograms")
            if not base_path.exists():
                base_path.mkdir(parents=True, exist_ok=True)
            with open(base_path / Path(f"{name}.pkl"), "wb") as f:
                pickle.dump(var, f)

        def __resize(scalogram):
            return cv2.resize(scalogram, (64, 64))

        def __compute_wavelets(beats):
            scalograms = []
            for beat, label in beats:
                scalogram, _ = pywt.cwt(beat, np.arange(1, wavelet_y_axis + 1), wavelet_name)
                resized_scalogram = __resize(scalogram)
                scalograms.append((resized_scalogram, label))
            return np.array(scalograms, dtype=object)

        pickle_name = f"scalograms_mitdir_{self.__mit_dir}_wavelet_{self.__wavelet_name}_channel_{self.__channel}"
        if not self.__pickle_path or not Path(self.__pickle_path).joinpath(pickle_name).with_suffix(".pkl").exists():
            normal_scalograms = __compute_wavelets(normal_beats)
            abnormal_scalograms = __compute_wavelets(abnormal_beats)
            scalograms = np.concatenate((normal_scalograms, abnormal_scalograms), axis=0)
            __save_pickle(scalograms, pickle_name)

        else:
            scalograms = __load_pickle(Path(self.__pickle_path).joinpath(pickle_name).with_suffix(".pkl"))
            normal_idxs = np.where((scalograms[:, 1] == 'N') | (scalograms[:, 1] == 'L') | (scalograms[:, 1] == 'R'))
            abnormal_idxs = np.where((scalograms[:, 1] != 'N') & (scalograms[:, 1] != 'L') & (scalograms[:, 1] != 'R'))

            normal_scalograms = scalograms[normal_idxs]
            abnormal_scalograms = scalograms[abnormal_idxs]

        return normal_scalograms, abnormal_scalograms

    @staticmethod
    def normalize(features):
        normalized_features = []
        for feature, label in features:
            new_feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
            normalized_features.append((new_feature, label))
        return np.array(normalized_features, dtype=object)

    def shuffle_and_split_dataset(self, dataset):
        idxs = np.arange(len(dataset))
        np.random.shuffle(idxs)

        train_data = dataset[idxs][:int(self.__len_split * self.__split_pct)]
        val_data = dataset[idxs][int(self.__len_split * self.__split_pct) : self.__len_split]
        test_data = dataset[idxs][self.__len_split:]

        return train_data, val_data, test_data

    @staticmethod
    def to_torch_dataloader(dataset, batch_size):
        X_data = dataset[:, 0]
        X_data = np.array([arr.astype('float64') for arr in X_data])
        X_data_torch_ds = torch.from_numpy(X_data).unsqueeze(dim=1)

        loader = DataLoader(
            X_data_torch_ds,
            batch_size=batch_size,
            shuffle=True
        )
        return loader

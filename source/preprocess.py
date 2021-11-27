import pywt
import numpy as np

import torch
from torch.nn.functional import interpolate

normal = ['N', 'L', 'R']
ignore = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', "'", '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@']


class Preprocess:
    def __init__(self, signals, annotations, peaks, split_pct, wavelet_name):
        self.__signals = signals
        self.__annotations = annotations
        self.__peaks = peaks

        self.__split_pct = split_pct
        self.__wavelet_name = wavelet_name
        self.__beats, self.__mean_len = self.__segment(self.__signals, self.__peaks, self.__annotations)

        self.normal_beats = self.__beats[0]
        self.abnormal_beats = self.__beats[1]
        self.len_split = len(self.normal_beats) - len(self.abnormal_beats)

        self.normal_scalograms_torch_ds, \
        self.abnormal_scalograms_torch_ds = self.__extract_scalograms(self.__beats,
                                                                      self.__wavelet_name,
                                                                      self.__mean_len)

    @property
    def get_normal_beats(self):
        return self.normal_beats

    @property
    def get_abnormal_beats(self):
        return self.abnormal_beats

    @property
    def get_normal_scalograms(self):
        return self.normal_scalograms_torch_ds

    @property
    def get_abnormal_scalograms(self):
        return self.abnormal_scalograms_torch_ds

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

    @staticmethod
    def __extract_scalograms(beats, wavelet_name, wavelet_y_axis):
        normal_beats, abnormal_beats = beats

        def __normalize(scalogram):
            return scalogram - np.min(scalogram) / (np.max(scalogram) - np.min(scalogram))

        def __compute_wavelets(beats):
            scalograms = []
            for beat, label in beats:
                scalogram, _ = pywt.cwt(beat, np.arange(1, wavelet_y_axis + 1), wavelet_name)
                normalized_scalogram = __normalize(scalogram)
                scalogram_tensor = torch.from_numpy(normalized_scalogram)

                scalogram_tensor = scalogram_tensor.view(1, 1, scalogram_tensor.shape[0], scalogram_tensor.shape[1])
                scalogram_tensor = interpolate(scalogram_tensor, size=(64, 64))
                scalogram_tensor = torch.squeeze(scalogram_tensor, dim=0)

                scalograms.append((scalogram_tensor, label))
            return scalograms

        normal_scalograms = __compute_wavelets(normal_beats)
        abnormal_scalograms = __compute_wavelets(abnormal_beats)

        return normal_scalograms, abnormal_scalograms

    @staticmethod
    def split_and_shuffle_dataset(dataset):
        shuffled_idxs = torch.randperm(len(dataset))
        shuffled_dataset = dataset[shuffled_idxs]


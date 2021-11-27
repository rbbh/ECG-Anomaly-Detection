import pywt
import numpy as np

import torch
from torch.nn.functional import interpolate

normal = ['N', 'L', 'R']
ignore = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', "'", '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@']


class Preprocess:
    def __init__(self, signals, annotations, wavelet_name):
        self.__signals = signals
        self.__annotations = annotations
        self.__wavelet_name = wavelet_name
        self.__beats, self.__mean_len = self.__segment(self.__signals, self.__annotations)
        self.__normal_scalograms_torch_ds, \
        self.__abnormal_scalograms_torch_ds = self.__extract_wavelets(self.__beats,
                                                                      self.__wavelet_name,
                                                                      self.__mean_len)

    @property
    def get_normal_beats(self):
        return self.__beats[0]

    @property
    def get_abnormal_beats(self):
        return self.__beats[1]

    @property
    def get_normal_scalograms(self):
        return self.__normal_scalograms_torch_ds

    @property
    def get_abnormal_scalograms(self):
        return self.__abnormal_scalograms_torch_ds

    @staticmethod
    def __segment(signals, annotations):

        peak_idx_array = np.array(annotations.sample)
        label_array = np.array(annotations.symbol)

        iterable = zip(peak_idx_array[:-2],
                       peak_idx_array[1:-1],
                       peak_idx_array[2:],
                       label_array[1:-1])

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
    def __extract_wavelets(beats, wavelet_name, wavelet_y_axis):
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

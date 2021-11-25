import numpy as np

ignore = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', "'", '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@']


class Preprocess:
    def __init__(self, signals, annotations, wavelet_name):
        self.__signals = signals
        self.__annotations = annotations
        self.__wavelet_name = wavelet_name

    def segment(self):

        peak_idx_array = np.array(self.__annotations.sample)
        label_array = np.array(self.__annotations.symbol)

        iterable = zip(peak_idx_array[:-2],
                       peak_idx_array[1:-1],
                       peak_idx_array[2:],
                       label_array[1:-1])

        segmented_beats = []
        for previous_idx, idx, next_idx, label in iterable:
            diff_1 = (idx - previous_idx) // 2
            diff_2 = (next_idx - idx) // 2

            if label in ignore:
                continue

            segmented_beats.append((self.__signals[idx - diff_1: idx + diff_2], label))
        return segmented_beats

    def extract_wavelets(self, beats):
        pass

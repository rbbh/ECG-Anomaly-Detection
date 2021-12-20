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
    """ This class implements a series of methods with the goal to pre-process the ECG signals, such as extracting
    the single beats, features and also split and normalize the extracted data.

    Attributes
    __________
    _Preprocess.__signals : numpy-array
                            Raw ECG signals.

    _Preprocess.__annotations : numpy-array
                                ECG beat labels

    _Preprocess.__peaks : numpy-array
                          ECG peak locations.

    _Preprocess.__split_pct : float
                              Split percentage between train and validation data.

    _Preprocess.__wavelet_name: str
                                Continuous Wavelet Transform type of wavelet that will extract scalograms.

    _Preprocess.__channel : int
                            Channel id of the electrode used to extract the ECG signals.

    _Preprocess.__mit_dir : str
                            MIT-BIH database extract directory.

    _Preprocess.__pickle_path: str
                               Path where the features are saved in the pickle format.

    _Preprocess.__beats : tuple
                          ECG extracted single beats.

    _Preprocess.__mean_len : float
                             Mean length of the extracted ECG beats.

    _Preprocess.__normal_beats : numpy-array
                                 Normal ECG beats.

    _Preprocess.__abnormal_beats : numpy-array
                                   Abnormal ECG beats.

    _Preprocess.__len_split : float
                              Split size that will be used to balance the normal and abnormal classes on the test data.

    _Preprocess.__normal_scalograms : numpy-array
                                      Normal extracted scalograms.

    _Preprocess.__abnormal_scalograms : numpy-array
                                        Abnormal extracted scalograms.

    Methods
    -------
    _Preprocess.__segment : Private
                            Segments raw ECG signals into single beats.

    _Preprocess.__extract_scalograms : Private
                                       Extracts features of the ECG beats in the form of scalograms.

    _Preprocess.normalize : Public
                            Applies a min-max normalization on the data.

    _Preprocess.shuffle_and_split_dataset : Public
                                            Shuffles and splits the data.

    _Preprocess.join_datasets : Public
                                Concatenates datasets.

    _Preprocess.to_torch_ds : Public
                              Converts dataset to torch dataset format.

    """
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
        """Segments ECG signals into single beats.

        Parameters
        ----------
        signals : numpy-array
                  ECG time signals.
        peaks : numpy-array
                Indexes of the ECG peaks.
        annotations : numpy-array
                      Labels of each beat.

        Returns
        -------
        normal_beats : numpy-array
                       Segmented normal beats.
        abnormal_beats : numpy-array
                         Segmented anomaly beats.
        mean_len : float
                   Average amount of beat samples.

        """

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
        """Extracts signal scalograms using the Continuous Wavelet Transform (CWT).

        Parameters
        ----------
        beats : numpy-array
                ECG single-beat time signals.
        wavelet_name : str
                       Type of CWT to be applied.
        wavelet_y_axis : float
                         Y-axis size of the scalograms.

        Returns
        -------
        normal_beats : numpy-array
                       ECG beats that are not considered anomalies.
        abnormal_beats : numpy-array
                         ECG arrhythmia beats (anomalies).

        """

        normal_beats, abnormal_beats = beats

        def __load_pickle(path):
            """Loads features from pickle file.

            Parameters
            ----------
            path : str
                   Path to load pickle file from.

            Returns
            -------
            var : numpy-array
                  Loaded features.

            """
            with open(path, "rb") as f:
                var = pickle.load(f)
            return var

        def __save_pickle(var, name):
            """

            Parameters
            ----------
            var : numpy-array
                  Features.
            name : str
                   Name from the pickle file to be saved.

            """
            base_path = Path("dataset/scalograms")
            if not base_path.exists():
                base_path.mkdir(parents=True, exist_ok=True)
            with open(base_path / f"{name}.pkl", "wb") as f:
                pickle.dump(var, f)

        def __resize(feature):
            """Resizes matrix of features extracted from the beats.

            Parameters
            ----------
            feature : numpy-array
                      Features extracted from the signals.

            Returns
            -------
            resized_features : numpy-array
                               Resized features.

            """
            return cv2.resize(feature, (64, 64))

        def __compute_wavelets(beats):
            """Extracts scalogram matrices from the signals.

            Parameters
            ----------
            beats : numpy-array
                    Single-beat ECG time signals.

            Returns
            -------
            normal_scalograms : numpy-array
                                Normal scalogram matrices.
            abnormal_scalograms : numpy-array
                                  Arrhythmia scalogram matrices (anomalies).

            """
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

            normal_scalograms = np.array([(feature, 1, label) for (feature, label) in normal_scalograms], dtype=object)
            abnormal_scalograms = np.array([(feature, -1, label) for (feature, label) in abnormal_scalograms],
                                           dtype=object)

            scalograms = np.concatenate((normal_scalograms, abnormal_scalograms), axis=0)
            __save_pickle(scalograms, pickle_name)

        else:
            scalograms = __load_pickle(Path(self.__pickle_path).joinpath(pickle_name).with_suffix(".pkl"))
            normal_idxs = np.where((scalograms[:, 2] == 'N') | (scalograms[:, 2] == 'L') | (scalograms[:, 2] == 'R'))
            abnormal_idxs = np.where((scalograms[:, 2] != 'N') & (scalograms[:, 2] != 'L') & (scalograms[:, 2] != 'R'))

            normal_scalograms = scalograms[normal_idxs]
            abnormal_scalograms = scalograms[abnormal_idxs]

        return normal_scalograms, abnormal_scalograms

    @staticmethod
    def normalize(features):
        """Makes a Min-Max normalization on the matrices of features.

        Parameters
        ----------
        features : numpy-array
                   Matrices of features.

        Returns
        -------
        normalized_features : numpy-array
                              Normalized matrices of features.

        """
        normalized_features = []
        for feature, label_num, label in features:
            new_feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
            normalized_features.append((new_feature, label_num, label))
        return np.array(normalized_features, dtype=object)

    def shuffle_and_split_dataset(self, dataset):
        """Shuffles and splits the dataset.

        Parameters
        ----------
        dataset : numpy-array
                  Dataset to be shuffled and splitted.

        Returns
        -------
        train_data : numpy-array
                     Train slice of the dataset.
        val_data : numpy-array
                   Validation slice of the dataset.
        test_data : numpy-array
                    Test slice of the dataset.

        """
        idxs = np.arange(len(dataset))
        np.random.shuffle(idxs)

        train_data = dataset[idxs][:int(self.__len_split * self.__split_pct)]
        val_data = dataset[idxs][int(self.__len_split * self.__split_pct): self.__len_split]
        test_data = dataset[idxs][self.__len_split:]

        return train_data, val_data, test_data

    @staticmethod
    def join_datasets(ds_1, ds_2):
        """Concatenates datasets.

        Parameters
        ----------
        ds_1 : numpy-array
               First dataset.
        ds_2 : numpy-array
               Second dataset.

        Returns
        -------
        joined_ds : numpy-array
                    Concatenated dataset.

        """
        joined_ds = np.concatenate((ds_1, ds_2), axis=0)
        np.random.shuffle(joined_ds)
        return joined_ds

    @staticmethod
    def to_torch_ds(dataset, batch_size=None, test=False):
        """Creates torch datasets out of numpy arrays.

        Parameters
        ----------
        dataset : numpy-array
                  Dataset to be converted.
        batch_size : int
                     Mini-batch length of the final dataset.
        test : bool
               Indicated whether it is the test slice of the dataset or not.

        Returns
        -------
        torch_ds : torch-tensor
                   Dataset converted to torch tensor.

        """
        X_data = dataset[:, 0]
        y_data = dataset[:, 1]

        X_data = np.array([arr.astype('float64') for arr in X_data])
        y_data = y_data.astype('int32')

        X_data_torch_ds = torch.from_numpy(X_data).unsqueeze(dim=1)
        y_data_torch_ds = torch.from_numpy(y_data)

        torch_ds = [(X, y) for (X, y) in zip(X_data_torch_ds, y_data_torch_ds)]

        if not test:
            torch_ds = DataLoader(
                torch_ds,
                batch_size=batch_size,
                shuffle=True
            )
        else:
            torch_ds = [(X.unsqueeze(dim=0), y) for (X, y) in torch_ds]

        return torch_ds

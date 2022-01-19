import cv2
import pywt
import random
import pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

import torch

normal = ['N', 'L', 'R']
ignore = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', "'", '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@']


class Preprocess:
    """This class implements a series of methods with the goal to pre-process the ECG signals, such as extracting the
    single beats, features and also split and normalize the extracted data.

    Attributes
    ----------
    _Preprocess.__signals : numpy-array
                            Raw ECG signals.

    _Preprocess.__annotations : numpy-array
                                ECG beat labels.

    _Preprocess.__peaks : numpy-array
                          ECG peak locations.

    _Preprocess.__feature_type : str
                                 Type of feature used.

    _Preprocess.__sample_amount : int
                                  Desired number of samples in each ECG signal.

    _Preprocess.__window_len : int
                               Fourier frame analysis size.

    _Preprocess.__n_fft : int
                          Fourier window resolution.

    _Preprocess.__step_len : int
                             Fourier frame step.

    _Preprocess.__window_type : str
                                Type of window used on the Fourier frame analysis.

    _Preprocess.__wavelet_name: str
                                Continuous Wavelet Transform wavelet name that will extract scalograms.

    _Preprocess.__split_pct : float
                              Split percentage between train and validation data.

    _Preprocess.__pickle_name: str
                               Path where the features are saved in the pickle format.

    _Preprocess.__beats : tuple
                          ECG extracted single beats.

    _Preprocess.__normal_beats : numpy-array
                                 Normal ECG beats.

    _Preprocess.__abnormal_beats : numpy-array
                                   Abnormal ECG beats.

    _Preprocess.__len_split : float
                              Split size that will be used to balance the normal and abnormal classes on the test data.

    _Preprocess.__normal_2d_features : numpy-array
                                       Normal extracted features.

    _Preprocess.__abnormal_2d_features : numpy-array
                                         Abnormal extracted features.

    Methods
    -------
    _Preprocess.__segment : Private
                            Segments raw ECG signals into single beats.

    _Preprocess.__extract_features : Private
                                     Extracts 2d features of the ECG beats.

    _Preprocess.__compute_scalograms : Private
                                       Computes scalogram matrices.

    _Preprocess.__compute_spectrograms : Private
                                         Computes spectrogram matrices.

    _Preprocess.__plot_feature : Private
                                 Displays the feature matrices chosen at random and saves the image.

    _Preprocess.__load_pickle : Private
                                Loads features from pickle file.

    _Preprocess.__save_pickle : Private
                                Saves features to pickle file.

    _Preprocess.__resize : Private
                           Resizes matrices of features.

    _Preprocess.normalize : Public
                            Applies a min-max normalization on the data.

    _Preprocess.shuffle_and_split_dataset : Public
                                            Shuffles and splits the data.

    _Preprocess.join_datasets : Public
                                Concatenates datasets.

    _Preprocess.to_torch_ds : Public
                              Converts dataset to torch dataset format.

    """

    def __init__(self, signals, annotations, peaks, split_pct, feature_type, pickle_name, samples, window_len=None,
                 window_type=None, step_len=None, wavelet_name=None):

        self.__signals = signals
        self.__annotations = annotations
        self.__peaks = peaks
        self.__feature_type = feature_type

        self.__sample_amount = samples
        self.__window_len = window_len
        self.__nfft = self.__window_len if np.ceil(np.log2(self.__window_len)) == np.floor(
            np.log2(self.__window_len)) else 128
        self.__step_len = step_len
        self.__window_type = window_type
        self.__wavelet_name = wavelet_name

        self.__split_pct = split_pct
        self.__pickle_name = pickle_name
        self.__beats = self.__segment(self.__signals, self.__peaks, self.__annotations)

        self.__normal_beats = self.__beats[0]
        self.__abnormal_beats = self.__beats[1]
        self.__len_split = len(self.__normal_beats) - len(self.__abnormal_beats)

        self.__normal_2d_features, self.__abnormal_2d_features = self.__extract_features(self.__beats,
                                                                                         self.__feature_type,
                                                                                         self.__pickle_name)

    @property
    def get_feature_type(self):
        return self.__feature_type

    @property
    def get_normal_beats(self):
        return self.__normal_beats

    @property
    def get_abnormal_beats(self):
        return self.__abnormal_beats

    @property
    def get_normal_2d_features(self):
        return self.__normal_2d_features

    @property
    def get_abnormal_2d_features(self):
        return self.__abnormal_2d_features

    def __segment(self, signals, peaks, annotations):
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

        """

        step = self.__sample_amount // 2
        iterable = zip(peaks[:-2],
                       peaks[1:-1],
                       peaks[2:],
                       annotations[1:-1])

        normal_beats = []
        abnormal_beats = []
        signals_len = []
        for previous_idx, idx, next_idx, label in iterable:

            if label in ignore:
                continue

            if (step >= idx - previous_idx) or (step >= next_idx - idx):
                continue

            beat = (signals[idx - step: idx + step], label)
            if label in normal:
                normal_beats.append(beat)
            else:
                abnormal_beats.append(beat)
            signals_len.append(len(beat[0]))

        return normal_beats, abnormal_beats

    def __extract_features(self, beats, feature_type, pickle_name):
        """Extracts signal 2D features using either the Short-time Fourier Transform (STFT) to extract spectrograms or
        the Continuous Wavelet Transform (CWT) to extract scalograms.

        Parameters
        ----------
        beats : numpy-array
                ECG single-beat time signals.

        feature_type : str
                       Type of feature used.

        pickle_name : str
                      Path where the features are saved in the pickle format.

        Returns
        -------
        normal_beats : numpy-array
                       ECG beats that are not considered anomalies.
        abnormal_beats : numpy-array
                         ECG arrhythmia beats (anomalies).

        """

        normal_beats, abnormal_beats = beats

        if not Path(f"dataset/{feature_type}").joinpath(pickle_name).with_suffix(".pkl").exists():

            if feature_type == 'scalograms':
                normal_features = self.__compute_scalograms(normal_beats)
                abnormal_features = self.__compute_scalograms(abnormal_beats)

            elif feature_type == 'spectrograms':
                normal_features = self.__compute_spectrograms(normal_beats)
                abnormal_features = self.__compute_spectrograms(abnormal_beats)

            normal_features = [(self.__resize(feature), label) for feature, label in normal_features if
                               feature.shape != (64, 64)]
            abnormal_features = [(self.__resize(feature), label) for feature, label in abnormal_features if
                                 feature.shape != (64, 64)]

            normal_features = np.array([(feature, 1, label) for (feature, label) in normal_features], dtype=object)
            abnormal_features = np.array([(feature, -1, label) for (feature, label) in abnormal_features],
                                         dtype=object)

            self.__plot_feature(normal_features)
            self.__plot_feature(abnormal_features)

            features = np.concatenate((normal_features, abnormal_features), axis=0)
            self.__save_pickle(features, pickle_name)

        else:
            features = self.__load_pickle(Path(f"dataset/{feature_type}").joinpath(pickle_name).with_suffix(".pkl"))
            normal_idxs = np.where((features[:, 2] == 'N') | (features[:, 2] == 'L') | (features[:, 2] == 'R'))
            abnormal_idxs = np.where((features[:, 2] != 'N') & (features[:, 2] != 'L') & (features[:, 2] != 'R'))

            normal_features = features[normal_idxs]
            abnormal_features = features[abnormal_idxs]

            self.__plot_feature(normal_features)
            self.__plot_feature(abnormal_features)

        return normal_features, abnormal_features

    def __compute_scalograms(self, beats):
        """Extracts scalogram matrices from the signals by using the Continuous Wavelet Transform (CWT).

        Parameters
        ----------
        beats : numpy-array
                Single-beat ECG time signals.

        Returns
        -------
        scalograms : list
                     Scalogram matrices.

        """
        scalograms = []
        for beat, label in beats:
            scalogram, _ = pywt.cwt(beat, np.arange(1, self.__sample_amount + 1), self.__wavelet_name)
            scalograms.append((scalogram, label))
        return scalograms

    def __compute_spectrograms(self, beats):
        """Extracts spectrogram matrices from the signals by using the Short-time Fourier Transform (STFT).

        Parameters
        ----------
        beats : numpy-array
                Single-beat ECG time signals.

        Returns
        -------
        spectrograms : list
                       Spectrogram matrices.

        """

        spectrograms = []
        for beat, label in beats:
            spectrogram = librosa.stft(beat,
                                       n_fft=self.__nfft,
                                       win_length=self.__window_len,
                                       hop_length=self.__step_len,
                                       window=self.__window_type)

            power_spectrogram = np.abs(spectrogram) ** 2
            power_spectrogram[power_spectrogram == 0.] = 1e-6

            log_spectrogram = np.log10(power_spectrogram)
            spectrograms.append((log_spectrogram, label))

        return spectrograms

    def __plot_feature(self, features):
        """Displays both a normal and an abnormal feature matrix chosen at random and saves the image.

        Parameters
        ----------
        features : numpy-array
                   2 dimensional features.

        """
        chosen = random.choice(features)
        chosen_feature = chosen[0]
        label_id = chosen[1]
        chosen_label = chosen[2]

        base_path = Path(f"outputs/plots/{self.__feature_type}")
        if not base_path.exists():
            base_path.mkdir(parents=True, exist_ok=True)
        if label_id == 1:
            save_name = "Normal Feature Sample"
        else:
            save_name = "Abnormal Feature Sample"

        if self.__feature_type == "spectrograms":

            fig, ax = plt.subplots()
            img = librosa.display.specshow(chosen_feature, sr=360, hop_length=self.__step_len, x_axis='s',
                                           y_axis='linear', cmap='inferno', ax=ax)
            ax.set(title=f"Label: {chosen_label}")
            fig.colorbar(img, ax=ax, format="%+2.f log-Hz²")
        else:
            plt.imshow(chosen_feature, aspect="auto", origin="lower", cmap="inferno")
            plt.title(f"Label: {chosen_label}")

        plt.savefig((base_path / save_name).with_suffix(".png"))
        plt.show()

    @staticmethod
    def __load_pickle(path):
        """Loads features from pickle file.

        Parameters
        ----------
        path : Path object
               Path to load pickle file from.

        Returns
        -------
        var : numpy-array
              Loaded features.

        """
        with open(path, "rb") as f:
            var = pickle.load(f)
        return var

    def __save_pickle(self, var, name):
        """Saves features to pickle file.

        Parameters
        ----------
        var : numpy-array
              Features.
        name : str
               Name from the pickle file to be saved.

        """
        base_path = Path(f"dataset/{self.__feature_type}")
        if not base_path.exists():
            base_path.mkdir(parents=True, exist_ok=True)
        with open(base_path / f"{name}.pkl", "wb") as f:
            pickle.dump(var, f)

    @staticmethod
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

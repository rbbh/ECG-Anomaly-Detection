import wfdb
import numpy as np
import pandas as pd
from pathlib import Path


class MITBIH:
    def __init__(self, dir, channel):
        self.__dir = Path(dir)
        self.__channel = channel
        self.__base_path = Path('dataset/signals')
        self.signals = self.__load_signals(self.__dir, self.__channel)
        self.annotations, self.peaks = self.__load_annotations(self.__dir)

    @property
    def get_signals(self):
        return self.signals

    @property
    def get_annotations(self):
        return self.annotations

    @property
    def get_peak_locations(self):
        return self.peaks

    def __load_signals(self, dir, channel):
        """Loads ECG time signals.

        Parameters
        ----------
        dir : str
              MIT-BIH signal directory.
        channel : int
                  Channel ID from where the signals will be extracted.


        Returns
        -------
        signals : numpy-array
                  Extracted signals.

        """
        csv_path = self.__base_path / self.__dir.with_suffix('.csv')
        if csv_path.exists():
            signals_df = pd.read_csv(csv_path)
            signals = signals_df[f"channel_{channel}"]
            signals = np.array(signals)
        else:
            signals, _ = wfdb.rdsamp(str(dir), pn_dir='mitdb')
            signals_df = pd.DataFrame(signals, columns=["channel_0", "channel_1"])
            signals_df.to_csv(csv_path, index=False)
            signals = signals[:, channel]
        return signals

    def __load_annotations(self, dir):
        """Loads ECG peak locations and beat labels.

        Parameters
        ----------
        dir : str
              MIT-BIH signal directory.

        Returns
        -------
        annotations : numpy-array
                      Labels of each beat.
        peaks : numpy-array
                Indexes of each signal peak.

        """
        annot_path = self.__base_path / self.__dir.with_suffix('.annot')
        if annot_path.exists():
            annotations_df = pd.read_csv(annot_path)
            annotations = np.array(annotations_df["annotations"])
            peaks = np.array(annotations_df["peak_idxs"])
        else:
            annotations_obj = wfdb.rdann(str(dir), 'atr', pn_dir='mitdb')
            annotations = annotations_obj.symbol
            annotations = np.array([[value] for value in annotations])

            peaks = annotations_obj.sample
            peaks = np.array([[value] for value in peaks])
            concat_arrays = np.concatenate([annotations, peaks], axis=1)

            annotations_df = pd.DataFrame(concat_arrays, columns=["annotations", "peak_idxs"])
            annotations_df.to_csv(annot_path, index=False)
        return annotations, peaks

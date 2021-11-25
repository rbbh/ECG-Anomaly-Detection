import wfdb


class MITBIH:
    def __init__(self, dir, channel):
        self.__dir = dir
        self.__channel = channel
        self.signals = self.__load_signals(self.__dir, self.__channel)
        self.annotations = self.__load_annotations(self.__dir)

    @staticmethod
    def __load_signals(dir, channel):
        signals, _ = wfdb.rdsamp(dir, pn_dir='mitdb')
        return signals[:, channel]

    @staticmethod
    def __load_annotations(dir):
        annotations = wfdb.rdann(dir, 'atr', pn_dir='mitdb')
        return annotations

    @property
    def get_signals(self):
        return self.signals

    @property
    def get_annotations(self):
        return self.annotations

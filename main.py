import argparse
import configparser

from dataset.mit_bih import MITBIH
from source.preprocess import Preprocess


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="-c inputs/configs/default.conf", help="Config input file.")

    args = arg_parser.parse_args()
    config = args.config

    config_parser = configparser.ConfigParser()
    config_parser.read(config)

    dir = config_parser["SIGNALS"]["signal_dir"]
    channel = int(config_parser["SIGNALS"]["channel"])
    wavelet_name = config_parser["PREPROCESS"]["wavelet"]

    dataset = MITBIH(dir, channel)
    signals = dataset.get_signals
    annotations = dataset.get_annotations

    preprocess_obj = Preprocess(signals,
                                annotations,
                                wavelet_name)

    beats = preprocess_obj.segment()
    wavelets = preprocess_obj.extract_wavelets(beats)


if __name__ == "__main__":
    main()

import argparse
import configparser

from dataset.mit_bih import MITBIH
from source.preprocess import Preprocess


def get_config_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="inputs/configs/default.conf", help="Config input file.")
    args = arg_parser.parse_args()

    config = args.config
    config_parser = configparser.ConfigParser()
    config_parser.read(config)

    return config_parser


def run(parser):

    dir = parser["SIGNALS"]["signal_dir"]
    channel = int(parser["SIGNALS"]["channel"])
    wavelet_name = parser["PREPROCESS"]["wavelet"]

    dataset = MITBIH(dir, channel)
    signals = dataset.get_signals
    annotations = dataset.get_annotations

    preprocess_obj = Preprocess(signals,
                                annotations,
                                wavelet_name)
    beats = preprocess_obj.get_normal_beats
    scalograms = preprocess_obj.get_normal_scalograms


if __name__ == "__main__":
    parser = get_config_parser()
    run(parser)

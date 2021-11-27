import argparse
import configparser

import torch

from dataset.mit_bih import MITBIH
from source.preprocess import Preprocess
from models.autoencoder import AutoEncoder
from source.train import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_config_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="inputs/configs/default.conf", help="Config input file.")
    args = arg_parser.parse_args()

    config = args.config
    config_parser = configparser.ConfigParser()
    config_parser.read(config)

    return config_parser


def run(parser):
    mit_dir = parser["SIGNALS"]["signal_dir"]
    channel = int(parser["SIGNALS"]["channel"])
    wavelet_name = parser["PREPROCESS"]["wavelet"]

    epochs = int(parser['MODEL']['epochs'])
    batch_size = int(parser['MODEL']['batch_size'])
    learning_rate = float(parser['MODEL']['learning_rate'])

    dataset = MITBIH(mit_dir, channel)
    signals = dataset.get_signals
    annotations = dataset.get_annotations

    preprocess_obj = Preprocess(signals,
                                annotations,
                                wavelet_name)
    normal_beats_len = preprocess_obj.get_normal_beats_len
    abnormal_beats_len = preprocess_obj.get_abnormal_beats_len

    normal_scalograms = preprocess_obj.get_normal_scalograms

    model = AutoEncoder(in_channels=1, dense_neurons=32)
    train_obj = Trainer(normal_scalograms,
                        normal_beats_len,
                        abnormal_beats_len,
                        epochs,
                        batch_size,
                        learning_rate,
                        device)


if __name__ == "__main__":
    parser = get_config_parser()
    run(parser)

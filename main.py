import torch
import argparse
import configparser

from source.train import Trainer
from dataset.mit_bih import MITBIH
from source.preprocess import Preprocess
from models.autoencoder import AutoEncoder

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
    mit_dir = parser["SIGNALS"]["mit_dir"]
    channel = int(parser["SIGNALS"]["channel"])

    val_split_pct = float(parser["PREPROCESS"]["train_val_split_pct"])
    wavelet_name = parser["PREPROCESS"]["wavelet"]
    pickle_path = parser["PREPROCESS"]["pickle_path"]

    epochs = int(parser['DL-MODEL']['epochs'])
    batch_size = int(parser['DL-MODEL']['batch_size'])
    learning_rate = float(parser['DL-MODEL']['learning_rate'])

    dataset = MITBIH(mit_dir, channel)
    signals = dataset.get_signals
    annotations = dataset.get_annotations
    peaks = dataset.get_peak_locations

    preprocess_obj = Preprocess(signals, annotations, peaks, val_split_pct, wavelet_name, mit_dir, pickle_path)
    normal_scalograms = preprocess_obj.get_normal_scalograms

    train_data, val_data, test_data = preprocess_obj.shuffle_and_split_dataset(normal_scalograms)

    model = AutoEncoder(in_channels=1, dense_neurons=32)
    train_obj = Trainer(normal_scalograms,
                        epochs,
                        batch_size,
                        learning_rate,
                        device)


if __name__ == "__main__":
    parser = get_config_parser()
    run(parser)

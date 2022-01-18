import torch
from torchsummary import summary

import argparse
import configparser
from pathlib import Path

from source.test import Tester
from source.train import Trainer
from dataset.mit_bih import MITBIH
from source.preprocess import Preprocess
from source.autoencoder import AutoEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_config_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="inputs/configs/default.conf", help="Config input file.")
    args = arg_parser.parse_args()

    config = args.config
    config_parser = configparser.ConfigParser()
    config_parser.read(config)

    return config_parser


def preprocess_pipeline(parser, dataset_obj):

    feature_type = parser["PREPROCESS"]["feature_type"]
    pickle_name = parser["PREPROCESS"]["pickle_name"]

    samples = int(parser["PREPROCESS"]["signal_sample_amount"])
    window_len = int(parser["PREPROCESS"]["window_len"])
    step_len = int(parser["PREPROCESS"]["step_len"])

    window_type = parser["PREPROCESS"]["window"]
    wavelet_name = parser["PREPROCESS"]["wavelet"]

    val_split_pct = float(parser["PREPROCESS"]["train_val_split_pct"])
    batch_size = int(parser['DL-MODEL']['batch_size'])
    val_batch_size = 10

    signals = dataset_obj.get_signals
    annotations = dataset_obj.get_annotations
    peaks = dataset_obj.get_peak_locations

    preprocess_obj = Preprocess(signals, annotations, peaks, val_split_pct, feature_type, pickle_name, samples,
                                window_len, window_type, step_len, wavelet_name)
    normal_features = preprocess_obj.get_normal_2d_features
    abnormal_features = preprocess_obj.get_abnormal_2d_features

    normal_features = preprocess_obj.normalize(normal_features)
    abnormal_features = preprocess_obj.normalize(abnormal_features)

    train_data, val_data, normal_test_data = preprocess_obj.shuffle_and_split_dataset(normal_features)
    test_data = preprocess_obj.join_datasets(normal_test_data, abnormal_features)

    train_torch_ds = preprocess_obj.to_torch_ds(train_data, batch_size)
    val_torch_ds = preprocess_obj.to_torch_ds(val_data, val_batch_size)
    test_torch_ds = preprocess_obj.to_torch_ds(test_data, test=True)

    return train_torch_ds, val_torch_ds, test_torch_ds, preprocess_obj


def run(parser):
    mit_dir = parser["SIGNALS"]["mit_dir"]
    channel = int(parser["SIGNALS"]["channel"])
    epochs = int(parser['DL-MODEL']['epochs'])
    learning_rate = float(parser['DL-MODEL']['learning_rate'])

    dense_neurons = int(parser['DL-MODEL']['dense_neurons'])
    checkpoint_pct = float(parser['DL-MODEL']['checkpoint_pct'])
    weights_path = parser['DL-MODEL']['weights_path']
    ml_model = parser['ML-MODEL']['model']

    dataset_obj = MITBIH(mit_dir, channel)
    train_loader, val_loader, test_loader, preprocess_obj = preprocess_pipeline(parser, dataset_obj)

    model = AutoEncoder(in_channels=1, dense_neurons=dense_neurons).to(device)
    input_shape = train_loader.dataset[0][0].shape
    summary(model, input_shape)

    if not weights_path:
        train_obj = Trainer(model, preprocess_obj, train_loader, val_loader, epochs, learning_rate, checkpoint_pct, device)
        final_model_path = train_obj.train_autoencoder()
        weights_path = Path(final_model_path)

    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_obj = Tester(test_loader, ml_model, preprocess_obj, device)
    test_obj.evaluate(model)


if __name__ == "__main__":
    parser = get_config_parser()
    run(parser)

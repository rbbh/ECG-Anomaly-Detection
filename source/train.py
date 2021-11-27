import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, normal_scalograms, normal_beats_len, abnormal_beats_len, epochs, batch_size, learning_rate,
                 device='cpu'):
        self.__normal_scalograms = normal_scalograms
        self.__normal_beats_len = normal_beats_len
        self.__abnormal_beats_len = abnormal_beats_len

        self.__division_len = self.__normal_beats_len - self.__abnormal_beats_len
        self.__features = np.array(self.__normal_scalograms, dtype=object)[:, 0]
        self.__labels = np.array(self.__normal_scalograms, dtype=object)[:, 1]

        self.__train_size = int(self.__division_len * 0.9)
        self.__val_size = int(self.__division_len * 0.1)
        self.__train_data = torch.from_numpy(self.__features[:self.__train_size])
        self.__val_data = torch.from_numpy(self.__features[self.__train_size:self.__val_size])

        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__lr = learning_rate

        self.__train_loader = DataLoader(self.__train_data,
                                         batch_size=self.__batch_size,
                                         shuffle=True)
        self.__val_loader = DataLoader(self.__val_data,
                                         batch_size=self.__batch_size,
                                         shuffle=True)

        self.__device = device

        if self.__device.type == 'cuda':
            print('__CUDNN VERSION:', torch.backends.cudnn.version())
            print('__Number CUDA Devices:', torch.cuda.device_count())
            print('__CUDA Device Name:', torch.cuda.get_device_name(0))
            print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)
        else:
            print('No GPU available. Training with CPU.')

    def train_autoencoder(self, model):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.__lr)

        avg_train_losses = []
        avg_val_losses = []
        for epoch in tqdm(range(self.__epochs), desc='Training'):
            train_losses = []
            model.train()
            for in_feature in self.__train_loader:
                in_feature = in_feature.float().to(self.__device)

                # Forward
                out_feature = model(in_feature)
                out_feature.float().to(self.__device)

                loss = criterion(in_feature, out_feature)
                train_losses.append(loss)

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Optimizer Step
                optimizer.step()

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_train_losses.append(avg_train_loss)

            val_losses = []
            model.eval()
            for in_feature in self.__val_loader:
                in_feature = in_feature.float().to(self.__device)

                # Forward
                out_feature = model(in_feature)
                out_feature.float().to(self.__device)

                loss = criterion(in_feature, out_feature)
                val_losses.append(loss)

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_losses.append(avg_val_loss)

            print(f'Train loss at epoch {epoch + 1} : {avg_train_loss}')
            print(f'Val loss at epoch {epoch + 1} : {avg_val_loss}')
            print()

        return avg_train_losses, avg_val_losses

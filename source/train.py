from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, train_loader, val_loader, epochs, learning_rate, device='cpu'):
        self.__model = model
        self.__train_loader = train_loader
        self.__val_loader = val_loader
        self.__epochs = epochs
        self.__lr = learning_rate
        self.__device = device

        if self.__device.type == 'cuda':
            print('Trainable GPU Device(s) Detected!')
            print('CUDNN VERSION:', torch.backends.cudnn.version())
            print('Number of CUDA Devices:', torch.cuda.device_count())
            print('CUDA Device Name:', torch.cuda.get_device_name(0))
            print('CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)
        else:
            print('No GPU available. Training with CPU.')

    def train_autoencoder(self):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.__model.parameters(), lr=self.__lr)

        avg_train_losses = []
        avg_val_losses = []
        for epoch in tqdm(range(self.__epochs), desc='Training'):
            train_losses = []
            self.__model.train()
            for in_feature in self.__train_loader:
                in_feature = in_feature.float().to(self.__device)

                # Forward
                out_feature = self.__model(in_feature)
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
            self.__model.eval()
            for in_feature in self.__val_loader:
                in_feature = in_feature.float().to(self.__device)

                # Forward
                out_feature = self.__model(in_feature)
                out_feature.float().to(self.__device)

                loss = criterion(in_feature, out_feature)
                val_losses.append(loss)

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_losses.append(avg_val_loss)

            print()
            print(f'Train loss at epoch {epoch + 1} : {avg_train_loss}')
            print(f'Val loss at epoch {epoch + 1} : {avg_val_loss}')
            print()

        return avg_train_losses, avg_val_losses
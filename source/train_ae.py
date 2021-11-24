from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, epochs, learning_rate, device='cpu'):
        self.__model = model
        self.__epochs = epochs
        self.__lr = learning_rate
        self.__device = device

    def train(self):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.__model.parameters(), lr=self.__lr)

        avg_train_losses = []
        avg_val_losses = []
        for epoch in tqdm(range(self.__epochs), desc='Training'):
            train_losses = []
            self.__model.train()
            for in_feature in train_loader:
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
            for in_feature in val_loader:
                in_feature = in_feature.float().to(self.__device)

                # Forward
                out_feature = self.__model(in_feature)
                out_feature.float().to(self.__device)

                loss_ = criterion(in_feature, out_feature)
                val_losses.append(loss_)

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_losses.append(avg_val_loss)

            print(f'Train loss at epoch {epoch + 1} : {avg_train_loss}')
            print(f'Val loss at epoch {epoch + 1} : {avg_val_loss}')
            print()

        return avg_train_losses, avg_val_losses
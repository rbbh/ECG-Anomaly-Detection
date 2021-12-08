import sys
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, train_loader, val_loader, epochs, learning_rate, checkpoint_pct, device='cpu'):
        self.__model = model
        self.__train_loader = train_loader
        self.__val_loader = val_loader
        self.__epochs = epochs
        self.__lr = learning_rate
        self.__checkpoint_pct = checkpoint_pct
        self.__device = device

    def train_autoencoder(self):
        if self.__device.type == 'cuda':
            print('Trainable GPU Device(s) Detected!')
            print('CUDNN VERSION:', torch.backends.cudnn.version())
            print('Number of CUDA Devices:', torch.cuda.device_count())
            print('CUDA Device Name:', torch.cuda.get_device_name(0))
            print('CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)
        else:
            print('No GPU available. Training with CPU.')

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.__model.parameters(), lr=self.__lr)
        model_base_path = Path("models") / Path("saved_models") / \
                          Path(f"{self.__model.get_dense_neurons}_dense_neurons_{self.__epochs}_epochs")
        curr_best_loss = sys.maxsize
        avg_train_losses = []
        avg_val_losses = []
        interval = int(self.__checkpoint_pct * self.__epochs)
        checkpoints = [e for e in range(interval, self.__epochs + interval, interval)]

        def __save_model_checkpoint(model, epochs, val_loss):
            if not model_base_path.exists():
                model_base_path.mkdir(parents=True, exist_ok=True)
            curr_files = [file for file in model_base_path.glob("*.pt")]
            for current in curr_files:
                current.unlink()

            final_path = model_base_path / Path(f"torch_model_val_loss_{val_loss:.6f}.pt")
            torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, final_path)

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

            print(f'\nTrain loss at epoch {epoch + 1} : {avg_train_loss}')
            print(f'\nVal loss at epoch {epoch + 1} : {avg_val_loss}\n')

            if epoch + 1 in checkpoints:
                print("Evaluating Checkpoint\n")
                if avg_val_loss < curr_best_loss:
                    print("New Best Weights Found!\n")
                    print(f"Current best validation loss: {avg_val_loss:.6f}\n")
                    __save_model_checkpoint(self.__model, self.__epochs, avg_val_loss)
                    curr_best_loss = avg_val_loss
                else:
                    print("No best weights in this checkpoint.\n")

        return avg_train_losses, avg_val_losses

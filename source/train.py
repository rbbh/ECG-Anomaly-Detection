import sys
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from source.preprocess import Preprocess


class Trainer:
    """This class is responsible for the training of the DL-Model.

    Attributes
    ----------
    _Trainer.__model : object
                       DL-Model object.

    _Trainer.__feature_type : str
                              Type of feature used.

    _Trainer.__train_loader : object
                              Torch train Dataloader object.

    _Trainer.__val_loader : object
                            Torch validation Dataloader object.

    _Trainer.__epochs : int
                        Number of epochs to train the DL-Model.

    _Trainer.__lr : float
                    Learning rate used to train the DL-Model.

    _Trainer.__checkpoint_pct : float
                                Checkpoint interval in the epochs to verify whether to save a new model or not.

    _Trainer.__device : str
                        Device where the training will occur.

    Methods
    -------
    _Trainer.train_autoencoder : Public
                                 Method that implements the Auto-Encoder training.

    """

    def __init__(self, model, preprocess_obj: Preprocess, train_loader, val_loader, epochs, learning_rate,
                 checkpoint_pct, device='cpu'):

        self.__model = model
        self.__feature_type = preprocess_obj.get_feature_type
        self.__train_loader = train_loader
        self.__val_loader = val_loader

        self.__epochs = epochs
        self.__lr = learning_rate
        self.__checkpoint_pct = checkpoint_pct
        self.__device = device

    def train_autoencoder(self):
        """This method implements the Auto-Encoder training. Despite being public, the idea is that an external user
        does not worry about its operation and should only call it whenever they feel like training the Auto-Encoder
        from scratch.

        Returns
        -------
        final_model_path : str
                           Path from where to get the best saved model from the training.

        """
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
        model_base_path = f"inputs/models/saved_models/{self.__feature_type}/{self.__model.get_dense_neurons}_dense_neurons_{self.__epochs}_epochs"
        curr_best_loss = sys.maxsize
        avg_train_losses = []
        avg_val_losses = []
        interval = int(self.__checkpoint_pct * self.__epochs)
        checkpoints = [e for e in range(interval, self.__epochs + interval, interval)]

        def __save_model_checkpoint(model, epochs, val_loss):
            if not Path(model_base_path).exists():
                Path(model_base_path).mkdir(parents=True, exist_ok=True)
            curr_files = [file for file in Path(model_base_path).glob("*.pt")]
            for current in curr_files:
                current.unlink()

            final_path = Path(model_base_path) / f"torch_model_val_loss_{val_loss:.6f}.pt"
            torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, final_path)

            return final_path

        for epoch in tqdm(range(self.__epochs), desc='Training'):
            train_losses = []
            self.__model.train()
            for in_feature, _ in self.__train_loader:
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
            for in_feature, _ in self.__val_loader:
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
                    final_model_path = __save_model_checkpoint(self.__model, self.__epochs, avg_val_loss)
                    curr_best_loss = avg_val_loss
                else:
                    print("No best weights in this checkpoint.\n")

        return final_model_path

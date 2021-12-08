import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=1, dense_neurons=32):
        super(AutoEncoder, self).__init__()

        self.__dense_neurons = dense_neurons

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Dropout(0.2)
        )

        self.linear_1 = nn.Linear(8 * 8 * 64, dense_neurons)
        self.linear_2 = nn.Linear(dense_neurons, 8 * 8 * 64)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),

            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), padding="same"),
            nn.Sigmoid()
        )

    @property
    def get_dense_neurons(self):
        return self.__dense_neurons

    def encoder_forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear_1(x)

        return x

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = x.reshape(x.shape[0], 64, 8, 8)
        x = self.decoder(x)

        return x

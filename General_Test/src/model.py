import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self, input_size: int = 256, latent_dim: int = 64) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.latent_dim = int(latent_dim)

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_size, self.input_size)
            encoded = self.encoder_conv(dummy)

        self.feature_shape = tuple(encoded.shape[1:])
        self.flatten_dim = int(np.prod(self.feature_shape))

        self.encoder_fc = nn.Linear(self.flatten_dim, self.latent_dim)
        self.decoder_fc = nn.Linear(self.latent_dim, self.flatten_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder_conv(x)
        flattened = torch.flatten(features, start_dim=1)
        return self.encoder_fc(flattened)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.decoder_fc(z)
        features = features.view(z.size(0), *self.feature_shape)
        reconstruction = self.decoder_conv(features)
        if reconstruction.shape[-2:] != (self.input_size, self.input_size):
            reconstruction = F.interpolate(
                reconstruction,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent)

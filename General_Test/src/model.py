import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_encoder_conv() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
    )


def _infer_feature_shape(encoder_conv: nn.Module, input_size: int) -> tuple[tuple[int, ...], int]:
    with torch.no_grad():
        dummy = torch.zeros(1, 1, int(input_size), int(input_size))
        encoded = encoder_conv(dummy)
    feature_shape = tuple(encoded.shape[1:])
    flatten_dim = int(np.prod(feature_shape))
    return feature_shape, flatten_dim


class ConvEncoder(nn.Module):
    def __init__(self, input_size: int = 256, latent_dim: int = 64) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.latent_dim = int(latent_dim)

        self.encoder_conv = _build_encoder_conv()
        self.feature_shape, self.flatten_dim = _infer_feature_shape(self.encoder_conv, self.input_size)
        self.encoder_fc = nn.Linear(self.flatten_dim, self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder_conv(x)
        flattened = torch.flatten(features, start_dim=1)
        return self.encoder_fc(flattened)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(input_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(input_dim), int(projection_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionHead(nn.Module):
    def __init__(self, projection_dim: int) -> None:
        super().__init__()
        hidden_dim = max(32, int(projection_dim))
        self.net = nn.Sequential(
            nn.Linear(int(projection_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, int(projection_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        input_size: int = 256,
        latent_dim: int = 128,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.latent_dim = int(latent_dim)
        self.projection_dim = int(projection_dim)

        self.encoder = ConvEncoder(input_size=self.input_size, latent_dim=self.latent_dim)
        self.projection_head = ProjectionHead(self.latent_dim, self.projection_dim)
        self.predictor = PredictionHead(self.projection_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project(self, latent: torch.Tensor) -> torch.Tensor:
        return self.projection_head(latent)

    def predict(self, projection: torch.Tensor) -> torch.Tensor:
        return self.predictor(projection)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        projection = self.project(latent)
        return latent, projection


class ConvAutoencoder(nn.Module):
    def __init__(self, input_size: int = 256, latent_dim: int = 64) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.latent_dim = int(latent_dim)

        self.encoder_conv = _build_encoder_conv()
        self.feature_shape, self.flatten_dim = _infer_feature_shape(self.encoder_conv, self.input_size)

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

"""
Модуль с архитектурами моделей для 3D Autoencoder
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionModule3D(nn.Module):
    """Модуль пространственного внимания для 3D данных"""
    
    def __init__(self, kernel_size=3):
        super(SpatialAttentionModule3D, self).__init__()
        padding = kernel_size // 2
        self.conv3d = nn.Conv3d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        std_out = torch.std(x, dim=1, keepdim=True)
        
        x_combined = torch.cat([avg_out, max_out, std_out], dim=1) 
        spatial_attention_map = self.conv3d(x_combined)

        return x * self.sigmoid(spatial_attention_map)


class Conv3DAutoencoder(nn.Module):
    """
    3D сверточный автоэнкодер с attention модулями
    """
    def __init__(self, latent_dim=64, input_shape=(1, 32, 96, 84), dropout_rate=0.1,
                 use_attention=True, use_global_encoder=False,
                 use_global_decoder=False):
        super(Conv3DAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_channels = input_shape[0]
        self.input_depth = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.use_global_encoder = use_global_encoder
        self.use_global_decoder = use_global_decoder

        encoder_layers = []
        
        # Блок 1 энкодера: (1, 32, 96, 84) -> (32, 32, 48, 42)
        encoder_layers.append(nn.Conv3d(self.input_channels, 32, 3, stride=(1, 2, 2), padding=1))
        encoder_layers.append(nn.ReLU())
        if use_attention:
            encoder_layers.append(SpatialAttentionModule3D())
        if dropout_rate > 0:
            encoder_layers.append(nn.Dropout3d(dropout_rate))
        
        # Блок 2 энкодера: (32, 32, 48, 42) -> (64, 32, 24, 21)
        encoder_layers.append(nn.Conv3d(32, 64, 3, stride=(1, 2, 2), padding=1))
        encoder_layers.append(nn.ReLU())
        if use_attention:
            encoder_layers.append(SpatialAttentionModule3D())
        if dropout_rate > 0:
            encoder_layers.append(nn.Dropout3d(dropout_rate))

        # Блок 3 энкодера: (64, 32, 24, 21) -> (128, 32, 12, 11)
        encoder_layers.append(nn.Conv3d(64, 128, 3, stride=(1, 2, 2), padding=1))
        encoder_layers.append(nn.ReLU())
        if use_attention:
            encoder_layers.append(SpatialAttentionModule3D())
        if dropout_rate > 0:
            encoder_layers.append(nn.Dropout3d(dropout_rate))

        self.encoder_conv = nn.Sequential(*encoder_layers)

        # Вычисляем размер после свёрток
        with torch.no_grad():
            dummy = torch.randn(1, *input_shape)
            encoded = self.encoder_conv(dummy)
            self.flatten_size = encoded.numel() // encoded.shape[0]
            self.encoder_output_shape = encoded.shape[1:]
        
        self.fc_encoder = nn.Linear(self.flatten_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flatten_size)

        self.global_encoder = (
            nn.Linear(math.prod(input_shape), latent_dim)
            if use_global_encoder else None
        )
        self.latent_fusion = (
            nn.Linear(2 * latent_dim, latent_dim)
            if use_global_encoder else None
        )
        if self.latent_fusion is not None:
            # Start from an equal mixture of global and SAM features. Training
            # can then learn a different combination for every latent element.
            with torch.no_grad():
                self.latent_fusion.weight.zero_()
                self.latent_fusion.bias.zero_()
                identity = torch.eye(latent_dim)
                self.latent_fusion.weight[:, :latent_dim].copy_(0.5 * identity)
                self.latent_fusion.weight[:, latent_dim:].copy_(0.5 * identity)

        self.global_decoder = (
            nn.Linear(latent_dim, math.prod(input_shape))
            if use_global_decoder else None
        )

        # For Conv3d(kernel=3, stride=2, padding=1), each spatial size becomes
        # ceil(size/2). Compute the inverse output_padding for every decoder
        # stage so that ConvTranspose3d restores the input exactly.
        spatial_sizes = [(self.input_height, self.input_width)]
        for _ in range(3):
            height, width = spatial_sizes[-1]
            spatial_sizes.append(((height + 1) // 2, (width + 1) // 2))

        def inverse_output_padding(source, target):
            value = target - (2 * source - 1)
            if value not in (0, 1):
                raise ValueError(f"Невозможно восстановить размер {source} -> {target}")
            return value

        decoder_output_padding = []
        for source, target in zip(spatial_sizes[:0:-1], spatial_sizes[-2::-1]):
            decoder_output_padding.append((
                0,
                inverse_output_padding(source[0], target[0]),
                inverse_output_padding(source[1], target[1]),
            ))
        
        decoder_layers = []
        
        # Блок 1 декодера: (128, 32, 12, 11) -> (64, 32, 24, 21)
        decoder_layers.append(nn.ConvTranspose3d(
            128, 64, 3, stride=(1, 2, 2), padding=1,
            output_padding=decoder_output_padding[0],
        ))
        decoder_layers.append(nn.ReLU())
        if use_attention:
            decoder_layers.append(SpatialAttentionModule3D())
        if dropout_rate > 0:
            decoder_layers.append(nn.Dropout3d(dropout_rate))

        # Блок 2 декодера: (64, 32, 24, 21) -> (32, 32, 48, 42)
        decoder_layers.append(nn.ConvTranspose3d(
            64, 32, 3, stride=(1, 2, 2), padding=1,
            output_padding=decoder_output_padding[1],
        ))
        decoder_layers.append(nn.ReLU())
        if use_attention:
            decoder_layers.append(SpatialAttentionModule3D())
        if dropout_rate > 0:
            decoder_layers.append(nn.Dropout3d(dropout_rate))

        # Блок 3 декодера: (32, 32, 48, 42) -> (1, 32, 96, 84)
        decoder_layers.append(nn.ConvTranspose3d(
            32, self.input_channels, 3, stride=(1, 2, 2), padding=1,
            output_padding=decoder_output_padding[2],
        ))
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
        
        self.target_depth = input_shape[1]
        self.target_height = input_shape[2]
        self.target_width = input_shape[3]

    def forward(self, x):
        input_field = x
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1) 
        latent = self.fc_encoder(x)
        if self.global_encoder is not None:
            global_latent = self.global_encoder(input_field.flatten(start_dim=1))
            latent = self.latent_fusion(torch.cat([global_latent, latent], dim=1))

        x = self.fc_decoder(latent)
        x = x.view(x.size(0), *self.encoder_output_shape) 
        
        x = self.decoder_conv(x)
        if self.global_decoder is not None:
            # V2 adds global modes from the same transmitted latent vector; the
            # SAM convolutional path learns nonlinear and local corrections.
            global_reconstruction = self.global_decoder(latent).view(
                latent.size(0), self.input_channels, self.target_depth,
                self.target_height, self.target_width
            )
            x = x + global_reconstruction

        expected_shape = (self.input_channels, self.target_depth,
                          self.target_height, self.target_width)
        if tuple(x.shape[1:]) != expected_shape:
            raise RuntimeError(
                f"Декодер восстановил {tuple(x.shape[1:])}, ожидалось {expected_shape}"
            )
        
        return x, latent


class SAM3DAutoencoderV2(Conv3DAutoencoder):
    """SAM3D autoencoder with a global linear reconstruction decoder branch."""

    def __init__(self, latent_dim=64, input_shape=(1, 32, 96, 84), dropout_rate=0.1, **kwargs):
        super().__init__(
            latent_dim=latent_dim,
            input_shape=input_shape,
            dropout_rate=dropout_rate,
            use_attention=True,
            use_global_encoder=False,
            use_global_decoder=True,
        )


class SAM3DAutoencoderV3(Conv3DAutoencoder):
    """SAM3D autoencoder with symmetric global encoder and decoder branches."""

    def __init__(self, latent_dim=64, input_shape=(1, 32, 96, 84), dropout_rate=0.1, **kwargs):
        super().__init__(
            latent_dim=latent_dim,
            input_shape=input_shape,
            dropout_rate=dropout_rate,
            use_attention=True,
            use_global_encoder=True,
            use_global_decoder=True,
        )


class PlainConv3DAutoencoder(Conv3DAutoencoder):
    """
    Обычный 3D сверточный автоэнкодер без attention модулей.

    Архитектурно совпадает с Conv3DAutoencoder, но все блоки внимания отключены.
    Это удобный baseline для оценки вклада attention-механизмов.
    """

    def __init__(self, latent_dim=64, input_shape=(1, 32, 96, 84), dropout_rate=0.1, **kwargs):
        super(PlainConv3DAutoencoder, self).__init__(
            latent_dim=latent_dim,
            input_shape=input_shape,
            dropout_rate=dropout_rate,
            use_attention=False,
            use_global_encoder=False,
            use_global_decoder=False,
        )


class SimpleConv3DAutoencoder(nn.Module):
    """
    Простой 3D автоэнкодер без attention модулей
    """
    def __init__(self, latent_dim=64, input_shape=(1, 32, 96, 84), dropout_rate=0.1, **kwargs):
        super(SimpleConv3DAutoencoder, self).__init__()
        
        self.input_channels = input_shape[0]
        self.target_depth = input_shape[1]
        self.target_height = input_shape[2]
        self.target_width = input_shape[3]
        
        # Энкодер
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        
        # Вычисляем размер после энкодера
        with torch.no_grad():
            dummy = torch.randn(1, *input_shape)
            encoded = self.enc3(self.enc2(self.enc1(dummy)))
            self.flatten_size = encoded.numel() // encoded.shape[0]
            self.encoder_shape = encoded.shape[1:]
        
        # Latent space
        self.fc_enc = nn.Linear(self.flatten_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flatten_size)
        
        # Декодер
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=(1, 2, 2), padding=1, output_padding=(0, 0, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, stride=(1, 2, 2), padding=1, output_padding=(0, 0, 0)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(16, 1, 3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encode
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        
        # Latent
        x = x.view(x.size(0), -1)
        latent = self.fc_enc(x)
        
        # Decode
        x = self.fc_dec(latent)
        x = x.view(x.size(0), *self.encoder_shape)
        
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        
        # Interpolate до точных размеров
        x = F.interpolate(x, size=(self.target_depth, self.target_height, self.target_width),
                          mode='trilinear', align_corners=False)
        
        return x, latent


def get_model(name, **kwargs):
    """Фабрика моделей"""
    models = {
        'Conv3DAutoencoder': Conv3DAutoencoder,
        'SAM3DAutoencoderV2': SAM3DAutoencoderV2,
        'SAM3DAutoencoderV3': SAM3DAutoencoderV3,
        'PlainConv3DAutoencoder': PlainConv3DAutoencoder,
        'SimpleConv3DAutoencoder': SimpleConv3DAutoencoder
    }
    
    if name not in models:
        raise ValueError(f"Неизвестная модель: {name}. Доступные: {list(models.keys())}")
    
    return models[name](**kwargs)


if __name__ == "__main__":
    # Тест моделей
    input_shape = (1, 32, 96, 84)
    
    print("Тест Conv3DAutoencoder:")
    model = Conv3DAutoencoder(latent_dim=64, input_shape=input_shape)
    x = torch.randn(1, *input_shape)
    out, latent = model(x)
    print(f"  Вход: {x.shape}, Выход: {out.shape}, Latent: {latent.shape}")

    print("\nТест PlainConv3DAutoencoder:")
    model = PlainConv3DAutoencoder(latent_dim=64, input_shape=input_shape)
    out, latent = model(x)
    print(f"  Вход: {x.shape}, Выход: {out.shape}, Latent: {latent.shape}")

    print("\nТест SimpleConv3DAutoencoder:")
    model = SimpleConv3DAutoencoder(latent_dim=64, input_shape=input_shape)
    out, latent = model(x)
    print(f"  Вход: {x.shape}, Выход: {out.shape}, Latent: {latent.shape}")

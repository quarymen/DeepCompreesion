"""Original article architecture restored from commit 58f40c4.

Includes the historical output paddings and final trilinear interpolation.
Only class names differ; experimental architectures stay in src/models.py.
"""
"""
Модуль с архитектурами моделей для 3D Autoencoder
"""

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


class ArticleSAMAutoencoder(nn.Module):
    """
    3D сверточный автоэнкодер с attention модулями
    """
    def __init__(self, latent_dim=64, input_shape=(1, 32, 96, 84), dropout_rate=0.1, use_attention=True):
        super(ArticleSAMAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_channels = input_shape[0]
        self.input_depth = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention

        encoder_layers = []
        
        # Блок 1 энкодера: (1, 32, 96, 84) -> (32, 16, 48, 42)
        encoder_layers.append(nn.Conv3d(self.input_channels, 32, 3, stride=(1, 2, 2), padding=1))
        encoder_layers.append(nn.ReLU())
        if use_attention:
            encoder_layers.append(SpatialAttentionModule3D())
        if dropout_rate > 0:
            encoder_layers.append(nn.Dropout3d(dropout_rate))
        
        # Блок 2 энкодера: (32, 16, 48, 42) -> (64, 8, 24, 21)
        encoder_layers.append(nn.Conv3d(32, 64, 3, stride=(1, 2, 2), padding=1))
        encoder_layers.append(nn.ReLU())
        if use_attention:
            encoder_layers.append(SpatialAttentionModule3D())
        if dropout_rate > 0:
            encoder_layers.append(nn.Dropout3d(dropout_rate))

        # Блок 3 энкодера: (64, 8, 24, 21) -> (128, 4, 12, 11)
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
        
        decoder_layers = []
        
        # Блок 1 декодера: (128, 4, 12, 11) -> (64, 8, 24, 22)
        decoder_layers.append(nn.ConvTranspose3d(128, 64, 3, stride=(1, 2, 2), padding=1, output_padding=(0, 0, 1)))
        decoder_layers.append(nn.ReLU())
        if use_attention:
            decoder_layers.append(SpatialAttentionModule3D())
        if dropout_rate > 0:
            decoder_layers.append(nn.Dropout3d(dropout_rate))

        # Блок 2 декодера: (64, 8, 24, 22) -> (32, 16, 48, 44)
        decoder_layers.append(nn.ConvTranspose3d(64, 32, 3, stride=(1, 2, 2), padding=1, output_padding=(0, 0, 0)))
        decoder_layers.append(nn.ReLU())
        if use_attention:
            decoder_layers.append(SpatialAttentionModule3D())
        if dropout_rate > 0:
            decoder_layers.append(nn.Dropout3d(dropout_rate))

        # Блок 3 декодера: (32, 16, 48, 44) -> (1, 32, 96, 88)
        decoder_layers.append(nn.ConvTranspose3d(32, self.input_channels, 3, stride=(1, 2, 2), padding=1, output_padding=(0, 0, 0)))
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
        
        self.target_depth = input_shape[1]
        self.target_height = input_shape[2]
        self.target_width = input_shape[3]

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1) 
        return self.fc_encoder(x)

    def decode(self, latent):
        x = self.fc_decoder(latent)
        x = x.view(x.size(0), *self.encoder_output_shape) 
        
        x = self.decoder_conv(x)
        
        # Interpolate до точных размеров
        x = F.interpolate(x, size=(self.target_depth, self.target_height, self.target_width),
                          mode='trilinear', align_corners=False)
        
        return x

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent), latent


class ArticlePlainAutoencoder(ArticleSAMAutoencoder):
    """
    Обычный 3D сверточный автоэнкодер без attention модулей.

    Архитектурно совпадает с ArticleSAMAutoencoder, но все блоки внимания отключены.
    Это удобный baseline для оценки вклада attention-механизмов.
    """

    def __init__(self, latent_dim=64, input_shape=(1, 32, 96, 84), dropout_rate=0.1, **kwargs):
        super(ArticlePlainAutoencoder, self).__init__(
            latent_dim=latent_dim,
            input_shape=input_shape,
            dropout_rate=dropout_rate,
            use_attention=False,
        )

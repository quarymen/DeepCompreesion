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


class Conv3DAutoencoder(nn.Module):
    """
    3D сверточный автоэнкодер с attention модулями
    """
    def __init__(self, latent_dim=64, input_shape=(1, 32, 96, 84), dropout_rate=0.1, use_attention=True):
        super(Conv3DAutoencoder, self).__init__()
        
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

    def forward(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1) 
        latent = self.fc_encoder(x)
        
        x = self.fc_decoder(latent)
        x = x.view(x.size(0), *self.encoder_output_shape) 
        
        x = self.decoder_conv(x)
        
        # Interpolate до точных размеров
        x = F.interpolate(x, size=(self.target_depth, self.target_height, self.target_width),
                          mode='trilinear', align_corners=False)
        
        return x, latent


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
        )


def _group_count(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups


class ResidualBlock3D(nn.Module):
    """Two convolutions with an identity skip, GroupNorm and SiLU."""

    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        groups = _group_count(channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class ResidualSpatialAttention3D(nn.Module):
    """Near-identity residual attention with a learnable contribution."""

    def __init__(self, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(3, 1, kernel_size, padding=padding, bias=False)
        # A small non-zero value lets the mask convolution receive gradients
        # immediately while the module starts very close to an identity map.
        self.alpha = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x):
        statistics = torch.cat(
            [x.mean(dim=1, keepdim=True),
             x.amax(dim=1, keepdim=True),
             x.std(dim=1, keepdim=True, unbiased=False)],
            dim=1,
        )
        centered_mask = 2.0 * torch.sigmoid(self.conv(statistics)) - 1.0
        return x * (1.0 + self.alpha * centered_mask)


class ResidualConv3DAutoencoder(nn.Module):
    """Symmetric residual 3D CAE with an exact spatial decoder."""

    def __init__(self, latent_dim=64, input_shape=(1, 3, 96, 84),
                 dropout_rate=0.1, use_attention=False, **kwargs):
        super().__init__()
        if len(input_shape) != 4:
            raise ValueError(f'input_shape must be (C,D,H,W), got {input_shape}')
        channels, depth, height, width = map(int, input_shape)
        if min(depth, height, width) < 1:
            raise ValueError(f'Invalid input shape: {input_shape}')
        self.input_shape = tuple(map(int, input_shape))
        self.latent_dim = int(latent_dim)

        def norm(ch):
            return nn.GroupNorm(_group_count(ch), ch)

        def attention():
            return ResidualSpatialAttention3D() if use_attention else nn.Identity()

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3,
                          stride=(1, 2, 2), padding=1, bias=False),
                norm(out_channels),
                nn.SiLU(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv3d(channels, 16, 3, padding=1, bias=False),
            norm(16),
            nn.SiLU(inplace=True),
        )
        self.enc1 = ResidualBlock3D(16, dropout_rate)
        self.attn_enc1 = attention()
        self.down1 = down(16, 32)
        self.enc2 = ResidualBlock3D(32, dropout_rate)
        self.attn_enc2 = attention()
        self.down2 = down(32, 48)
        self.enc3 = ResidualBlock3D(48, dropout_rate)
        self.attn_enc3 = attention()
        self.down3 = down(48, 64)
        self.bottleneck = ResidualBlock3D(64, dropout_rate)
        self.attn_bottleneck = attention()

        spatial_sizes = [(height, width)]
        for _ in range(3):
            previous_h, previous_w = spatial_sizes[-1]
            spatial_sizes.append(((previous_h + 1) // 2, (previous_w + 1) // 2))
        encoded_h, encoded_w = spatial_sizes[-1]
        self.encoder_shape = (64, depth, encoded_h, encoded_w)
        self.flatten_size = 64 * depth * encoded_h * encoded_w
        self.fc_encoder = nn.Linear(self.flatten_size, self.latent_dim)
        self.fc_decoder = nn.Linear(self.latent_dim, self.flatten_size)

        def output_padding(source, target):
            value = target - (2 * source - 1)
            if value not in (0, 1):
                raise ValueError(f'Cannot invert spatial size {source} -> {target}')
            return value

        def up(in_channels, out_channels, source_size, target_size):
            op_h = output_padding(source_size[0], target_size[0])
            op_w = output_padding(source_size[1], target_size[1])
            return nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels, out_channels, 3, stride=(1, 2, 2), padding=1,
                    output_padding=(0, op_h, op_w), bias=False,
                ),
                norm(out_channels),
                nn.SiLU(inplace=True),
            )

        self.dec_bottleneck = ResidualBlock3D(64, dropout_rate)
        self.attn_dec_bottleneck = attention()
        self.up3 = up(64, 48, spatial_sizes[3], spatial_sizes[2])
        self.dec3 = ResidualBlock3D(48, dropout_rate)
        self.attn_dec3 = attention()
        self.up2 = up(48, 32, spatial_sizes[2], spatial_sizes[1])
        self.dec2 = ResidualBlock3D(32, dropout_rate)
        self.attn_dec2 = attention()
        self.up1 = up(32, 16, spatial_sizes[1], spatial_sizes[0])
        self.dec1 = ResidualBlock3D(16, dropout_rate)
        self.attn_dec1 = attention()
        self.head = nn.Conv3d(16, channels, 3, padding=1)

    def forward(self, x):
        original_shape = x.shape[1:]
        x = self.attn_enc1(self.enc1(self.stem(x)))
        x = self.attn_enc2(self.enc2(self.down1(x)))
        x = self.attn_enc3(self.enc3(self.down2(x)))
        x = self.attn_bottleneck(self.bottleneck(self.down3(x)))
        latent = self.fc_encoder(x.flatten(1))
        x = self.fc_decoder(latent).view(x.size(0), *self.encoder_shape)
        x = self.attn_dec_bottleneck(self.dec_bottleneck(x))
        x = self.attn_dec3(self.dec3(self.up3(x)))
        x = self.attn_dec2(self.dec2(self.up2(x)))
        x = self.attn_dec1(self.dec1(self.up1(x)))
        reconstructed = self.head(x)
        if reconstructed.shape[1:] != original_shape:
            raise RuntimeError(
                f'Exact decoder shape mismatch: {reconstructed.shape[1:]} vs {original_shape}'
            )
        return reconstructed, latent


class ResidualAttentionConv3DAutoencoder(ResidualConv3DAutoencoder):
    """Residual CAE with near-identity spatial attention in matching stages."""

    def __init__(self, **kwargs):
        super().__init__(use_attention=True, **kwargs)


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
        'PlainConv3DAutoencoder': PlainConv3DAutoencoder,
        'ResidualConv3DAutoencoder': ResidualConv3DAutoencoder,
        'ResidualAttentionConv3DAutoencoder': ResidualAttentionConv3DAutoencoder,
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

    print("\nТест ResidualAttentionConv3DAutoencoder:")
    model = ResidualAttentionConv3DAutoencoder(latent_dim=64, input_shape=input_shape)
    out, latent = model(x)
    print(f"  Вход: {x.shape}, Выход: {out.shape}, Latent: {latent.shape}")

    print("\nТест SimpleConv3DAutoencoder:")
    model = SimpleConv3DAutoencoder(latent_dim=64, input_shape=input_shape)
    out, latent = model(x)
    print(f"  Вход: {x.shape}, Выход: {out.shape}, Latent: {latent.shape}")

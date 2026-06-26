#!/usr/bin/env python3
"""
Основной скрипт для обучения 3D Autoencoder на данных концентрации CO
"""

import sys
import yaml
from pathlib import Path

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models import get_model
from src.data import load_data, normalize_data, create_dataloaders
from src.train import Trainer


def load_config(config_path='configs/config.yaml'):
    """Загрузка конфигурации"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def plot_history(history, save_path='results/plots/training_history.png'):
    """Построение графиков обучения"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Total Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # MSE
    axes[1].plot(history['train_mse'], label='Train MSE')
    axes[1].plot(history['val_mse'], label='Val MSE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('MSE Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # MAE
    axes[2].plot(history['train_mae'], label='Train MAE')
    axes[2].plot(history['val_mae'], label='Val MAE')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MAE')
    axes[2].set_title('MAE Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Графики сохранены в {save_path}")
    plt.close()


def main():
    print("="*70)
    print("3D Autoencoder для данных концентрации CO")
    print("="*70)
    
    # Загрузка конфигурации
    config = load_config()
    print(f"\nКонфигурация загружена")
    print(f"  Модель: {config['model']['name']}")
    print(f"  Latent dim: {config['model']['latent_dim']}")
    print(f"  Dropout: {config['model']['dropout_rate']}")
    print(f"  Use attention: {config['model']['use_attention']}")
    
    # Загрузка данных
    print("\n" + "="*70)
    print("ЗАГРУЗКА ДАННЫХ")
    print("="*70)
    
    base_dir = config['data']['dataset_dir']
    all_files = config['data']['files']
    
    print(f"Всего файлов: {len(all_files)}")
    
    # Загружаем все данные
    train_data = load_data(all_files, base_dir=base_dir)
    
    # Нормализация
    print("\nНормализация...")
    train_normalized = normalize_data(train_data)
    
    # Создание DataLoader (случайное разбиение 90/10)
    print("\nСоздание DataLoader (случайное разбиение 90/10)...")
    train_loader, val_loader = create_dataloaders(
        train_normalized,
        val_split=config['data']['val_split_ratio'],
        batch_size=config['training']['batch_size'],
        random_state=config['data']['random_seed']
    )
    
    # Создание модели
    print("\n" + "="*70)
    print("СОЗДАНИЕ МОДЕЛИ")
    print("="*70)
    
    model = get_model(
        name=config['model']['name'],
        latent_dim=config['model']['latent_dim'],
        input_shape=tuple(config['model']['input_shape']),
        dropout_rate=config['model']['dropout_rate'],
        use_attention=config['model']['use_attention']
    )
    
    # Проверка размеров
    test_input = torch.randn(1, *config['model']['input_shape'])
    test_output, _ = model(test_input)
    print(f"Вход: {tuple(test_input.shape)}, Выход: {tuple(test_output.shape)}")
    assert test_input.shape == test_output.shape, "Размеры не совпадают!"
    print("✓ Размеры совпадают")
    
    # Обучение
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ")
    print("="*70)
    
    trainer = Trainer(model, config, save_dir='models')
    
    history, best_loss, best_mse = trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=config['training']['num_epochs'],
        log_every=config['logging']['log_every_n_epochs']
    )
    
    # Сохранение
    print("\n" + "="*70)
    print("СОХРАНЕНИЕ")
    print("="*70)
    
    trainer.save_model('best_model.pth', metadata={
        'input_shape': config['model']['input_shape'],
        'latent_dim': config['model']['latent_dim']
    })
    trainer.save_history()
    
    plot_history(history)
    
    print(f"\n{'='*70}")
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*70}")
    print(f"Лучший validation loss: {best_loss:.5f}")
    print(f"Лучший validation MSE: {best_mse:.5f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

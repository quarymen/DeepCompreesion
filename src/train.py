"""
Модуль для обучения модели
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from pathlib import Path


class WeightedHeightLoss(nn.Module):
    """
    Функция потерь с повышенным весом для первых 5 высотных уровней
    """
    def __init__(self, alpha=1.0, beta=0.2, weight_first_5=5.0):
        super(WeightedHeightLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')
        self.alpha = alpha
        self.beta = beta
        self.weight_first_5 = weight_first_5

    def forward(self, reconstructed, target):
        assert reconstructed.shape == target.shape, \
            f"Размеры не совпадают! {reconstructed.shape} vs {target.shape}"

        rec_first = reconstructed[:, :, :5, :, :]
        tgt_first = target[:, :, :5, :, :]
        rec_rest = reconstructed[:, :, 5:, :, :]
        tgt_rest = target[:, :, 5:, :, :]
        
        mse_first = self.mse_loss(rec_first, tgt_first).mean()
        mae_first = self.mae_loss(rec_first, tgt_first).mean()
        mse_rest = self.mse_loss(rec_rest, tgt_rest).mean()
        mae_rest = self.mae_loss(rec_rest, tgt_rest).mean()
        
        loss_first = self.alpha * mse_first + self.beta * mae_first
        loss_rest = self.alpha * mse_rest + self.beta * mae_rest
        
        total_loss = (self.weight_first_5 * loss_first + loss_rest) / (self.weight_first_5 + 1)
        
        n_first = 5
        n_rest = reconstructed.shape[2] - 5
        avg_mse = (mse_first * n_first + mse_rest * n_rest) / reconstructed.shape[2]
        avg_mae = (mae_first * n_first + mae_rest * n_rest) / reconstructed.shape[2]

        return total_loss, avg_mse, avg_mae


def train_epoch(model, loader, criterion, optimizer, device):
    """Одна эпоха обучения"""
    model.train()
    total_loss, mse_loss, mae_loss, n = 0, 0, 0, 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        rec, _ = model(batch)
        loss, mse, mae = criterion(rec, batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        mse_loss += mse.item()
        mae_loss += mae.item()
        n += 1
    
    return total_loss/n, mse_loss/n, mae_loss/n


def validate_epoch(model, loader, criterion, device):
    """Валидация"""
    model.eval()
    total_loss, mse_loss, mae_loss, n = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            rec, _ = model(batch)
            loss, mse, mae = criterion(rec, batch)
            
            total_loss += loss.item()
            mse_loss += mse.item()
            mae_loss += mae.item()
            n += 1
    
    return total_loss/n, mse_loss/n, mae_loss/n


class Trainer:
    """Класс для управления процессом обучения"""
    
    def __init__(self, model, config, save_dir='models'):
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Оптимизатор
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['model']['weight_decay']
        )
        
        # Функция потерь
        self.criterion = WeightedHeightLoss(
            alpha=config['loss']['alpha'],
            beta=config['loss']['beta'],
            weight_first_5=config['loss']['height_weight']
        ).to(self.device)
        
        # Scheduler (опционально)
        scheduler_config = config['training'].get('scheduler', {})
        if scheduler_config.get('use_scheduler', True):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 20),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        else:
            self.scheduler = None
            print("  Scheduler отключён")
        
        # Early stopping
        self.patience = config['training']['early_stopping']['patience']
        self.min_delta = config['training']['early_stopping']['min_delta']
        
        # История
        self.history = {
            'train_loss': [],
            'train_mse': [],
            'train_mae': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_mse = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def train(self, train_loader, val_loader, num_epochs, log_every=5):
        """Полный цикл обучения"""
        
        print(f"\n{'='*70}")
        print(f"Обучение на устройстве: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*70}")
        print(f"Эпох: {num_epochs}, Batch: {self.config['training']['batch_size']}")
        print(f"LR: {self.config['training']['learning_rate']}, Weight decay: {self.config['model']['weight_decay']}")
        print(f"Early stopping: {self.patience}, Dropout: {self.config['model']['dropout_rate']}")
        if self.scheduler is not None:
            print(f"Scheduler: StepLR (step={self.config['training']['scheduler']['step_size']}, gamma={self.config['training']['scheduler']['gamma']})")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Обучение
            train_loss, train_mse, train_mae = train_epoch(
                self.model, train_loader, self.criterion, self.optimizer, self.device
            )
            
            # Валидация
            val_loss, val_mse, val_mae = validate_epoch(
                self.model, val_loader, self.criterion, self.device
            )
            
            # Scheduler (если включён)
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['train_mse'].append(train_mse)
            self.history['train_mae'].append(train_mae)
            self.history['val_loss'].append(val_loss)
            self.history['val_mse'].append(val_mse)
            self.history['val_mae'].append(val_mae)
            
            # Early stopping check
            improved = self.best_val_loss - val_loss > self.min_delta
            if improved:
                self.best_val_loss = val_loss
                self.best_val_mse = val_mse
                self.patience_counter = 0
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                status = "✓"
            else:
                self.patience_counter += 1
                status = f"({self.patience_counter})"
            
            # Логирование
            if (epoch + 1) % log_every == 0 or epoch == 0 or self.patience_counter >= self.patience - 1:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                      f"Train MSE: {train_mse:.5f} | Val MSE: {val_mse:.5f} | "
                      f"Best: {self.best_val_loss:.5f} {status} | "
                      f"Time: {epoch_time:.1f}s")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⚠ Early stopping на эпохе {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        # Восстановление лучшей модели
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✓ Восстановлена лучшая модель (val_loss={self.best_val_loss:.5f}, val_mse={self.best_val_mse:.5f})")
        
        print(f"\n{'='*70}")
        print(f"Обучение завершено! Общее время: {total_time:.1f}s")
        print(f"{'='*70}")
        
        return self.history, self.best_val_loss, self.best_val_mse
    
    def save_model(self, name='best_model.pth', metadata=None):
        """Сохранение модели"""
        save_path = self.save_dir / name
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_mse': self.best_val_mse,
            'config': self.config
        }
        
        if metadata:
            checkpoint.update(metadata)
        
        torch.save(checkpoint, save_path)
        print(f"✓ Модель сохранена в {save_path}")
        
        return save_path
    
    def save_history(self, name='training_history.json'):
        """Сохранение истории обучения"""
        save_path = self.save_dir / name
        
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"✓ История сохранена в {save_path}")
        return save_path


def load_model(checkpoint_path, model_class, device=None):
    """Загрузка сохранённой модели"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Извлекаем параметры для создания модели
    config = checkpoint.get('config', {})
    model_params = config.get('model', {})
    
    model = model_class(
        latent_dim=model_params.get('latent_dim', 64),
        input_shape=tuple(model_params.get('input_shape', [1, 32, 96, 84])),
        dropout_rate=model_params.get('dropout_rate', 0.1),
        use_attention=model_params.get('use_attention', True)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Модель загружена из {checkpoint_path}")
    print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"  Best Val MSE: {checkpoint.get('best_val_mse', 'N/A')}")
    
    return model, checkpoint


if __name__ == "__main__":
    # Тест
    print("Тест модуля train...")
    
    from models import SimpleConv3DAutoencoder
    from data import ConcentrationFieldDataset
    from torch.utils.data import DataLoader
    import numpy as np
    
    # Создаём тестовые данные
    test_data = np.random.randn(100, 32, 96, 84)
    dataset = ConcentrationFieldDataset(test_data)
    loader = DataLoader(dataset, batch_size=4)
    
    # Модель
    model = SimpleConv3DAutoencoder(latent_dim=64)
    
    # Конфиг
    config = {
        'model': {'latent_dim': 64, 'dropout_rate': 0.1, 'weight_decay': 1e-4, 'input_shape': [1, 32, 96, 84]},
        'training': {'learning_rate': 0.001, 'batch_size': 4, 'scheduler': {'step_size': 10, 'gamma': 0.5}, 'early_stopping': {'patience': 5, 'min_delta': 0.0001}},
        'loss': {'alpha': 1.0, 'beta': 0.2, 'height_weight': 5.0}
    }
    
    # Trainer
    trainer = Trainer(model, config, save_dir='test_models')
    history, best_loss, best_mse = trainer.train(loader, loader, num_epochs=3, log_every=1)
    
    print(f"\nРезультаты теста:")
    print(f"  Best Val Loss: {best_loss:.5f}")
    print(f"  Best Val MSE: {best_mse:.5f}")

"""
Модуль для работы с данными: загрузка, нормализация, Dataset классы
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import netCDF4 as ncdf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


class ConcentrationFieldDataset(Dataset):
    """Dataset для 3D полей концентрации"""
    
    def __init__(self, data_array, variable_name='co'):
        """
        Args:
            data_array: numpy array формы (num_samples, depth, height, width)
            variable_name: имя переменной (для совместимости)
        """
        self.data = torch.tensor(data_array).float().unsqueeze(1)
        if self.data.ndim != 5 or self.data.shape[1] != 1:
            raise ValueError(
                f"Некорректная форма данных. Ожидалось (num_samples, 1, D, H, W), "
                f"получено {self.data.shape}"
            )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def shape(self):
        return self.data.shape[1:]  # Без размерности батча


def load_data(file_paths, base_dir='', variable_name='co', verbose=True):
    """
    Загрузка данных из netCDF файлов
    
    Args:
        file_paths: список путей к файлам
        base_dir: базовая директория
        variable_name: имя переменной в netCDF
        verbose: выводить ли прогресс
    
    Returns:
        numpy array формы (num_samples, depth, height, width)
    """
    all_data = []
    
    iterator = tqdm(file_paths, desc="Загрузка файлов") if verbose else file_paths
    
    for file_path in iterator:
        full_path = f"{base_dir}/{file_path}" if base_dir else file_path
        try:
            dataset = ncdf.Dataset(full_path)
            data = np.array(dataset.variables[variable_name])
            all_data.append(data)
            dataset.close()
        except Exception as e:
            print(f"Ошибка при обработке файла {full_path}: {e}")
    
    if not all_data:
        raise ValueError("Не удалось загрузить ни одного файла")
    
    combined = np.concatenate(all_data, axis=0)
    if verbose:
        print(f"Объединенные данные: {combined.shape}")
    
    return combined


def normalize_data(data, eps=1e-7, verbose=True):
    """
    Min-Max нормализация данных для каждого образца и высоты
    
    Args:
        data: numpy array формы (num_samples, depth, height, width)
        eps: константа для избежания деления на 0
        verbose: выводить ли информацию
    
    Returns:
        нормализованный numpy array той же формы
    """
    normalized = np.zeros_like(data)
    
    for i in range(data.shape[0]):  
        for j in range(data.shape[1]): 
            slice_data = data[i, j, :, :]
            min_val, max_val = slice_data.min(), slice_data.max()
            
            if max_val != min_val:
                normalized[i, j, :, :] = (slice_data - min_val) / (max_val - min_val + eps)
            else:
                normalized[i, j, :, :] = 0.0
    
    if verbose:
        print(f"Нормализованные данные: min={normalized.min():.4f}, max={normalized.max():.4f}")
    
    return normalized


def create_dataloaders(
    data,
    val_split=0.1,
    batch_size=32,
    random_state=42
):
    """
    Создание DataLoader для train/val со случайным разбиением
    
    Args:
        data: numpy array с данными
        val_split: доля валидации (по умолчанию 10%)
        batch_size: размер батча
        random_state: seed для воспроизводимости
    
    Returns:
        train_loader, val_loader
    """
    # Случайное разбиение по кадрам
    indices = list(range(len(data)))
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=val_split, 
        random_state=random_state
    )
    
    full_dataset = ConcentrationFieldDataset(data)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Случайное разбиение: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"Доля валидации: {len(val_dataset)/len(full_dataset)*100:.1f}%")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def denormalize_data(normalized_data, original_min, original_max, eps=1e-7):
    """
    Обратная нормализация
    
    Args:
        normalized_data: нормализованные данные
        original_min: оригинальные минимумы
        original_max: оригинальные максимумы
        eps: константа
    
    Returns:
        денормализованные данные
    """
    range_val = original_max - original_min
    range_val[range_val == 0] = 1  # Избегаем деления на 0
    
    return normalized_data * range_val + original_min


if __name__ == "__main__":
    # Тест
    print("Тест модуля data...")
    
    # Создаём тестовые данные
    test_data = np.random.randn(100, 32, 96, 84)
    
    # Нормализация
    normalized = normalize_data(test_data)
    print(f"Нормализация: {test_data.shape} -> {normalized.shape}")
    
    # Dataset
    dataset = ConcentrationFieldDataset(normalized)
    print(f"Dataset: len={len(dataset)}, shape={dataset.shape}")
    
    # DataLoader
    loader = DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    print(f"Batch: {batch.shape}")

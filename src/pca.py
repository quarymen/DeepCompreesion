"""
Модуль для сжатия данных методом главных компонент (PCA)
"""

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

try:
    from .metrics import compute_reconstruction_metrics
except ImportError:
    from metrics import compute_reconstruction_metrics


class PCACompressor:
    """
    Класс для сжатия 3D данных с помощью PCA
    """
    
    def __init__(self, n_components=64):
        """
        Args:
            n_components: количество главных компонент (размерность latent space)
        """
        self.n_components = n_components
        self.pca = None
        self.mean_ = None
        self.std_ = None
        self.original_shape = None
    
    def fit(self, data, verbose=True):
        """
        Обучение PCA на данных
        
        Args:
            data: numpy array формы (n_samples, depth, height, width)
            verbose: выводить ли прогресс
        """
        self.original_shape = data.shape[1:]  # (depth, height, width)
        
        # Преобразуем данные в 2D форму для PCA: (n_samples, depth*height*width)
        n_samples = data.shape[0]
        data_2d = data.reshape(n_samples, -1)
        
        if verbose:
            print(f"Форма данных для PCA: {data_2d.shape}")
            print(f"Количество главных компонент: {self.n_components}")
        
        # Обучаем PCA
        self.pca = PCA(n_components=self.n_components)
        
        if verbose:
            print("Обучение PCA...")
        
        self.pca.fit(data_2d)
        
        # Сохраняем статистику
        self.mean_ = data_2d.mean(axis=0)
        self.std_ = data_2d.std(axis=0)
        
        if verbose:
            print(f"✓ PCA обучено")
            print(f"  Объяснённая дисперсия: {self.pca.explained_variance_ratio_.sum():.4f}")
            print(f"  Объяснённая дисперсия (по компонентам): {self.pca.explained_variance_ratio_[:10]}...")
        
        return self
    
    def transform(self, data, verbose=False):
        """
        Сжатие данных
        
        Args:
            data: numpy array формы (n_samples, depth, height, width)
            verbose: выводить ли прогресс
        
        Returns:
            compressed: numpy array формы (n_samples, n_components)
        """
        if self.pca is None:
            raise ValueError("Сначала вызовите fit() или fit_transform()")
        
        n_samples = data.shape[0]
        data_2d = data.reshape(n_samples, -1)
        
        if verbose:
            print("Сжатие данных...")
        
        compressed = self.pca.transform(data_2d)
        
        return compressed
    
    def reconstruct(self, compressed):
        """
        Восстановление данных из сжатого представления
        
        Args:
            compressed: numpy array формы (n_samples, n_components)
        
        Returns:
            reconstructed: numpy array формы (n_samples, depth, height, width)
        """
        if self.pca is None:
            raise ValueError("Сначала вызовите fit()")
        
        # Восстанавливаем из сжатого представления
        reconstructed_2d = self.pca.inverse_transform(compressed)
        
        # Возвращаем к исходной форме
        n_samples = compressed.shape[0]
        reconstructed = reconstructed_2d.reshape(n_samples, *self.original_shape)
        
        return reconstructed
    
    def fit_transform(self, data, verbose=True):
        """
        Обучение и сжатие данных
        
        Args:
            data: numpy array формы (n_samples, depth, height, width)
            verbose: выводить ли прогресс
        
        Returns:
            compressed: numpy array формы (n_samples, n_components)
        """
        self.fit(data, verbose)
        return self.transform(data, verbose)
    
    def reconstruction_error(self, data):
        """
        Вычисление ошибки реконструкции
        
        Args:
            data: numpy array формы (n_samples, depth, height, width)
        
        Returns:
            mse: средняя квадратичная ошибка
            mae: средняя абсолютная ошибка
        """
        compressed = self.transform(data)
        reconstructed = self.reconstruct(compressed)
        
        metrics = compute_reconstruction_metrics(data, reconstructed)
        mse = metrics["mse"]
        mae = metrics["mae"]
        
        return mse, mae
    
    def explained_variance_ratio(self):
        """Возвращает долю объяснённой дисперсии для каждой компоненты"""
        if self.pca is None:
            raise ValueError("Сначала вызовите fit()")
        return self.pca.explained_variance_ratio_
    
    def cumulative_variance_ratio(self):
        """Возвращает накопленную долю объяснённой дисперсии"""
        if self.pca is None:
            raise ValueError("Сначала вызовите fit()")
        return np.cumsum(self.pca.explained_variance_ratio_)
    
    def save(self, filepath):
        """Сохранение модели PCA"""
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'n_components': self.n_components,
                'original_shape': self.original_shape,
                'mean_': self.mean_,
                'std_': self.std_
            }, f)
        
        print(f"✓ PCA модель сохранена в {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Загрузка модели PCA"""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        compressor = cls(n_components=data['n_components'])
        compressor.pca = data['pca']
        compressor.original_shape = data['original_shape']
        compressor.mean_ = data['mean_']
        compressor.std_ = data['std_']
        
        print(f"✓ PCA модель загружена из {filepath}")
        
        return compressor


def analyze_pca_components(
    data,
    n_components_list=[16, 32, 64, 128, 256],
    train_data=None,
    eval_data=None,
    verbose=True,
):
    """
    Анализ качества сжатия PCA с разным количеством компонент
    
    Args:
        data: numpy array формы (n_samples, depth, height, width)
        n_components_list: список количеств компонент для тестирования
        verbose: выводить ли прогресс
    
    Returns:
        results: словарь с результатами для каждого количества компонент
    """
    if train_data is None:
        train_data = data
    if eval_data is None:
        eval_data = data

    results = {}
    max_components = min(train_data.shape[0], train_data[0].size)
    
    for n_comp in tqdm(n_components_list, desc="Анализ PCA"):
        if n_comp > max_components:
            if verbose:
                print(
                    f"\nPCA: пропускаю n_components={n_comp}, "
                    f"так как максимум для данных равен {max_components}"
                )
            continue

        compressor = PCACompressor(n_components=n_comp)
        compressor.fit(train_data, verbose=False)
        compressed = compressor.transform(eval_data, verbose=False)
        
        # Вычисляем ошибку реконструкции
        reconstructed = compressor.reconstruct(compressed)
        metrics = compute_reconstruction_metrics(eval_data, reconstructed)
        
        # Доля объяснённой дисперсии
        var_ratio = compressor.explained_variance_ratio().sum()
        cum_var = compressor.cumulative_variance_ratio()[-1]
        
        results[n_comp] = {
            **metrics,
            'explained_variance': var_ratio,
            'cumulative_variance': cum_var,
            'compression_ratio': eval_data[0].size / n_comp,
            'compressed_size': n_comp,
            'compressor': compressor
        }
        
        if verbose:
            print(f"\nn_components={n_comp}:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  NRMSE: {metrics['nrmse']:.6f}")
            print(f"  R2: {metrics['r2']:.6f}")
            print(f"  Объяснённая дисперсия: {var_ratio:.4f}")
            print(f"  Коэффициент сжатия: {eval_data[0].size / n_comp:.1f}x")
    
    return results


if __name__ == "__main__":
    # Тест
    print("Тест PCA модуля...")
    
    # Создаём тестовые данные
    np.random.seed(42)
    test_data = np.random.randn(100, 32, 96, 84)
    
    # Обучение PCA
    compressor = PCACompressor(n_components=64)
    compressed = compressor.fit_transform(test_data)
    
    print(f"\nОригинал: {test_data.shape}")
    print(f"Сжатие: {compressed.shape}")
    
    # Восстановление
    reconstructed = compressor.reconstruct(compressed)
    print(f"Восстановление: {reconstructed.shape}")
    
    # Ошибка
    mse, mae = compressor.reconstruction_error(test_data)
    print(f"\nMSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    # Объяснённая дисперсия
    print(f"\nОбъяснённая дисперсия: {compressor.explained_variance_ratio().sum():.4f}")

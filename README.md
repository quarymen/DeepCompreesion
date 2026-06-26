# Сжатие полей концентрации CO

## 📁 Структура проекта

```
co_autoencoder_project/
├── configs/
│   └── config.yaml          # Конфигурационный файл
├── src/
│   ├── __init__.py
│   ├── models.py            # Архитектуры моделей
│   ├── data.py              # Загрузка и обработка данных
│   └── train.py             # Обучение и метрики
├── models/                   # Сохранённые модели
├── notebooks/
│   └── analysis.ipynb       # Анализ результатов
├── results/
│   ├── plots/               # Графики обучения
│   └── predictions/         # Предсказания
├── data/                     # Данные (опционально)
├── train.py                  # Основной скрипт обучения
├── requirements.txt
└── README.md
```

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Обучение модели

```bash
cd co_autoencoder_project
python train.py
```

### Анализ результатов

Откройте `notebooks/analysis.ipynb` в Jupyter Notebook/JupyterLab для анализа модели.

### Сравнение с PCA

Откройте `notebooks/pca_analysis.ipynb` для:
- Анализа качества сжатия PCA
- Сравнения PCA и автоэнкодера
- Подбора оптимального количества компонент

## 📊 Модели

### Conv3DAutoencoder
3D сверточный автоэнкодер с модулями пространственного внимания:
- Encoder: 3 блока Conv3D + Attention + Dropout
- Latent space: линейный слой
- Decoder: 3 блока ConvTranspose3D + Attention + Dropout

### SimpleConv3DAutoencoder
Упрощённая версия без attention модулей:
- Encoder: 3 блока Conv3D + BatchNorm + Dropout
- Latent space: линейный слой
- Decoder: 3 блока ConvTranspose3D + BatchNorm + Dropout

### PCACompressor
Метод главных компонент для сжатия данных:
- Линейное сжатие
- Интерпретируемые компоненты
- Базовый уровень для сравнения с автоэнкодером

## ⚙️ Конфигурация

Основные параметры в `configs/config.yaml`:

```yaml
model:
  name: "Conv3DAutoencoder"  # или "SimpleConv3DAutoencoder"
  latent_dim: 64
  dropout_rate: 0.1
  use_attention: true

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  early_stopping:
    patience: 10
```

## 📈 Метрики

- **Total Loss**: взвешенная комбинация MSE + MAE
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error

Функция потерь с повышенным весом для первых 5 высотных уровней:

```
loss = (weight_first_5 * loss_first_5 + loss_rest) / (weight_first_5 + 1)
```

## 📉 Валидация

В проекте используется **случайная валидация по кадрам**: 10% данных случайно выбираются из всех файлов для валидации. Это обеспечивает более репрезентативную выборку и лучшее обобщение.

Параметры валидации в `configs/config.yaml`:
```yaml
data:
  val_split_ratio: 0.10    # 10% кадров для валидации
  random_seed: 42           # Для воспроизводимости

## 🔬 Результаты

После обучения в `results/plots/` сохраняются:
- `training_history.png` - графики обучения
- `reconstruction_examples.png` - примеры реконструкции
- `errors_by_height.png` - ошибки по высотам
- `error_distribution.png` - распределение ошибок

## 📝 Примеры использования

### Загрузка модели

```python
from src.models import load_model, Conv3DAutoencoder

model, checkpoint = load_model('models/best_model.pth', Conv3DAutoencoder)
```

### Предсказание

```python
import torch
from src.data import ConcentrationFieldDataset

model.eval()
with torch.no_grad():
    sample = dataset[0].unsqueeze(0).cuda()
    reconstructed, latent = model(sample)
```

## 👨‍💻 Авторы

ИВМиМГ СО РАН Поляков С.А 
ИВМиМГ СО РАН д.ф-м. н Пененко А.В


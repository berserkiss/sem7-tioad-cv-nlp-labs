"""
Скрипт для скачивания датасетов с Kaggle для классификации текста

Использование:
    python download_dataset.py

Требования:
    pip install kaggle pandas

Перед использованием:
    1. Зарегистрируйтесь на Kaggle (https://www.kaggle.com)
    2. Скачайте API credentials: Settings → API → Create New Token
    3. Поместите kaggle.json в ~/.kaggle/ (или C:\Users\YourUsername\.kaggle\ на Windows)
"""

import os
import zipfile
from pathlib import Path
import pandas as pd

# Популярные датасеты для русскоязычной классификации текста
DATASETS = {
    'russian_sentiment': {
        'name': 'cryptexcode/russian-sentiment-analysis-dataset',
        'description': 'Русскоязычный датасет для анализа тональности',
        'file': 'reviews.csv'
    },
    'russian_reviews': {
        'name': 'artemkonevskoy/russian-reviews-dataset',
        'description': 'Большой датасет русскоязычных отзывов',
        'file': 'reviews.csv'
    },
    'lenta_news': {
        'name': 'yutkin/corpus-of-russian-news-articles-from-lenta',
        'description': 'Корпус новостей Lenta.ru с категориями',
        'file': 'lenta-ru-news.csv'
    }
}


def download_kaggle_dataset(dataset_key: str, output_dir: Path = Path('datasets')):
    """
    Скачивает датасет с Kaggle
    
    Args:
        dataset_key: Ключ датасета из словаря DATASETS
        output_dir: Директория для сохранения
        
    Returns:
        Путь к скачанному CSV файлу или None
    """
    try:
        import kaggle
    except ImportError:
        print("Ошибка: библиотека kaggle не установлена!")
        print("Установите: pip install kaggle")
        print("\nТакже убедитесь, что:")
        print("1. Зарегистрировались на Kaggle (https://www.kaggle.com)")
        print("2. Скачали API credentials (Settings → API → Create New Token)")
        print("3. Поместили kaggle.json в ~/.kaggle/ (или C:\\Users\\YourUsername\\.kaggle\\ на Windows)")
        return None
    
    if dataset_key not in DATASETS:
        print(f"Ошибка: неизвестный датасет '{dataset_key}'")
        print(f"Доступные датасеты: {list(DATASETS.keys())}")
        return None
    
    dataset_info = DATASETS[dataset_key]
    dataset_name = dataset_info['name']
    
    print(f"Скачивание датасета: {dataset_info['description']}")
    print(f"Kaggle dataset: {dataset_name}")
    
    # Создаем директорию для датасетов
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Скачиваем датасет
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(output_dir),
            unzip=True
        )
        
        # Ищем CSV файл
        csv_files = list(output_dir.glob('*.csv'))
        if csv_files:
            csv_path = csv_files[0]
            print(f"Датасет успешно скачан: {csv_path}")
            return csv_path
        else:
            print("Предупреждение: CSV файл не найден в скачанном архиве")
            return None
            
    except Exception as e:
        print(f"Ошибка при скачивании датасета: {e}")
        print("\nВозможные причины:")
        print("1. Неправильные API credentials (проверьте kaggle.json)")
        print("2. Датасет требует принятия правил использования (примите на сайте Kaggle)")
        print("3. Проблемы с интернет-соединением")
        return None


def list_available_datasets():
    """Выводит список доступных датасетов"""
    print("Доступные датасеты для скачивания:")
    print("=" * 60)
    for key, info in DATASETS.items():
        print(f"\nКлюч: {key}")
        print(f"  Описание: {info['description']}")
        print(f"  Kaggle: {info['name']}")


def main():
    """Основная функция"""
    print("=" * 60)
    print("Скачивание датасетов для классификации текста")
    print("=" * 60)
    
    list_available_datasets()
    
    print("\n" + "=" * 60)
    print("Выберите датасет для скачивания:")
    print("1. russian_sentiment - Анализ тональности")
    print("2. russian_reviews - Отзывы на товары")
    print("3. lenta_news - Новости с категориями")
    print("4. Показать список всех датасетов")
    print("0. Выход")
    
    choice = input("\nВведите номер (или ключ датасета): ").strip()
    
    dataset_map = {
        '1': 'russian_sentiment',
        '2': 'russian_reviews',
        '3': 'lenta_news',
    }
    
    if choice in dataset_map:
        dataset_key = dataset_map[choice]
    elif choice in DATASETS:
        dataset_key = choice
    elif choice == '4':
        list_available_datasets()
        return
    elif choice == '0':
        return
    else:
        print("Неверный выбор!")
        return
    
    # Скачиваем датасет
    csv_path = download_kaggle_dataset(dataset_key)
    
    if csv_path:
        print(f"\nДатасет готов к использованию: {csv_path}")
        print("\nДля использования в text_classification.py:")
        print(f"  df = pd.read_csv('{csv_path}')")
    else:
        print("\nИспользуйте встроенный синтетический датасет в text_classification.py")


if __name__ == "__main__":
    main()


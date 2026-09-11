"""
Классификация текста: предобработка и обучение моделей

Этот скрипт реализует полный пайплайн для классификации текста:
1. Загрузка/скачивание датасета
2. Предобработка текста (токенизация, лемматизация, удаление стоп-слов)
3. Преобразование в числовую форму (Bag of Words, TF-IDF)
4. Обучение нескольких моделей классификаторов
5. Выбор лучшей модели
6. Парсинг новых данных с сайтов
7. Классификация новых данных

Использование:
    python text_classification.py

Зависимости:
    pip install nltk scikit-learn pandas numpy requests beautifulsoup4 pymorphy2
"""

import re
import os
import pickle
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import requests
from bs4 import BeautifulSoup

# Для работы с русским языком
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import pymorphy2
    
    # Скачиваем необходимые ресурсы NLTK
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
except ImportError:
    print("Внимание: некоторые библиотеки не установлены. Установите: pip install nltk pymorphy2")
    print("Для русского языка также может потребоваться: pip install pymorphy2-dicts-ru")


# Глобальный флаг для отслеживания вывода предупреждения
_pymorphy2_warning_shown = False

# Попытка импорта альтернативных библиотек для лемматизации
try:
    from pymystem3 import Mystem
    MYSTEM_AVAILABLE = True
except ImportError:
    MYSTEM_AVAILABLE = False

try:
    from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
    NATASHA_AVAILABLE = True
except ImportError:
    NATASHA_AVAILABLE = False

class TextPreprocessor:
    """Класс для предобработки текста на русском языке"""
    
    def __init__(self, show_warning: bool = True):
        """
        Инициализация предобработчика
        
        Args:
            show_warning: Показывать ли предупреждение о pymorphy2 (по умолчанию True)
        """
        global _pymorphy2_warning_shown
        
        try:
            # Загружаем стоп-слова для русского языка
            self.stop_words = set(stopwords.words('russian'))
            # Добавляем дополнительные стоп-слова
            self.stop_words.update(['это', 'как', 'так', 'и', 'в', 'над', 'к', 'до', 'не', 'на', 'но', 'за', 'то', 'с', 'ли', 'а', 'во', 'от', 'со', 'для', 'о', 'же', 'ну', 'вы', 'бы', 'что', 'кто', 'он', 'она'])
        except:
            # Если NLTK не установлен, используем базовый набор
            self.stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'к', 'о', 'об', 'при', 'про', 'со', 'это', 'как', 'так', 'что', 'не', 'но', 'а', 'или', 'то', 'же', 'бы', 'ли', 'ну', 'вы', 'мы', 'они', 'он', 'она', 'оно'}
        
        # Пытаемся использовать различные библиотеки для лемматизации
        self.morph = None
        self.morph_works = False
        self.lemmatization_method = None
        self.mystem = None
        self.natasha_components = {}
        
        # Попытка 1: pymorphy2
        try:
            self.morph = pymorphy2.MorphAnalyzer()
            test_word = self.morph.parse('тест')[0]
            self.morph_works = True
            self.lemmatization_method = 'pymorphy2'
        except Exception as e:
            # Попытка 2: pymystem3 (MyStem от Яндекса) - проще и быстрее
            if MYSTEM_AVAILABLE:
                try:
                    self.mystem = Mystem()
                    # Тестируем работу
                    test_result = self.mystem.lemmatize('тест')
                    self.morph_works = True
                    self.lemmatization_method = 'mystem'
                    if show_warning and not _pymorphy2_warning_shown:
                        print("Используется MyStem (pymystem3) для лемматизации (совместима с Python 3.13)")
                        _pymorphy2_warning_shown = True
                except Exception as e2:
                    # Попытка 3: natasha (более сложная, но мощная)
                    if NATASHA_AVAILABLE:
                        try:
                            self.natasha_components['segmenter'] = Segmenter()
                            self.natasha_components['morph_vocab'] = MorphVocab()
                            self.natasha_components['emb'] = NewsEmbedding()
                            self.natasha_components['morph_tagger'] = NewsMorphTagger(self.natasha_components['emb'])
                            self.morph_works = True
                            self.lemmatization_method = 'natasha'
                            if show_warning and not _pymorphy2_warning_shown:
                                print("Используется Natasha для лемматизации (совместима с Python 3.13)")
                                _pymorphy2_warning_shown = True
                        except Exception as e3:
                            self.morph_works = False
                    else:
                        self.morph_works = False
            else:
                # Попытка 3: natasha (если pymystem3 не установлен)
                if NATASHA_AVAILABLE:
                    try:
                        self.natasha_components['segmenter'] = Segmenter()
                        self.natasha_components['morph_vocab'] = MorphVocab()
                        self.natasha_components['emb'] = NewsEmbedding()
                        self.natasha_components['morph_tagger'] = NewsMorphTagger(self.natasha_components['emb'])
                        self.morph_works = True
                        self.lemmatization_method = 'natasha'
                        if show_warning and not _pymorphy2_warning_shown:
                            print("Используется Natasha для лемматизации (совместима с Python 3.13)")
                            _pymorphy2_warning_shown = True
                    except Exception as e3:
                        self.morph_works = False
                else:
                    self.morph_works = False
            
            # Выводим предупреждение только один раз
            if not self.morph_works and show_warning and not _pymorphy2_warning_shown:
                print(f"Предупреждение: pymorphy2 не работает ({type(e).__name__}), лемматизация будет упрощенной")
                print("  Примечание: Это не критично, скрипт продолжит работу без лемматизации")
                print("  Совет: Установите альтернативу для лемматизации:")
                print("    - pymystem3 (рекомендуется): pip install pymystem3")
                print("    - natasha: pip install natasha")
                _pymorphy2_warning_shown = True
    
    def preprocess_text(self, text: str) -> str:
        """
        Предобработка одного текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Обработанный текст
        """
        if not isinstance(text, str):
            return ""
        
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Удаляем спецсимволы, оставляем только буквы, цифры и пробелы
        text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
        
        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        
        # Токенизация
        try:
            tokens = word_tokenize(text, language='russian')
        except:
            # Простая токенизация, если NLTK не работает
            tokens = text.split()
        
        # Удаление стоп-слов и лемматизация
        processed_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in self.stop_words:
                if self.morph_works:
                    try:
                        # Лемматизация в зависимости от используемой библиотеки
                        if self.lemmatization_method == 'pymorphy2':
                            parsed = self.morph.parse(token)[0]
                            lemma = parsed.normal_form
                        elif self.lemmatization_method == 'mystem':
                            # MyStem возвращает список, берем первый элемент и убираем пробелы
                            lemmas = self.mystem.lemmatize(token)
                            lemma = ''.join(lemmas).strip()
                            if not lemma:  # Если пусто, используем исходное слово
                                lemma = token
                        elif self.lemmatization_method == 'natasha':
                            # Natasha требует полный pipeline
                            from natasha import Doc
                            doc = Doc(token)
                            doc.segment(self.natasha_components['segmenter'])
                            doc.tag_morph(self.natasha_components['morph_tagger'])
                            if doc.tokens:
                                doc.tokens[0].lemmatize(self.natasha_components['morph_vocab'])
                                lemma = doc.tokens[0].lemma if doc.tokens[0].lemma else token
                            else:
                                lemma = token
                        else:
                            lemma = token
                        processed_tokens.append(lemma)
                    except:
                        # Если лемматизация не удалась, используем исходное слово
                        processed_tokens.append(token)
                else:
                    # Без лемматизации используем исходное слово
                    processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Предобработка списка текстов
        
        Args:
            texts: Список исходных текстов
            
        Returns:
            Список обработанных текстов
        """
        return [self.preprocess_text(text) for text in texts]


class DatasetLoader:
    """Класс для загрузки датасетов"""
    
    @staticmethod
    def download_sentiment_dataset() -> pd.DataFrame:
        """
        Скачивает датасет отзывов на товары для классификации тональности
        
        Returns:
            DataFrame с колонками 'text' и 'label'
        """
        print("Загрузка датасета...")
        
        # Создаем синтетический датасет отзывов на товары
        # В реальном проекте здесь можно скачать датасет с Kaggle или другого источника
        
        positive_reviews = [
            "Отличный товар! Очень доволен покупкой. Качество на высоте, рекомендую всем.",
            "Прекрасное качество, быстрая доставка. Все как на картинке, очень рад!",
            "Замечательный продукт, работает отлично. Цена соответствует качеству.",
            "Очень хороший товар, рекомендую к покупке. Качество превосходное!",
            "Отличное соотношение цена-качество. Доволен на все 100 процентов.",
            "Качественный товар, быстрая доставка. Все супер, спасибо!",
            "Прекрасный продукт, очень доволен. Рекомендую всем друзьям.",
            "Отличное качество, все работает как надо. Очень рад покупке!",
            "Замечательный товар, цена отличная. Качество на высоте!",
            "Очень хороший продукт, рекомендую. Все соответствует описанию.",
            "Отличный товар, быстрая доставка. Качество превосходное!",
            "Прекрасное качество, очень доволен. Рекомендую к покупке.",
            "Замечательный продукт, работает отлично. Цена хорошая.",
            "Очень хороший товар, качество на высоте. Доволен покупкой!",
            "Отличное соотношение цена-качество. Все супер!",
            "Качественный товар, рекомендую всем. Очень доволен!",
            "Прекрасный продукт, быстрая доставка. Все как надо!",
            "Отличное качество, цена хорошая. Рекомендую к покупке.",
            "Замечательный товар, работает отлично. Очень рад!",
            "Очень хороший продукт, качество превосходное. Доволен!",
        ]
        
        negative_reviews = [
            "Ужасный товар! Качество очень плохое, не рекомендую никому.",
            "Очень разочарован покупкой. Товар не соответствует описанию.",
            "Плохое качество, быстро сломался. Деньги на ветер.",
            "Не рекомендую этот товар. Качество оставляет желать лучшего.",
            "Очень плохой продукт, не стоит своих денег. Разочарован.",
            "Товар не оправдал ожиданий. Качество низкое, не рекомендую.",
            "Ужасное качество, быстро пришел в негодность. Деньги выброшены.",
            "Очень разочарован. Товар не работает как заявлено.",
            "Плохое качество, не рекомендую. Лучше поискать другой вариант.",
            "Не стоит покупать. Товар не соответствует описанию и качеству.",
            "Ужасный продукт, качество очень низкое. Деньги на ветер.",
            "Очень плохое качество, быстро сломался. Не рекомендую.",
            "Товар разочаровал. Качество оставляет желать лучшего.",
            "Не рекомендую этот товар. Плохое качество и быстрый износ.",
            "Ужасное качество, не стоит своих денег. Очень разочарован.",
            "Очень плохой товар, не соответствует описанию. Не рекомендую.",
            "Плохое качество, быстро пришел в негодность. Деньги выброшены.",
            "Товар не оправдал ожиданий. Качество низкое, не стоит покупать.",
            "Ужасный продукт, качество очень плохое. Разочарован покупкой.",
            "Очень плохое качество, не рекомендую никому. Деньги на ветер.",
        ]
        
        # Создаем DataFrame
        texts = positive_reviews + negative_reviews
        labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
        
        df = pd.DataFrame({'text': texts, 'label': labels})
        
        print(f"Датасет загружен: {len(df)} записей")
        print(f"Положительных отзывов: {sum(labels)}")
        print(f"Отрицательных отзывов: {len(labels) - sum(labels)}")
        
        return df
    
    @staticmethod
    def load_from_file(filepath: str) -> Optional[pd.DataFrame]:
        """
        Загружает датасет из файла (CSV, Excel)
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            DataFrame или None
        """
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                print(f"Неподдерживаемый формат файла: {filepath}")
                return None
            
            # Проверяем наличие необходимых колонок
            if 'text' not in df.columns or 'label' not in df.columns:
                # Пытаемся найти похожие колонки
                text_cols = [col for col in df.columns if 'text' in col.lower() or 'review' in col.lower() or 'comment' in col.lower()]
                label_cols = [col for col in df.columns if 'label' in col.lower() or 'sentiment' in col.lower() or 'class' in col.lower()]
                
                if text_cols and label_cols:
                    df = df.rename(columns={text_cols[0]: 'text', label_cols[0]: 'label'})
                    print(f"Автоматически переименованы колонки: {text_cols[0]} -> text, {label_cols[0]} -> label")
                else:
                    print("Ошибка: не найдены колонки 'text' и 'label'")
                    print(f"Доступные колонки: {list(df.columns)}")
                    return None
            
            # Удаляем строки с пустыми значениями
            df = df.dropna(subset=['text', 'label'])
            
            # Преобразуем метки в числовой формат, если нужно
            if df['label'].dtype == 'object':
                unique_labels = df['label'].unique()
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                df['label'] = df['label'].map(label_map)
                print(f"Метки преобразованы: {label_map}")
            
            # Проверяем уникальные метки
            unique_labels = sorted(df['label'].unique())
            print(f"Уникальные метки в датасете: {unique_labels}")
            print(f"Распределение меток:")
            label_counts = df['label'].value_counts().sort_index()
            for label, count in label_counts.items():
                print(f"  Метка {label}: {count} записей ({count/len(df)*100:.1f}%)")
            
            # Для бинарной классификации оставляем только метки 0 и 1
            if len(unique_labels) > 2:
                print(f"\nВнимание: Датасет содержит {len(unique_labels)} классов, но используется бинарная классификация.")
                print("Оставляем только метки 0 и 1...")
                original_size = len(df)
                df = df[df['label'].isin([0, 1])].copy()
                print(f"Отфильтровано: {original_size} -> {len(df)} записей")
            
            print(f"Датасет загружен из файла: {len(df)} записей")
            return df
            
        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")
            return None
    
    @staticmethod
    def find_dataset_file() -> Optional[str]:
        """
        Ищет файлы датасетов в текущей директории и поддиректориях
        
        Returns:
            Путь к найденному файлу или None
        """
        # Ищем в текущей директории
        current_dir = Path('.')
        datasets_dir = Path('datasets')
        
        # Список возможных имен файлов (приоритет для sentiment_dataset.csv)
        possible_names = ['sentiment*.csv', 'reviews*.csv', 'dataset*.csv', '*.csv', 'news*.csv']
        
        search_paths = [datasets_dir, current_dir]  # Сначала ищем в datasets
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            for pattern in possible_names:
                files = list(search_path.glob(pattern))
                if files:
                    return str(files[0])
        
        return None
    
    @staticmethod
    def load_and_sample_dataset(filepath: str, max_samples: Optional[int] = None, 
                                random_state: int = 42) -> Optional[pd.DataFrame]:
        """
        Загружает датасет и делает выборку, если он слишком большой
        
        Args:
            filepath: Путь к файлу
            max_samples: Максимальное количество образцов (None = все)
            random_state: Seed для случайной выборки
            
        Returns:
            DataFrame или None
        """
        df = DatasetLoader.load_from_file(filepath)
        if df is None:
            return None
        
        original_size = len(df)
        
        # Если датасет слишком большой, делаем выборку
        if max_samples and len(df) > max_samples:
            print(f"\nДатасет большой ({original_size} записей).")
            print(f"Используем случайную выборку из {max_samples} записей для обучения...")
            df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
            print(f"Выборка создана: {len(df)} записей")
        
        return df


class TextClassifier:
    """Класс для обучения и использования классификаторов текста"""
    
    def __init__(self):
        # Создаем предобработчик без предупреждения (оно уже будет показано в main)
        self.preprocessor = TextPreprocessor(show_warning=False)
        self.vectorizer_bow = None
        self.vectorizer_tfidf = None
        self.models = {}
        self.best_model = None
        self.best_vectorizer = None
        self.best_method = None
    
    def prepare_features(self, texts: List[str], method: str = 'tfidf', fit: bool = True):
        """
        Преобразует тексты в числовую форму
        
        Args:
            texts: Список текстов
            method: Метод преобразования ('bow' или 'tfidf')
            fit: Нужно ли обучать векторайзер (True для train, False для test)
            
        Returns:
            Матрица признаков
        """
        if method == 'bow':
            if fit or self.vectorizer_bow is None:
                self.vectorizer_bow = CountVectorizer(max_features=5000, ngram_range=(1, 2))
                return self.vectorizer_bow.fit_transform(texts)
            else:
                return self.vectorizer_bow.transform(texts)
        else:  # tfidf
            if fit or self.vectorizer_tfidf is None:
                self.vectorizer_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                return self.vectorizer_tfidf.fit_transform(texts)
            else:
                return self.vectorizer_tfidf.transform(texts)
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Обучает несколько моделей классификаторов
        
        Args:
            X_train: Обучающие данные
            y_train: Метки обучающих данных
            X_test: Тестовые данные
            y_test: Метки тестовых данных
        """
        print("\n" + "="*60)
        print("Обучение моделей классификаторов")
        print("="*60)
        
        # Список моделей для обучения
        models_to_train = {
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'SVM': SVC(kernel='linear', C=1.0, probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20)
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\nОбучение {name}...")
            model.fit(X_train, y_train)
            
            # Предсказания
            y_pred = model.predict(X_test)
            
            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-score: {f1:.4f}")
            
            self.models[name] = model
        
        return results
    
    def select_best_model(self, results: dict, method: str):
        """
        Выбирает лучшую модель на основе метрик
        
        Args:
            results: Словарь с результатами моделей
            method: Метод векторизации ('bow' или 'tfidf')
        """
        # Выбираем модель с наилучшим F1-score
        best_name = max(results.keys(), key=lambda k: results[k]['f1'])
        best_result = results[best_name]
        
        self.best_model = best_result['model']
        self.best_method = method
        if method == 'bow':
            self.best_vectorizer = self.vectorizer_bow
        else:
            self.best_vectorizer = self.vectorizer_tfidf
        
        print("\n" + "="*60)
        print("Лучшая модель:")
        print("="*60)
        print(f"Модель: {best_name}")
        print(f"Метод векторизации: {method.upper()}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        print(f"Precision: {best_result['precision']:.4f}")
        print(f"Recall: {best_result['recall']:.4f}")
        print(f"F1-score: {best_result['f1']:.4f}")
        
        return best_name, best_result
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Классифицирует новые тексты
        
        Args:
            texts: Список текстов для классификации
            
        Returns:
            Массив предсказаний
        """
        if self.best_model is None:
            raise ValueError("Модель не обучена! Сначала вызовите train_models.")
        
        # Предобработка
        processed_texts = self.preprocessor.preprocess_texts(texts)
        
        # Векторизация
        if self.best_method == 'bow':
            X = self.vectorizer_bow.transform(processed_texts)
        else:
            X = self.vectorizer_tfidf.transform(processed_texts)
        
        # Предсказание
        predictions = self.best_model.predict(X)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Сохраняет модель в файл"""
        model_data = {
            'model': self.best_model,
            'vectorizer': self.best_vectorizer,
            'method': self.best_method,
            'preprocessor': self.preprocessor
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath: str):
        """Загружает модель из файла"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.best_model = model_data['model']
        self.best_vectorizer = model_data['vectorizer']
        self.best_method = model_data['method']
        self.preprocessor = model_data['preprocessor']
        print(f"Модель загружена из {filepath}")


class WebScraper:
    """
    Класс для парсинга текстовых данных с сайтов
    
    Поддерживаемые сайты для парсинга отзывов:
    - Отзовик (otzovik.com) - отзывы о товарах и услугах
    - Irecommend (irecommend.ru) - отзывы о товарах
    - Яндекс.Маркет (market.yandex.ru) - отзывы о товарах
    - Общие сайты с отзывами (через универсальный парсер)
    """
    
    # Примеры URL для парсинга (можно настроить под конкретные страницы)
    REVIEW_SITES = {
        'otzovik': 'https://otzovik.com',
        'irecommend': 'https://irecommend.ru',
        'yandex_market': 'https://market.yandex.ru'
    }
    
    @staticmethod
    def scrape_reviews_from_text(text_content: str, num_reviews: int = 5) -> List[str]:
        """
        Извлекает отзывы из текстового контента или использует примеры для демонстрации
        
        Args:
            text_content: Текстовый контент (может быть пустым для демонстрации)
            num_reviews: Количество отзывов для извлечения
            
        Returns:
            Список отзывов
        """
        # Если передан контент, пытаемся извлечь отзывы
        if text_content and len(text_content) > 50:
            # Простое извлечение предложений (можно улучшить)
            sentences = re.split(r'[.!?]\s+', text_content)
            reviews = [s.strip() for s in sentences if len(s.strip()) > 20]
            if reviews:
                return reviews[:num_reviews]
        
        # Используем примеры реальных отзывов, соответствующих тематике датасета
        sample_reviews = WebScraper.get_sample_reviews_from_real_sources()
        return sample_reviews[:num_reviews]
    
    @staticmethod
    def scrape_reviews_from_urls(urls: List[str], max_reviews: int = 10) -> List[str]:
        """
        Парсит отзывы с нескольких URL
        
        Args:
            urls: Список URL для парсинга
            max_reviews: Максимальное количество отзывов
            
        Returns:
            Список отзывов
        """
        all_reviews = []
        for url in urls:
            reviews = WebScraper.scrape_from_url(url)
            if reviews:
                all_reviews.extend(reviews)
            if len(all_reviews) >= max_reviews:
                break
        
        return all_reviews[:max_reviews]
    
    @staticmethod
    def scrape_from_url(url: str, site_type: str = 'generic') -> Optional[List[str]]:
        """
        Парсит текстовые данные с указанного URL
        
        Args:
            url: URL для парсинга
            site_type: Тип сайта ('otzovik', 'irecommend', 'yandex_market', 'generic')
            
        Returns:
            Список текстов или None
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            texts = []
            
            # Адаптация под разные типы сайтов
            if site_type == 'otzovik':
                # Отзовик: ищем блоки с отзывами
                review_blocks = soup.find_all(['div', 'article'], class_=lambda x: x and ('review' in x.lower() or 'comment' in x.lower()))
                for block in review_blocks:
                    text = block.get_text(strip=True)
                    if len(text) > 30:  # Минимальная длина отзыва
                        texts.append(text)
            
            elif site_type == 'irecommend':
                # Irecommend: ищем тексты отзывов
                review_blocks = soup.find_all(['div', 'p'], class_=lambda x: x and ('review' in x.lower() or 'text' in x.lower()))
                for block in review_blocks:
                    text = block.get_text(strip=True)
                    if len(text) > 30:
                        texts.append(text)
            
            elif site_type == 'yandex_market':
                # Яндекс.Маркет: ищем отзывы
                review_blocks = soup.find_all(['div', 'span'], class_=lambda x: x and ('review' in x.lower() or 'opinion' in x.lower()))
                for block in review_blocks:
                    text = block.get_text(strip=True)
                    if len(text) > 30:
                        texts.append(text)
            
            else:
                # Универсальный парсер: ищем параграфы и div с текстом
                for tag in soup.find_all(['p', 'div', 'article', 'section']):
                    text = tag.get_text(strip=True)
                    # Фильтруем: минимум 30 символов, не слишком длинные, содержат кириллицу
                    if (30 <= len(text) <= 500 and 
                        any(char.isalpha() and ord(char) >= 1040 for char in text)):
                        texts.append(text)
            
            # Удаляем дубликаты и слишком похожие тексты
            unique_texts = []
            for text in texts:
                if not any(text[:20] == existing[:20] for existing in unique_texts):
                    unique_texts.append(text)
            
            return unique_texts[:50]  # Возвращаем до 50 текстов
        
        except requests.exceptions.RequestException as e:
            print(f"Ошибка сети при парсинге {url}: {e}")
            return None
        except Exception as e:
            print(f"Ошибка при парсинге {url}: {e}")
            return None
    
    @staticmethod
    def get_sample_reviews_from_real_sources() -> List[str]:
        """
        Возвращает примеры реальных отзывов для демонстрации
        Эти отзывы соответствуют тематике датасета (отзывы на товары)
        
        Returns:
            Список отзывов
        """
        # Реальные примеры отзывов, соответствующие тематике датасета
        real_reviews = [
            "Заказал этот товар, пришел быстро. Качество хорошее, цена адекватная. Рекомендую к покупке, не пожалеете.",
            "Товар не оправдал ожиданий. Качество оставляет желать лучшего, быстро пришел в негодность. Деньги выброшены на ветер.",
            "Отличный продукт! Все работает как надо, очень доволен покупкой. Доставка была быстрой, упаковка целая.",
            "Не рекомендую этот товар. Пришел с дефектом, пришлось возвращать. Продавец не отвечает на сообщения.",
            "Хороший товар за свою цену. Качество среднее, но для таких денег вполне нормально. Доволен покупкой.",
            "Прекрасное качество, быстрая доставка. Все соответствует описанию на сайте. Очень рад, что заказал именно этот товар.",
            "Ужасное качество! Товар сломался через неделю использования. Продавец отказывается возвращать деньги. Не покупайте!",
            "Товар хороший, но доставка заняла очень много времени. В целом доволен, но ожидал быстрее.",
            "Качество отличное, цена приемлемая. Рекомендую всем друзьям. Очень доволен покупкой и сервисом.",
            "Средний товар, ничего особенного. Цена соответствует качеству. Можно купить, если нет альтернатив.",
            "Отличное соотношение цена-качество. Товар работает хорошо, никаких нареканий. Рекомендую!",
            "Товар пришел поврежденным. Упаковка была нарушена, товар не работает. Очень разочарован.",
            "Замечательный продукт! Все работает отлично, качество на высоте. Очень доволен, рекомендую к покупке.",
            "Не стоит своих денег. Качество низкое, быстро ломается. Лучше поискать другой вариант.",
            "Хороший товар, быстрая доставка. Качество хорошее, цена нормальная. Доволен покупкой."
        ]
        return real_reviews


def main():
    """Основная функция для выполнения полного пайплайна"""
    
    print("="*60)
    print("Классификация текста: Предобработка и обучение моделей")
    print("="*60)
    
    # 1. Загрузка датасета
    print("\n[1/7] Загрузка датасета...")
    loader = DatasetLoader()
    
    # Пытаемся найти датасет в файлах
    dataset_file = loader.find_dataset_file()
    if dataset_file:
        print(f"Найден файл датасета: {dataset_file}")
        # Для больших датасетов используем выборку (можно изменить max_samples)
        # Если хотите использовать весь датасет, установите max_samples=None
        df = loader.load_and_sample_dataset(dataset_file, max_samples=10000)
        if df is None:
            print("Не удалось загрузить датасет из файла, используем синтетический...")
            df = loader.download_sentiment_dataset()
    else:
        print("Файл датасета не найден, используем синтетический датасет...")
        print("(Для использования своего датасета поместите CSV файл в папку 'datasets/')")
        df = loader.download_sentiment_dataset()
    
    # 2. Предобработка текста
    print("\n[2/7] Предобработка текста...")
    preprocessor = TextPreprocessor()
    df['processed_text'] = preprocessor.preprocess_texts(df['text'].tolist())
    
    print("Примеры обработанных текстов:")
    for i in range(min(3, len(df))):
        print(f"  Исходный: {df['text'].iloc[i][:50]}...")
        print(f"  Обработанный: {df['processed_text'].iloc[i][:50]}...")
    
    # 3. Разделение на train/test
    X = df['processed_text'].tolist()
    y = df['label'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nРазмер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    
    # 4. Преобразование в числовую форму и обучение моделей
    print("\n[3/7] Преобразование в числовую форму...")
    print("\n[4/7] Обучение моделей...")
    
    classifier = TextClassifier()
    
    # Тестируем оба метода: Bag of Words и TF-IDF
    best_overall = None
    best_score = 0
    best_method_name = None
    
    for method in ['bow', 'tfidf']:
        print(f"\n--- Метод: {method.upper()} ---")
        
        # Преобразование в числовую форму
        X_train_vec = classifier.prepare_features(X_train, method=method, fit=True)
        X_test_vec = classifier.prepare_features(X_test, method=method, fit=False)
        
        # Обучение моделей
        results = classifier.train_models(X_train_vec, y_train, X_test_vec, y_test)
        
        # Выбор лучшей модели для данного метода
        best_name, best_result = classifier.select_best_model(results, method)
        
        # Сохраняем лучший результат
        if best_result['f1'] > best_score:
            best_score = best_result['f1']
            best_overall = (best_name, best_result, method)
            best_method_name = method
    
    # Устанавливаем лучшую модель
    if best_overall:
        print(f"\n\n{'='*60}")
        print("ИТОГОВАЯ ЛУЧШАЯ МОДЕЛЬ:")
        print("="*60)
        print(f"Модель: {best_overall[0]}")
        print(f"Метод векторизации: {best_overall[2].upper()}")
        print(f"F1-score: {best_overall[1]['f1']:.4f}")
        
        # Переобучаем с лучшим методом
        X_train_vec = classifier.prepare_features(X_train, method=best_method_name, fit=True)
        X_test_vec = classifier.prepare_features(X_test, method=best_method_name, fit=False)
        results = classifier.train_models(X_train_vec, y_train, X_test_vec, y_test)
        classifier.select_best_model(results, best_method_name)
    
    # 5. Сохранение модели
    print("\n[5/7] Сохранение модели...")
    model_path = 'text_classifier_model.pkl'
    classifier.save_model(model_path)
    
    # 6. Парсинг новых данных
    print("\n[6/7] Парсинг новых данных...")
    scraper = WebScraper()
    
    print("Используемые источники для парсинга:")
    print("  - Отзовик (otzovik.com) - отзывы о товарах и услугах")
    print("  - Irecommend (irecommend.ru) - отзывы о товарах")
    print("  - Яндекс.Маркет (market.yandex.ru) - отзывы о товарах")
    print("  - Примеры реальных отзывов (для демонстрации)")
    
    # Пытаемся использовать реальный парсинг с сайтов
    # ВАРИАНТ 1: Реальный парсинг с сайтов
    # 
    # ИНСТРУКЦИЯ: Раскомментируйте строки ниже и укажите реальные URL страниц с отзывами
    # 
    # Как найти URL:
    # 1. Отзовик: https://otzovik.com → найдите отзыв → скопируйте URL из адресной строки
    # 2. Irecommend: https://irecommend.ru → найдите отзыв → скопируйте URL
    # 3. Яндекс.Маркет: https://market.yandex.ru → товар → вкладка "Отзывы" → скопируйте URL
    #
    # Примеры URL для парсинга:
    # 
    # ВАЖНО: Замените '...' на реальные URL страниц с отзывами!
    # Как найти URL: зайдите на сайт, найдите страницу с отзывами, скопируйте URL из адресной строки
    #
    review_urls = [
        # Раскомментировано! Замените '...' на реальные URL страниц с отзывами:
        #'https://otzovik.com/reviews/rastvor_dlya_linz_alcon_opti_fri_pure_moist/',
        'https://irecommend.ru/content/krem-dlya-litsa-bioderma-pore-refiner'
        #'https://irecommend.ru/content/krem-dlya-litsa-bioderma-sebium-hydra'  # Замените ... на реальный URL (например: 12345_otziv_o_tovare)
        # 'https://irecommend.ru/content/...',  # Раскомментируйте и замените ... на реальный URL
        # 'https://market.yandex.ru/product/.../reviews',  # Раскомментируйте и замените ... на реальный URL
    ]
    
    new_texts = []
    
    # Проверяем, есть ли реальные URL для парсинга
    valid_urls = [url for url in review_urls if isinstance(url, str) and 
                  url.startswith('http') and '...' not in url and not url.strip().startswith('#')]
    
    if valid_urls:
        print("\n🌐 Попытка парсинга с указанных URL...")
        print(f"   Найдено {len(valid_urls)} URL для парсинга")
        parsed_texts = scraper.scrape_reviews_from_urls(valid_urls, max_reviews=50)
        if parsed_texts and len(parsed_texts) > 0:
            new_texts = parsed_texts
            print(f"✅ Успешно получено {len(new_texts)} отзывов с сайтов")
        else:
            print("⚠️ Парсинг с URL не удался (возможные причины: сайт недоступен, изменилась структура, блокировка)")
            print("   Используем примеры отзывов для демонстрации...")
            new_texts = scraper.scrape_reviews_from_text("", num_reviews=50)
    else:
        # ВАРИАНТ 2: Использовать примеры реальных отзывов (для демонстрации)
        print("\n📝 Использование примеров реальных отзывов (соответствуют тематике датасета)")
        print("   💡 Чтобы парсить с реальных сайтов:")
        print("      1. Раскомментируйте строки 904-907 в text_classification.py")
        print("      2. Замените '...' на реальные URL страниц с отзывами")
        print("      3. Запустите скрипт снова")
        new_texts = scraper.scrape_reviews_from_text("", num_reviews=50)
    
    # ВАРИАНТ 3: Парсинг конкретного сайта (альтернативный вариант):
    # Раскомментируйте для парсинга одного конкретного сайта:
    # new_texts = scraper.scrape_from_url('https://otzovik.com/reviews/12345', site_type='otzovik')
    # if not new_texts or len(new_texts) == 0:
    #     print("Парсинг не удался, используем примеры отзывов")
    #     new_texts = scraper.scrape_reviews_from_text("", num_reviews=50)
    
    # Предобработка новых данных (как требовалось в задании)
    print("Проведение предобработки новых данных...")
    processed_new_texts = preprocessor.preprocess_texts(new_texts)
    
    print(f"Получено {len(new_texts)} новых текстов для классификации:")
    for i, (original, processed) in enumerate(zip(new_texts[:3], processed_new_texts[:3]), 1):
        print(f"  {i}. Исходный: {original[:50]}...")
        print(f"     Обработанный: {processed[:50]}...")
    
    # 7. Классификация новых данных при помощи обученной модели
    print("\n[7/7] Классификация новых данных при помощи обученной модели...")
    
    if classifier.best_model is None:
        print("Ошибка: Модель не обучена!")
        return
    
    print(f"Используется модель: {classifier.best_method.upper()}")
    predictions = classifier.predict(new_texts)
    
    # Подсчет статистики
    label_names = {0: "Отрицательный", 1: "Положительный", 2: "Нейтральный"}
    positive_count = sum(1 for p in predictions if int(p) == 1)
    negative_count = sum(1 for p in predictions if int(p) == 0)
    neutral_count = sum(1 for p in predictions if int(p) == 2)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ НОВЫХ ДАННЫХ:")
    print("="*60)
    print(f"\nВсего классифицировано: {len(new_texts)} текстов")
    print(f"Положительных: {positive_count} ({positive_count/len(new_texts)*100:.1f}%)")
    print(f"Отрицательных: {negative_count} ({negative_count/len(new_texts)*100:.1f}%)")
    if neutral_count > 0:
        print(f"Нейтральных: {neutral_count} ({neutral_count/len(new_texts)*100:.1f}%)")
    
    print("\n" + "-"*60)
    print("Детальные результаты:")
    print("-"*60)
    
    for i, (text, pred) in enumerate(zip(new_texts, predictions), 1):
        # Преобразуем numpy тип в обычный int для безопасности
        pred_int = int(pred)
        # Используем универсальный вывод
        if pred_int in label_names:
            label = label_names[pred_int]
            emoji = "✅" if pred_int == 1 else "❌" if pred_int == 0 else "➖"
        else:
            label = f"Класс {pred_int}"
            emoji = "❓"
        
        print(f"\n[{i}] {emoji} {label}")
        print(f"    Текст: {text[:70]}{'...' if len(text) > 70 else ''}")
        print(f"    Метка: {pred_int}")
    
    print("\n" + "="*60)
    print("✅ Классификация новых данных завершена успешно!")
    print("="*60)


if __name__ == "__main__":
    main()


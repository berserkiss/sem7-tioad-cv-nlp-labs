"""
Сравнение качества эмбеддингов

Этот скрипт загружает предобученные модели эмбеддингов (Word2Vec, FastText)
и сравнивает их эффективность на различных задачах:
- Поиск семантически схожих слов
- Выполнение аналогий
- Визуализация векторов

Использование:
    python embeddings_comparison.py

Зависимости:
    pip install gensim matplotlib scikit-learn numpy

ПРИНЦИПЫ РАБОТЫ ЭМБЕДДИНГОВ:

1. WORD2VEC:
   - Использует нейронную сеть для обучения представлений слов
   - Skip-gram: предсказывает контекстные слова по заданному слову
   - CBOW: предсказывает слово по контексту
   - Слова в похожих контекстах получают похожие векторы
   
   Схема: [слово] → [скрытый слой 300D] → [контекстные слова]
   
2. GLOVE:
   - Использует глобальную статистику корпуса (матрицу совместной встречаемости)
   - Комбинирует локальный контекст с глобальной статистикой
   - Минимизирует функцию потерь на основе частоты совместной встречаемости
   
   Схема: [корпус] → [матрица встречаемости] → [оптимизация] → [векторы]

3. СЕМАНТИЧЕСКАЯ БЛИЗОСТЬ:
   - Измеряется через косинусное сходство: cos(θ) = (A·B)/(||A||×||B||)
   - Диапазон: -1 (противоположные) до 1 (идентичные)
   - Чем больше значение, тем ближе слова семантически

4. АНАЛОГИИ:
   - Используют векторную арифметику: result = word3 + word2 - word1
   - Пример: king - man + woman = queen
   - Находят ближайший вектор к результату вычисления
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    from gensim.models import Word2Vec, FastText, KeyedVectors
    from gensim import downloader as api
except ImportError:
    print("Ошибка: gensim не установлен. Установите: pip install gensim")
    exit(1)


class EmbeddingModel:
    """Класс для работы с моделью эмбеддингов"""
    
    def __init__(self, name: str, model=None):
        """
        Инициализация модели
        
        Args:
            name: Название модели
            model: Загруженная модель gensim
        """
        self.name = name
        self.model = model
        # Безопасное получение размера словаря (может быть медленным для больших моделей)
        try:
            if model:
                if hasattr(model, 'key_to_index'):
                    self.vocab_size = len(model.key_to_index)
                elif hasattr(model, 'vocab'):
                    self.vocab_size = len(model.vocab)
                elif hasattr(model, '__len__'):
                    self.vocab_size = len(model)
                else:
                    self.vocab_size = 0
            else:
                self.vocab_size = 0
        except Exception:
            self.vocab_size = 0  # Если не удалось определить, установим 0
    
    def get_vector(self, word: str):
        """Получить вектор слова"""
        try:
            return self.model[word]
        except KeyError:
            return None
    
    def word_exists(self, word: str) -> bool:
        """Проверить, существует ли слово в модели"""
        return word in self.model.key_to_index if hasattr(self.model, 'key_to_index') else word in self.model
    
    def find_similar_words(self, word: str, topn: int = 10):
        """
        Найти семантически схожие слова
        
        Алгоритм:
        1. Получить вектор слова из модели
        2. Вычислить косинусное сходство со всеми словами словаря
        3. Отсортировать по убыванию сходства
        4. Вернуть топ-N самых похожих слов
        
        Формула косинусного сходства:
        similarity = (word_vec · other_vec) / (||word_vec|| × ||other_vec||)
        
        Args:
            word: Искомое слово
            topn: Количество похожих слов
            
        Returns:
            Список кортежей (слово, схожесть)
        """
        if not self.word_exists(word):
            return []
        
        try:
            similar = self.model.most_similar(word, topn=topn)
            return similar
        except Exception as e:
            print(f"Ошибка при поиске похожих слов для '{word}': {e}")
            return []
    
    def analogy(self, word1: str, word2: str, word3: str, topn: int = 5):
        """
        Выполнить аналогию: word1 - word2 = word3 - ?
        
        Принцип работы (векторная арифметика):
        result_vector = word3_vector + word2_vector - word1_vector
        result = найти ближайший вектор к result_vector
        
        Пример: "king - man = queen - ?" -> "woman"
        
        Схема:
        king_vec - man_vec = [разница по полу]
        queen_vec + [разница по полу] = woman_vec
        
        Args:
            word1: Первое слово
            word2: Второе слово
            word3: Третье слово
            topn: Количество вариантов ответа
            
        Returns:
            Список кортежей (слово, схожесть)
        """
        words = [word1, word2, word3]
        for w in words:
            if not self.word_exists(w):
                return []
        
        try:
            # Вычисляем: word3 + (word2 - word1)
            result = self.model.most_similar(
                positive=[word3, word2],
                negative=[word1],
                topn=topn
            )
            return result
        except Exception as e:
            print(f"Ошибка при выполнении аналогии: {e}")
            return []
    
    def get_vectors_for_words(self, words: list):
        """
        Получить векторы для списка слов
        
        Args:
            words: Список слов
            
        Returns:
            Словарь {слово: вектор}
        """
        vectors = {}
        for word in words:
            if self.word_exists(word):
                vectors[word] = self.get_vector(word)
        return vectors


class EmbeddingComparator:
    """Класс для сравнения моделей эмбеддингов"""
    
    def __init__(self):
        """Инициализация компаратора"""
        self.models = {}
    
    def load_word2vec(self, model_path: str = None):
        """
        Загрузить модель Word2Vec
        
        Args:
            model_path: Путь к модели (если None, загрузит из gensim)
        """
        print("\n[1/2] Загрузка модели Word2Vec...")
        
        if model_path and os.path.exists(model_path):
            print(f"   Загрузка из файла: {model_path}")
            try:
                model = KeyedVectors.load_word2vec_format(model_path, binary=True)
            except Exception as e:
                print(f"   Ошибка загрузки из файла: {e}")
                return None
        else:
            print("   Загрузка предобученной модели 'word2vec-google-news-300'...")
            # Проверяем, есть ли модель в кэше
            from_cache = False
            try:
                model_info = api.info("word2vec-google-news-300")
                if model_info.get('file_name') and os.path.exists(model_info.get('file_name', '')):
                    from_cache = True
                    print("   ✅ Модель найдена в кэше")
                    print("   ⏳ Загрузка в память (~1.6 ГБ, может занять 30-60 секунд)...")
                    print("   💡 Это нормально - большие модели требуют времени для загрузки в RAM")
                else:
                    print("   ⚠️  ВНИМАНИЕ: Первая загрузка может занять 3-10 минут и ~1.6 ГБ места!")
                    print("   💡 После первой загрузки модель будет браться из кэша")
            except:
                print("   ⚠️  ВНИМАНИЕ: Первая загрузка может занять 3-10 минут и ~1.6 ГБ места!")
                print("   💡 После первой загрузки модель будет браться из кэша")
            
            import time
            import threading
            
            start_time = time.time()
            loading_done = threading.Event()
            
            def show_progress():
                """Показывает точки во время загрузки"""
                dots = 0
                while not loading_done.is_set():
                    print(f"\r   ⏳ Загрузка{'...'[:dots%3+1]:<3}", end='', flush=True)
                    dots += 1
                    loading_done.wait(1.0)
                print()
            
            progress_thread = threading.Thread(target=show_progress, daemon=True)
            progress_thread.start()
            
            try:
                model = api.load("word2vec-google-news-300")
            except KeyboardInterrupt:
                loading_done.set()
                progress_thread.join(timeout=0.5)
                print(f"\n   ⚠️  Загрузка прервана пользователем (Ctrl+C)")
                print(f"   Необходима хотя бы одна модель для работы!")
                return None
            except Exception as e:
                loading_done.set()
                progress_thread.join(timeout=0.5)
                print(f"   Ошибка загрузки: {e}")
                print("   Попробуйте загрузить модель вручную:")
                print("   wget https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM")
                return None
            finally:
                loading_done.set()
                progress_thread.join(timeout=0.5)
            
            load_time = time.time() - start_time
            if from_cache:
                print(f"   ✅ Загружено из кэша за {load_time:.1f} секунд")
            else:
                print(f"   ✅ Модель скачана и загружена за {load_time:.1f} секунд")
        
        embedding_model = EmbeddingModel("Word2Vec (Google News)", model)
        self.models["word2vec"] = embedding_model
        print(f"   ✅ Загружено слов: {embedding_model.vocab_size:,}")
        return embedding_model
    
    def load_fasttext(self, model_path: str = None):
        """
        Загрузить модель FastText
        
        Args:
            model_path: Путь к модели (если None, загрузит из gensim)
        """
        print("\n[2/2] Загрузка второй модели (FastText или GloVe)...")
        
        if model_path and os.path.exists(model_path):
            print(f"   Загрузка из файла: {model_path}")
            try:
                # Пробуем загрузить как KeyedVectors (для .vec файлов)
                if model_path.endswith('.vec'):
                    model = KeyedVectors.load_word2vec_format(model_path, binary=False)
                    embedding_model = EmbeddingModel("FastText (из файла)", model)
                else:
                    model = FastText.load_fasttext_format(model_path)
                    embedding_model = EmbeddingModel("FastText (из файла)", model)
                self.models["fasttext"] = embedding_model
                print(f"   ✅ Загружено слов: {embedding_model.vocab_size:,}")
                return embedding_model
            except Exception as e:
                print(f"   ❌ Ошибка загрузки из файла: {e}")
                return None
        else:
            # Пробуем несколько вариантов моделей (начинаем с ЛЕГКИХ!)
            # Порядок важен: сначала легкие GloVe, потом тяжелый FastText
            alternative_models = [
                ("glove-twitter-200", "GloVe (Twitter, 200d)", "~400 МБ"),  # Самая легкая и быстрая
                ("glove-wiki-gigaword-50", "GloVe (Wiki, 50d)", "~100 МБ"),  # Очень легкая
                ("glove-wiki-gigaword-100", "GloVe (Wiki, 100d)", "~200 МБ"),  # Легкая
                ("glove-wiki-gigaword-300", "GloVe (Wiki, 300d)", "~400 МБ"),  # Средняя
                ("fasttext-wiki-news-subwords-300", "FastText (Wiki News)", "~1.5 ГБ"),  # Тяжелая (последняя)
            ]
            
            print("\n   📋 Доступные модели (от легких к тяжелым):")
            for i, (_, display_name, size) in enumerate(alternative_models, 1):
                print(f"      {i}. {display_name} - {size}")
            
            print("\n   🔄 Начинаем загрузку с самых легких моделей...")
            print("   💡 Совет: Нажмите Ctrl+C, чтобы пропустить текущую модель и попробовать следующую")
            print("   💡 Примечание: Модели скачиваются только один раз, затем берутся из кэша!\n")
            
            model_loaded = False
            last_error = None
            
            for idx, (model_name, model_display_name, size) in enumerate(alternative_models, 1):
                print(f"   [{idx}/{len(alternative_models)}] Попытка: {model_display_name} ({size})")
                
                try:
                    import sys
                    import time
                    sys.stdout.flush()  # Принудительно выводим буфер
                    
                    # Проверяем, есть ли модель в кэше (более надежная проверка)
                    from_cache = False
                    try:
                        model_info = api.info(model_name)
                        # Проверяем несколько способов определения наличия модели в кэше
                        file_path = None
                        if 'file_name' in model_info:
                            file_path = model_info['file_name']
                        elif 'files' in model_info and len(model_info['files']) > 0:
                            file_path = model_info['files'][0].get('path', '')
                        
                        # Также проверяем базовую директорию кэша
                        if not file_path or not os.path.exists(file_path):
                            # Пробуем найти в стандартной директории кэша
                            cache_base = os.path.expanduser('~/.gensim')
                            if os.name == 'nt':  # Windows
                                cache_base = os.path.join(os.path.expanduser('~'), 'gensim-data')
                            
                            # Ищем модель по имени в кэше (упрощенная проверка без os.walk для скорости)
                            if os.path.exists(cache_base):
                                # Проверяем только первый уровень директорий для скорости
                                try:
                                    dirs = [d for d in os.listdir(cache_base) if os.path.isdir(os.path.join(cache_base, d))]
                                    model_dir_name = model_name.replace('-', '_')
                                    if any(model_dir_name in d or model_name.split('-')[0] in d for d in dirs):
                                        from_cache = True
                                except:
                                    pass  # Если не удалось проверить, предполагаем что нет в кэше
                        else:
                            from_cache = True
                        
                        if from_cache:
                            print(f"   ✅ Модель найдена в кэше")
                            print(f"   ⏳ Загрузка в память ({size}, может занять 30-120 секунд)...")
                            print(f"   💡 Это нормально - большие модели требуют времени для загрузки в RAM")
                        else:
                            print(f"   ⏳ Первая загрузка модели (может занять время)...")
                            print(f"   💡 После первой загрузки модель будет браться из кэша")
                    except Exception as e:
                        # Если проверка не удалась, предполагаем что это первая загрузка
                        print(f"   ⏳ Загрузка модели...")
                        # Можно раскомментировать для отладки:
                        # print(f"   (Отладка: {str(e)[:50]})")
                    
                    sys.stdout.flush()
                    start_time = time.time()
                    
                    # Добавляем индикатор прогресса во время загрузки
                    import threading
                    loading_done = threading.Event()
                    
                    def show_progress():
                        """Показывает точки во время загрузки"""
                        dots = 0
                        while not loading_done.is_set():
                            print(f"\r   ⏳ Загрузка{'...'[:dots%3+1]:<3}", end='', flush=True)
                            dots += 1
                            loading_done.wait(1.0)  # Обновляем каждую секунду
                        print()  # Новая строка после завершения
                    
                    # Запускаем индикатор прогресса в отдельном потоке
                    progress_thread = threading.Thread(target=show_progress, daemon=True)
                    progress_thread.start()
                    
                    try:
                        model = api.load(model_name)
                        load_time = time.time() - start_time
                        
                        if from_cache:
                            print(f"   ✅ Загружено из кэша за {load_time:.1f} секунд")
                        else:
                            print(f"   ✅ Модель скачана и загружена за {load_time:.1f} секунд")
                        sys.stdout.flush()
                    except KeyboardInterrupt:
                        loading_done.set()
                        progress_thread.join(timeout=0.5)
                        print(f"\n   ⚠️  Загрузка прервана пользователем (Ctrl+C)")
                        if idx < len(alternative_models):
                            print(f"   Пробуем следующую модель...\n")
                            continue
                        else:
                            print(f"   Это была последняя модель. Продолжаем работу только с Word2Vec...")
                            return None
                    except Exception as e:
                        loading_done.set()
                        progress_thread.join(timeout=0.5)
                        last_error = e
                        error_msg = str(e)
                        
                        # Анализируем тип ошибки
                        if "HTTP" in error_msg or "Connection" in error_msg or "timeout" in error_msg.lower():
                            print(f"   ⚠️  Проблема с сетью. Пробуем следующую модель...")
                        elif "404" in error_msg or "not found" in error_msg.lower():
                            print(f"   ⚠️  Модель не найдена. Пробуем следующую...")
                        elif "disk" in error_msg.lower() or "space" in error_msg.lower():
                            print(f"   ⚠️  Недостаточно места. Пробуем более легкую модель...")
                        else:
                            print(f"   ⚠️  Ошибка: {error_msg[:80]}... Пробуем следующую...")
                        
                        if idx < len(alternative_models):
                            print()  # Пустая строка для читаемости
                        continue
                    finally:
                        loading_done.set()  # Останавливаем индикатор
                        try:
                            progress_thread.join(timeout=0.5)  # Ждем завершения потока
                        except:
                            pass
                    
                    embedding_model = EmbeddingModel(model_display_name, model)
                    sys.stdout.flush()
                    
                    # Используем ключ "fasttext" для совместимости, даже если это GloVe
                    self.models["fasttext"] = embedding_model
                    print(f"   ✅ ✅ ✅ УСПЕХ! Загружена модель: {model_display_name}")
                    sys.stdout.flush()
                    
                    # Показываем размер словаря (уже вычислен в __init__)
                    if embedding_model.vocab_size > 0:
                        print(f"   ✅ Слов в словаре: {embedding_model.vocab_size:,}")
                    else:
                        print(f"   ✅ Модель загружена успешно!")
                    sys.stdout.flush()
                    
                    model_loaded = True
                    return embedding_model
                    
                except KeyboardInterrupt:
                    print(f"\n   ⚠️  Загрузка прервана пользователем (Ctrl+C)")
                    if idx < len(alternative_models):
                        print(f"   Пробуем следующую модель...\n")
                        continue
                    else:
                        print(f"   Это была последняя модель. Продолжаем работу только с Word2Vec...")
                        return None
                        
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    
                    # Анализируем тип ошибки
                    if "HTTP" in error_msg or "Connection" in error_msg or "timeout" in error_msg.lower():
                        print(f"   ⚠️  Проблема с сетью. Пробуем следующую модель...")
                    elif "404" in error_msg or "not found" in error_msg.lower():
                        print(f"   ⚠️  Модель не найдена. Пробуем следующую...")
                    elif "disk" in error_msg.lower() or "space" in error_msg.lower():
                        print(f"   ⚠️  Недостаточно места. Пробуем более легкую модель...")
                    else:
                        print(f"   ⚠️  Ошибка: {error_msg[:80]}... Пробуем следующую...")
                    
                    if idx < len(alternative_models):
                        print()  # Пустая строка для читаемости
                    continue
            
            if not model_loaded:
                print(f"\n   ❌ ❌ ❌ Не удалось загрузить ни одну из альтернативных моделей")
                if last_error:
                    print(f"\n   Последняя ошибка: {last_error}")
                print(f"\n   💡 РЕКОМЕНДАЦИИ:")
                print(f"   1. Проверьте интернет-соединение")
                print(f"   2. Убедитесь, что есть достаточно места на диске (минимум 2 ГБ)")
                print(f"   3. Попробуйте запустить скрипт позже")
                print(f"   4. Или скачайте модель вручную:")
                print(f"      - GloVe: https://nlp.stanford.edu/projects/glove/")
                print(f"      - FastText: https://fasttext.cc/docs/en/pretrained-vectors.html")
                print(f"      Затем: fasttext = comparator.load_fasttext(model_path='path/to/model.vec')")
                print(f"\n   ⚠️  Продолжаем работу только с Word2Vec...")
                return None
    
    def compare_similar_words(self, word: str, topn: int = 10):
        """
        Сравнить поиск похожих слов между моделями
        
        Args:
            word: Искомое слово
            topn: Количество похожих слов
        """
        print(f"\n{'='*70}")
        print(f"ПОИСК СЕМАНТИЧЕСКИ СХОЖИХ СЛОВ ДЛЯ: '{word}'")
        print(f"{'='*70}")
        
        for model_name, model in self.models.items():
            print(f"\n📊 {model.name}:")
            similar = model.find_similar_words(word, topn=topn)
            
            if similar:
                print(f"   Топ-{len(similar)} похожих слов:")
                for i, (similar_word, similarity) in enumerate(similar, 1):
                    print(f"   {i:2d}. {similar_word:20s} (схожесть: {similarity:.4f})")
            else:
                print(f"   ❌ Слово '{word}' не найдено в модели")
    
    def compare_analogies(self, word1: str, word2: str, word3: str, topn: int = 5):
        """
        Сравнить выполнение аналогий между моделями
        
        Args:
            word1: Первое слово
            word2: Второе слово
            word3: Третье слово
            topn: Количество вариантов ответа
        """
        print(f"\n{'='*70}")
        print(f"АНАЛОГИЯ: {word1} - {word2} = {word3} - ?")
        print(f"{'='*70}")
        
        for model_name, model in self.models.items():
            print(f"\n📊 {model.name}:")
            results = model.analogy(word1, word2, word3, topn=topn)
            
            if results:
                print(f"   Топ-{len(results)} вариантов ответа:")
                for i, (answer, similarity) in enumerate(results, 1):
                    print(f"   {i}. {answer:20s} (схожесть: {similarity:.4f})")
            else:
                missing_words = []
                for w in [word1, word2, word3]:
                    if not model.word_exists(w):
                        missing_words.append(w)
                if missing_words:
                    print(f"   ❌ Слова не найдены в модели: {', '.join(missing_words)}")
                else:
                    print(f"   ❌ Не удалось выполнить аналогию")
    
    def visualize_vectors(self, words: list, method: str = 'tsne', output_path: str = 'embeddings_visualization.png'):
        """
        Визуализировать векторы слов с помощью t-SNE или PCA
        
        Методы снижения размерности:
        
        1. PCA (Principal Component Analysis):
           - Находит главные компоненты (направления максимальной дисперсии)
           - Проецирует данные на первые 2 компоненты
           - Быстрый алгоритм (O(n²))
           
        2. t-SNE (t-Distributed Stochastic Neighbor Embedding):
           - Сохраняет локальную структуру данных
           - Минимизирует расхождение между распределениями в высоком и низком пространстве
           - Медленный алгоритм (O(n²) или выше)
           - Лучше показывает кластеры семантически похожих слов
        
        Схема:
        [Векторы 300D] → [Снижение размерности] → [Векторы 2D] → [График]
        
        Args:
            words: Список слов для визуализации
            method: Метод визуализации ('tsne' или 'pca')
            output_path: Путь для сохранения изображения
        """
        print(f"\n{'='*70}")
        print(f"ВИЗУАЛИЗАЦИЯ ВЕКТОРОВ ({method.upper()})")
        print(f"{'='*70}")
        
        # Определяем количество моделей
        n_models = len(self.models)
        if n_models == 0:
            print("❌ Нет загруженных моделей!")
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            print(f"\n📊 {model.name}:")
            
            # Получаем векторы для слов, которые есть в модели
            available_words = [w for w in words if model.word_exists(w)]
            if len(available_words) < 2:
                print(f"   ❌ Недостаточно слов для визуализации (найдено: {len(available_words)})")
                continue
            
            print(f"   Найдено слов в модели: {len(available_words)}/{len(words)}")
            
            # Собираем векторы
            vectors = []
            word_labels = []
            for word in available_words:
                vec = model.get_vector(word)
                if vec is not None:
                    vectors.append(vec)
                    word_labels.append(word)
            
            if len(vectors) < 2:
                print(f"   ❌ Недостаточно векторов для визуализации")
                continue
            
            vectors = np.array(vectors)
            
            # Применяем метод снижения размерности
            if method.lower() == 'tsne':
                print(f"   Применение t-SNE...")
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)-1))
            else:  # PCA
                print(f"   Применение PCA...")
                reducer = PCA(n_components=2, random_state=42)
            
            vectors_2d = reducer.fit_transform(vectors)
            
            # Визуализация
            ax = axes[idx]
            scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6, s=100)
            
            # Добавляем подписи
            for i, word in enumerate(word_labels):
                ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                           fontsize=9, alpha=0.7)
            
            ax.set_title(f'{model.name}\n({method.upper()})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Компонента 1', fontsize=12)
            ax.set_ylabel('Компонента 2', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Визуализация сохранена: {output_path}")
        plt.close()
    
    def test_analogies_batch(self, analogies: list):
        """
        Протестировать несколько аналогий и сравнить результаты
        
        Args:
            analogies: Список кортежей (word1, word2, word3, expected_answer)
        """
        print(f"\n{'='*70}")
        print("ТЕСТИРОВАНИЕ АНАЛОГИЙ")
        print(f"{'='*70}")
        
        results = {model_name: {'correct': 0, 'total': 0} for model_name in self.models.keys()}
        
        for word1, word2, word3, expected in analogies:
            print(f"\n📝 Аналогия: {word1} - {word2} = {word3} - ? (ожидается: {expected})")
            
            for model_name, model in self.models.items():
                results_list = model.analogy(word1, word2, word3, topn=5)
                results[model_name]['total'] += 1
                
                if results_list:
                    top_answer = results_list[0][0]
                    if top_answer.lower() == expected.lower():
                        results[model_name]['correct'] += 1
                        print(f"   ✅ {model.name}: {top_answer} (правильно!)")
                    else:
                        print(f"   ❌ {model.name}: {top_answer} (ожидалось: {expected})")
                else:
                    print(f"   ❌ {model.name}: не удалось выполнить аналогию")
        
        # Выводим итоговую статистику
        print(f"\n{'='*70}")
        print("ИТОГОВАЯ СТАТИСТИКА")
        print(f"{'='*70}")
        
        for model_name, stats in results.items():
            model = self.models[model_name]
            accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"\n{model.name}:")
            print(f"   Правильных ответов: {stats['correct']}/{stats['total']}")
            print(f"   Точность: {accuracy:.1f}%")


def main():
    """Основная функция"""
    
    print("="*70)
    print("СРАВНЕНИЕ КАЧЕСТВА ЭМБЕДДИНГОВ")
    print("="*70)
    print("\nЭтот скрипт загружает и сравнивает модели эмбеддингов:")
    print("  - Word2Vec (Google News)")
    print("  - FastText или GloVe (в качестве второй модели)")
    print("на задачах поиска похожих слов и выполнения аналогий.\n")
    
    # Создаем компаратор
    comparator = EmbeddingComparator()
    
    # Загружаем модели
    print("\n" + "="*70)
    print("ЗАГРУЗКА МОДЕЛЕЙ")
    print("="*70)
    print("\n💡 СОВЕТ: Если загрузка занимает слишком много времени,")
    print("   вы можете прервать её (Ctrl+C) и продолжить с одной моделью.")
    print("   Для полного сравнения нужны обе модели, но можно работать и с одной.")
    print("\n💾 КЭШИРОВАНИЕ:")
    print("   Модели скачиваются только один раз при первом запуске.")
    print("   После этого они сохраняются в кэше.")
    print("   ⚠️  ВАЖНО: Загрузка из кэша все равно требует времени (30-120 сек)")
    print("      потому что модели большие (~1-2 ГБ) и их нужно загрузить в RAM.")
    print("      Это нормальное поведение - файл читается с диска в память.")
    print("   Кэш gensim обычно находится в: C:\\Users\\ВашеИмя\\gensim-data\\")
    print("   или ~/.gensim/ на Linux/Mac\n")
    
    word2vec = comparator.load_word2vec()
    fasttext = comparator.load_fasttext()
    
    if not word2vec and not fasttext:
        print("\n❌ ОШИБКА: Не удалось загрузить ни одну модель!")
        print("\nАльтернативный способ загрузки:")
        print("1. Скачайте модели вручную")
        print("2. Укажите путь к файлам в коде")
        print("\nИли используйте более легкие модели:")
        print("  - word2vec-google-news-300 (Word2Vec)")
        print("  - fasttext-wiki-news-subwords-300 (FastText)")
        return
    
    if not word2vec:
        print("\n⚠️  Предупреждение: Word2Vec не загружен, работаем только со второй моделью")
    elif not fasttext:
        print("\n⚠️  Предупреждение: Вторая модель (FastText/GloVe) не загружена")
        print("   Работаем только с Word2Vec - все функции доступны!")
        print("   Для полного сравнения рекомендуется загрузить обе модели")
        print("   (можно попробовать запустить скрипт позже)")
    else:
        import sys
        sys.stdout.flush()
        
        # Безопасно получаем имя второй модели
        model_names = list(comparator.models.values())
        if len(model_names) >= 2:
            second_model_name = model_names[1].name
        else:
            second_model_name = "вторая модель"
        
        print(f"\n✅ Обе модели успешно загружены!")
        print(f"   - Word2Vec (Google News)")
        print(f"   - {second_model_name}")
        sys.stdout.flush()
        
        print("\n⏳ Переходим к тестированию моделей...")
        sys.stdout.flush()
    
    # Тестируем поиск похожих слов
    import sys
    sys.stdout.flush()
    
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ: ПОИСК ПОХОЖИХ СЛОВ")
    print("="*70)
    sys.stdout.flush()
    
    # Уменьшаем количество тестовых слов для ускорения
    test_words = ['king', 'queen', 'computer', 'science', 'music', 'love']
    
    print(f"Будет протестировано {len(test_words)} слов...")
    print("💡 Совет: Можно уменьшить количество слов в коде для ускорения")
    sys.stdout.flush()
    
    for idx, word in enumerate(test_words, 1):
        print(f"\n[{idx}/{len(test_words)}] Тестируем слово: '{word}'")
        sys.stdout.flush()
        comparator.compare_similar_words(word, topn=5)  # Уменьшили с 10 до 5
        sys.stdout.flush()
    
    # Тестируем аналогии
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ: ВЫПОЛНЕНИЕ АНАЛОГИЙ")
    print("="*70)
    
    # Примеры аналогий
    analogies = [
        ('king', 'man', 'queen', 'woman'),  # Король - мужчина = Королева - ?
        ('paris', 'france', 'minsk', 'belarus'),  # Париж - Франция = Минск - ?
        ('italy', 'pizza', 'japan', 'sushi'),  # Италия - пицца = Япония - ?
        ('good', 'better', 'bad', 'worse'),  # Хороший - лучше = Плохой - ?
        ('computer', 'keyboard', 'car', 'steering_wheel'),  # Компьютер - клавиатура = Машина - ?
    ]
    
    for word1, word2, word3, expected in analogies:
        comparator.compare_analogies(word1, word2, word3, topn=5)
    
    # Тестируем батч аналогий
    comparator.test_analogies_batch(analogies)
    
    # Визуализация
    print("\n" + "="*70)
    print("ВИЗУАЛИЗАЦИЯ ВЕКТОРОВ")
    print("="*70)
    
    # Выбираем слова для визуализации (разные категории)
    visualization_words = [
        # Короли и королевы
        'king', 'queen', 'prince', 'princess',
        # Страны и столицы
        'france', 'paris', 'germany', 'berlin', 'russia', 'moscow',
        # Еда
        'pizza', 'pasta', 'sushi', 'rice', 'bread', 'cheese',
        # Технологии
        'computer', 'keyboard', 'mouse', 'screen', 'internet',
        # Эмоции
        'love', 'hate', 'happy', 'sad', 'angry', 'joy'
    ]
    
    # Визуализация с t-SNE
    comparator.visualize_vectors(
        visualization_words, 
        method='tsne', 
        output_path='embeddings_tsne.png'
    )
    
    # Визуализация с PCA
    comparator.visualize_vectors(
        visualization_words, 
        method='pca', 
        output_path='embeddings_pca.png'
    )
    
    # Выводы
    print("\n" + "="*70)
    print("ВЫВОДЫ")
    print("="*70)
    print("""
    СРАВНЕНИЕ МОДЕЛЕЙ:
    
    1. WORD2VEC (Google News):
       ✅ Хорошо работает с частыми словами
       ✅ Быстрая загрузка и использование
       ✅ Отличные результаты на аналогиях для английского языка
       ❌ Не работает с редкими словами (OOV - out of vocabulary)
       ❌ Не может обработать слова, которых нет в словаре
    
    2. FASTTEXT / GLOVE (вторая модель):
       ✅ Работает с редкими словами (FastText) или глобальными статистиками (GloVe)
       ✅ FastText может обработать слова, которых нет в словаре (OOV)
       ✅ GloVe использует глобальную статистику корпусов
       ✅ Лучше для морфологически богатых языков (FastText)
       ✅ Хорошо работает с опечатками (FastText)
       ⚠️ Может быть медленнее из-за subword обработки (FastText)
    
    РЕКОМЕНДАЦИИ:
    - Для английского языка с частыми словами: Word2Vec
    - Для редких слов и OOV: FastText
    - Для морфологически богатых языков (русский, немецкий): FastText
    - Для общего использования: FastText (более универсальный)
    """)
    
    print("\n" + "="*70)
    print("✅ АНАЛИЗ ЗАВЕРШЕН!")
    print("="*70)
    print("\nСозданные файлы:")
    print("  - embeddings_tsne.png (визуализация t-SNE)")
    print("  - embeddings_pca.png (визуализация PCA)")


if __name__ == "__main__":
    main()


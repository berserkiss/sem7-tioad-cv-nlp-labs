"""
Облако слов из беседы в Telegram

Этот скрипт создает облако слов из экспортированной беседы в Telegram.
Можно выбрать сообщения конкретного человека и создать красивое облако слов,
которое можно использовать для создания индивидуальной открытки.

Использование:
    python wordcloud_telegram.py

Зависимости:
    pip install wordcloud matplotlib pandas nltk pillow
"""

import re
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
except ImportError:
    print("Ошибка: pandas не установлен. Установите: pip install pandas")
    exit(1)

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
except ImportError:
    print("Ошибка: wordcloud или matplotlib не установлены.")
    print("Установите: pip install wordcloud matplotlib pillow")
    exit(1)

# Для работы с русским языком
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
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
    print("Предупреждение: NLTK не установлен. Установите: pip install nltk")


class TelegramParser:
    """Класс для парсинга экспортированных бесед из Telegram"""
    
    @staticmethod
    def parse_json_export(filepath: str) -> List[Dict]:
        """
        Парсит JSON экспорт из Telegram
        
        Args:
            filepath: Путь к JSON файлу экспорта
            
        Returns:
            Список сообщений в виде словарей
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            messages = []
            if 'messages' in data:
                for msg in data['messages']:
                    if 'text' in msg and 'from' in msg:
                        # Обработка текста: может быть строкой или списком
                        text = msg['text']
                        if isinstance(text, list):
                            # Если текст - список, извлекаем все строки
                            text_parts = []
                            for item in text:
                                if isinstance(item, str):
                                    text_parts.append(item)
                                elif isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                            text = ' '.join(text_parts)
                        elif not isinstance(text, str):
                            # Если не строка и не список, пропускаем
                            continue
                        
                        if text and text.strip():  # Проверяем, что текст не пустой
                            messages.append({
                                'text': text,
                                'from': msg.get('from', 'Unknown'),
                                'date': msg.get('date', '')
                            })
            
            return messages
        except Exception as e:
            print(f"Ошибка при парсинге JSON: {e}")
            return []
    
    @staticmethod
    def parse_html_export(filepath: str) -> List[Dict]:
        """
        Парсит HTML экспорт из Telegram
        
        Args:
            filepath: Путь к HTML файлу экспорта
            
        Returns:
            Список сообщений в виде словарей
        """
        try:
            from bs4 import BeautifulSoup
            
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            messages = []
            # Ищем сообщения в HTML (структура может отличаться)
            message_divs = soup.find_all(['div', 'p'], class_=lambda x: x and ('message' in str(x).lower()))
            
            for div in message_divs:
                # Извлекаем текст и автора (адаптируйте под структуру вашего экспорта)
                text = div.get_text(strip=True)
                if text and len(text) > 3:
                    messages.append({
                        'text': text,
                        'from': 'Unknown',  # Нужно адаптировать под структуру
                        'date': ''
                    })
            
            return messages
        except Exception as e:
            print(f"Ошибка при парсинге HTML: {e}")
            return []
    
    @staticmethod
    def parse_text_export(filepath: str) -> List[Dict]:
        """
        Парсит текстовый экспорт из Telegram (простой формат)
        
        Args:
            filepath: Путь к текстовому файлу
            
        Returns:
            Список сообщений в виде словарей
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            messages = []
            # Простой парсинг формата: "Имя: Текст сообщения"
            pattern = r'([^:]+):\s*(.+?)(?=\n[^:]+:|$)'
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            
            for name, text in matches:
                if text.strip():
                    messages.append({
                        'text': text.strip(),
                        'from': name.strip(),
                        'date': ''
                    })
            
            return messages
        except Exception as e:
            print(f"Ошибка при парсинге текста: {e}")
            return []


class TextPreprocessor:
    """Класс для предобработки текста"""
    
    def __init__(self):
        """Инициализация предобработчика"""
        try:
            self.stop_words = set(stopwords.words('russian'))
            self.stop_words.update(['это', 'как', 'так', 'и', 'в', 'над', 'к', 'до', 'не', 'на', 'но', 
                                   'за', 'то', 'с', 'ли', 'а', 'во', 'от', 'со', 'для', 'о', 'же', 
                                   'ну', 'вы', 'бы', 'что', 'кто', 'он', 'она', 'мы', 'они', 'ты', 
                                   'тебя', 'меня', 'мне', 'тебе', 'его', 'её', 'их', 'нам', 'вам'])
        except:
            self.stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'к', 'о', 'об', 
                              'при', 'про', 'со', 'это', 'как', 'так', 'что', 'не', 'но', 'а', 
                              'или', 'то', 'же', 'бы', 'ли', 'ну', 'вы', 'мы', 'они', 'он', 'она', 'оно'}
    
    def preprocess_text(self, text: str) -> str:
        """
        Предобработка текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Обработанный текст
        """
        if not isinstance(text, str):
            return ""
        
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Удаляем эмодзи и спецсимволы
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        
        # Токенизация
        try:
            tokens = word_tokenize(text, language='russian')
        except:
            tokens = text.split()
        
        # Удаление стоп-слов
        processed_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in self.stop_words and token.isalpha():
                processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Предобработка списка текстов"""
        return [self.preprocess_text(text) for text in texts]


class WordCloudGenerator:
    """Класс для генерации облака слов"""
    
    def __init__(self, width: int = 800, height: int = 400, background_color: str = 'white'):
        """
        Инициализация генератора облака слов
        
        Args:
            width: Ширина изображения
            height: Высота изображения
            background_color: Цвет фона
        """
        self.width = width
        self.height = height
        self.background_color = background_color
    
    def generate(self, text: str, output_path: str = 'wordcloud.png', 
                 max_words: int = 200, colormap: str = 'viridis',
                 mask_path: Optional[str] = None) -> None:
        """
        Генерирует облако слов
        
        Args:
            text: Текст для облака слов
            output_path: Путь для сохранения изображения
            max_words: Максимальное количество слов
            colormap: Цветовая схема
            mask_path: Путь к маске (изображение контура)
        """
        # Подготовка маски, если указана
        mask = None
        if mask_path and os.path.exists(mask_path):
            try:
                mask = np.array(Image.open(mask_path))
                print(f"Использована маска: {mask_path}")
            except Exception as e:
                print(f"Ошибка при загрузке маски: {e}")
        
        # Создание облака слов
        wordcloud = WordCloud(
            width=self.width,
            height=self.height,
            background_color=self.background_color,
            max_words=max_words,
            colormap=colormap,
            mask=mask,
            relative_scaling=0.5,
            font_path=None,  # Можно указать путь к шрифту для русского языка
            collocations=False
        ).generate(text)
        
        # Сохранение
        plt.figure(figsize=(self.width/100, self.height/100), facecolor='white')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Облако слов сохранено: {output_path}")
    
    def generate_with_frequencies(self, word_freq: Dict[str, int], 
                                  output_path: str = 'wordcloud.png',
                                  max_words: int = 200, colormap: str = 'viridis',
                                  mask_path: Optional[str] = None) -> None:
        """
        Генерирует облако слов на основе частот слов
        
        Args:
            word_freq: Словарь {слово: частота}
            output_path: Путь для сохранения
            max_words: Максимальное количество слов
            colormap: Цветовая схема
            mask_path: Путь к маске
        """
        mask = None
        if mask_path and os.path.exists(mask_path):
            try:
                mask = np.array(Image.open(mask_path))
            except:
                pass
        
        wordcloud = WordCloud(
            width=self.width,
            height=self.height,
            background_color=self.background_color,
            max_words=max_words,
            colormap=colormap,
            mask=mask,
            relative_scaling=0.5,
            collocations=False
        ).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(self.width/100, self.height/100), facecolor='white')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Облако слов сохранено: {output_path}")




def main():
    """Основная функция"""
    
    print("="*60)
    print("ОБЛАКО СЛОВ ИЗ БЕСЕДЫ В TELEGRAM")
    print("="*60)
    
    # Парсинг беседы
    print("\n[1/5] Парсинг беседы из Telegram...")
    parser = TelegramParser()
    
    # Ищем файл result.json (стандартное имя экспорта из Telegram Desktop)
    export_file = 'result.json'
    
    # Если файл не найден, можно указать полный путь:
    # export_file = 'C:/Users/ВашеИмя/Downloads/result.json'
    
    if not os.path.exists(export_file):
        print("❌ ОШИБКА: Файл result.json не найден!")
        print("\n📋 ИНСТРУКЦИЯ ПО ЭКСПОРТУ ИЗ TELEGRAM:")
        print("   1. Откройте Telegram Desktop")
        print("   2. Выберите нужную беседу")
        print("   3. Нажмите на три точки (⋮) в правом верхнем углу")
        print("   4. Выберите 'Export chat history' (Экспорт истории чата)")
        print("   5. В окне экспорта:")
        print("      - Формат: JSON")
        print("      - Период: Entire history (Вся история)")
        print("      - Нажмите Export")
        print("   6. Сохраните файл как 'result.json' в папку со скриптом")
        print(f"      Текущая папка: {os.getcwd()}")
        print("\n   Или измените путь к файлу в коде (строка 470)")
        print("\n   Скрипт завершен. Экспортируйте беседу и запустите снова.")
        return
    
    print(f"✅ Найден файл: {export_file}")
    
    # Парсинг файла
    if export_file.endswith('.json'):
        messages = parser.parse_json_export(export_file)
    elif export_file.endswith('.html'):
        messages = parser.parse_html_export(export_file)
    else:
        messages = parser.parse_text_export(export_file)
    
    if not messages:
        print("❌ ОШИБКА: Не удалось загрузить сообщения из файла!")
        print("   Проверьте формат файла. Должен быть JSON экспорт из Telegram.")
        return
    
    print(f"   ✅ Успешно загружено сообщений: {len(messages)}")
    
    print(f"Загружено сообщений: {len(messages)}")
    
    # Определяем уникальных отправителей
    senders = set(msg.get('from', 'Unknown') for msg in messages)
    print(f"\nНайдены отправители: {', '.join(senders)}")
    
    # Выбор отправителя
    print("\n[2/5] Выбор отправителя для анализа...")
    if len(senders) > 1:
        print("Доступные отправители:")
        for i, sender in enumerate(senders, 1):
            msg_count = sum(1 for msg in messages if msg.get('from') == sender)
            print(f"  {i}. {sender} ({msg_count} сообщений)")
        
        # Интерактивный выбор
        try:
            choice = input("\nВведите номер отправителя (или Enter для первого): ").strip()
            if choice:
                selected_sender = list(senders)[int(choice) - 1]
            else:
                selected_sender = list(senders)[0]
        except (ValueError, IndexError):
            print("⚠️ Неверный выбор, используем первого отправителя")
            selected_sender = list(senders)[0]
        
        print(f"\n✅ Выбран: {selected_sender}")
    else:
        selected_sender = list(senders)[0] if senders else 'Unknown'
        print(f"✅ Выбран: {selected_sender}")
    
    # Фильтрация сообщений по отправителю
    filtered_messages = [msg for msg in messages if msg.get('from') == selected_sender]
    print(f"Сообщений от {selected_sender}: {len(filtered_messages)}")
    
    # Извлечение текстов
    texts = []
    for msg in filtered_messages:
        text = msg.get('text', '')
        # Обрабатываем случай, когда text может быть списком
        if isinstance(text, list):
            text = ' '.join(str(item) for item in text if item)
        elif not isinstance(text, str):
            text = str(text) if text else ''
        if text and text.strip():
            texts.append(text)
    
    if not texts:
        print("❌ ОШИБКА: Не найдено текстовых сообщений от выбранного отправителя!")
        return
    
    all_text = ' '.join(texts)
    
    # Предобработка
    print("\n[3/5] Предобработка текста (удаление стоп-слов)...")
    preprocessor = TextPreprocessor()
    processed_text = preprocessor.preprocess_text(all_text)
    
    # Подсчет частот
    print("\n[4/5] Подсчет частот слов...")
    words = processed_text.split()
    word_freq = Counter(words)
    
    print(f"\nТоп-10 самых частых слов:")
    for word, freq in word_freq.most_common(10):
        print(f"  {word}: {freq}")
    
    # Генерация облака слов
    print("\n[5/5] Генерация облака слов...")
    generator = WordCloudGenerator(width=1200, height=600, background_color='white')
    
    # Обычное облако слов
    output_path = f'wordcloud_{selected_sender.replace(" ", "_")}.png'
    generator.generate(
        processed_text,
        output_path=output_path,
        max_words=100,
        colormap='viridis'
    )
    
    # Облако слов с частотами (более точное)
    output_path_freq = f'wordcloud_freq_{selected_sender.replace(" ", "_")}.png'
    generator.generate_with_frequencies(
        dict(word_freq),
        output_path=output_path_freq,
        max_words=100,
        colormap='plasma'
    )
    
    print("\n" + "="*60)
    print("✅ ГОТОВО!")
    print("="*60)
    print(f"\nСозданы файлы:")
    print(f"  - {output_path}")
    print(f"  - {output_path_freq}")
    print(f"\n💡 Совет: Используйте эти изображения для создания индивидуальной открытки!")
    print(f"💡 Для создания облака в виде контура укажите путь к маске в параметре mask_path")


if __name__ == "__main__":
    main()


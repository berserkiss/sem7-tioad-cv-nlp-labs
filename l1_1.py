import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_specific_image():
    """
    Обработка конкретного изображения без запроса пути
    """
    # Укажите правильный путь к вашему изображению
    image_path = r"D:\Sem7\tioad\L1\scottish-fold-2.jpg"

    try:
        # Загрузка изображения
        image = cv2.imread(image_path)

        if image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return

        # Конвертация в RGB для matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Конвертация в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Пороговая бинаризация
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Адаптивная бинаризация
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 51, 2)

        # Отображение результатов
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(image_rgb)
        plt.title('Исходное изображение')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(gray, cmap='gray')
        plt.title('Оттенки серого')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(binary, cmap='gray')
        plt.title('Пороговая бинаризация')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(adaptive, cmap='gray')
        plt.title('Адаптивная бинаризация')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Ошибка: {e}")


# Запуск обработки
process_specific_image()
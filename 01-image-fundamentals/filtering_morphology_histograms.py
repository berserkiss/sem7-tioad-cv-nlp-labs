import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def create_directories():
    """Создает папки для сохранения результатов"""
    os.makedirs('results', exist_ok=True)
    os.makedirs('images', exist_ok=True)


def task3_histogram_equalization():
    """
    Задание 3: Выравнивание гистограммы затемненного изображения
    """
    print("=" * 60)
    print("ЗАДАНИЕ 3: Выравнивание гистограммы")
    print("=" * 60)

    # Путь к затемненному изображению
    dark_image_path = "images/dark_image.jpg"

    if not os.path.exists(dark_image_path):
        print(f"Изображение не найдено: {dark_image_path}")
        print("Создаем тестовое затемненное изображение...")
        create_dark_test_image()

    # Загрузка изображения
    img = cv2.imread(dark_image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Ошибка загрузки изображения")
        return

    # Выравнивание гистограммы
    img_equalized = cv2.equalizeHist(img)

    # Построение гистограмм
    plt.figure(figsize=(15, 10))

    # Исходное изображение и гистограмма
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Исходное затемненное изображение')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.hist(img.ravel(), bins=256, range=[0, 256], color='r', alpha=0.7)  # ИСПРАВЛЕНО
    plt.title('Гистограмма исходного изображения')
    plt.xlabel('Значение пикселя')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)

    # Выровненное изображение и гистограмма
    plt.subplot(2, 2, 3)
    plt.imshow(img_equalized, cmap='gray')
    plt.title('Выровненное изображение')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.hist(img_equalized.ravel(), bins=256, range=[0, 256], color='g', alpha=0.7)  # ИСПРАВЛЕНО
    plt.title('Гистограмма после выравнивания')
    plt.xlabel('Значение пикселя')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/task3_histogram_equalization.png')
    plt.show()

    # Сохранение результатов
    cv2.imwrite('results/dark_original.jpg', img)
    cv2.imwrite('results/dark_equalized.jpg', img_equalized)

    print("Результаты сохранены в папке 'results/'")


def task4_image_filtering():
    """
    Задание 4: Фильтрация зашумленного изображения
    """
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 4: Фильтрация шумов")
    print("=" * 60)

    # Путь к зашумленному изображению (замените на свой)
    noisy_image_path = "images/noisy_image.jpg"

    if not os.path.exists(noisy_image_path):
        print(f"Изображение не найдено: {noisy_image_path}")
        print("Создаем тестовое зашумленное изображение...")
        create_noisy_test_image()

    # Загрузка изображения
    img = cv2.imread(noisy_image_path)

    if img is None:
        print("Ошибка загрузки изображения")
        return

    # Применение различных фильтров
    # 1. Простое размытие (усредняющий фильтр)
    blur = cv2.blur(img, (15, 15))

    # 2. Гауссовское размытие
    gaussian = cv2.GaussianBlur(img, (15, 15), 0)

    # 3. Медианный фильтр (хорош для salt-and-pepper шума)
    median = cv2.medianBlur(img, 15)

    # Отображение результатов
    plt.figure(figsize=(20, 12))

    titles = ['Исходное зашумленное', 'Blur (усредняющий)', 'GaussianBlur', 'MedianBlur']
    images = [img, blur, gaussian, median]

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        # Конвертация BGR to RGB для правильного отображения
        if len(images[i].shape) == 3:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/task4_filtering_comparison.png')
    plt.show()

    # Сохранение результатов
    cv2.imwrite('results/noisy_original.jpg', img)
    cv2.imwrite('results/filtered_blur.jpg', blur)
    cv2.imwrite('results/filtered_gaussian.jpg', gaussian)
    cv2.imwrite('results/filtered_median.jpg', median)

    print("Результаты фильтрации сохранены в папке 'results/'")


def task5_morphological_operations():
    """
    Задание 5: Морфологические операции
    """
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 5: Морфологические операции")
    print("=" * 60)

    # Путь к изображению с предметами (замените на свой)
    objects_image_path = "images/objects_image.jpg"

    if not os.path.exists(objects_image_path):
        print(f"Изображение не найдено: {objects_image_path}")
        print("Создаем тестовое изображение с предметами...")
        create_objects_test_image()

    # Загрузка изображения
    img = cv2.imread(objects_image_path)

    if img is None:
        print("Ошибка загрузки изображения")
        return

    # Конвертация в серое и бинаризация
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Создание ядра для морфологических операций
    kernel = np.ones((5, 5), np.uint8)

    # Эрозия - уменьшает объекты
    erosion = cv2.erode(binary, kernel, iterations=1)

    # Дилатация - увеличивает объекты
    dilation = cv2.dilate(binary, kernel, iterations=1)

    # Комбинация: открытие (эрозия + дилатация) - убирает шум
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Комбинация: закрытие (дилатация + эрозия) - заполняет дыры
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Отображение результатов
    plt.figure(figsize=(20, 15))

    images = [
        img, binary, erosion, dilation, opening, closing
    ]

    titles = [
        'Исходное изображение',
        'Бинарное изображение',
        'Эрозия (erode)',
        'Дилатация (dilate)',
        'Открытие (opening)',
        'Закрытие (closing)'
    ]

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        if len(images[i].shape) == 3:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/task5_morphological_operations.png')
    plt.show()

    # Сохранение результатов
    cv2.imwrite('results/original_objects.jpg', img)
    cv2.imwrite('results/binary.jpg', binary)
    cv2.imwrite('results/erosion.jpg', erosion)
    cv2.imwrite('results/dilation.jpg', dilation)
    cv2.imwrite('results/opening.jpg', opening)
    cv2.imwrite('results/closing.jpg', closing)

    print("Результаты морфологических операций сохранены в папке 'results/'")


# Функции для создания тестовых изображений
def create_dark_test_image():
    """Создание затемненного тестового изображения"""
    # Создаем нормальное изображение
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(img, 'OpenCV Test Image', (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Затемняем изображение
    dark_img = (img * 0.3).astype(np.uint8)  # Уменьшаем яркость на 70%

    os.makedirs('images', exist_ok=True)
    cv2.imwrite('images/dark_image.jpg', dark_img)
    print("Создано затемненное изображение: images/dark_image.jpg")


def create_noisy_test_image():
    """Создание зашумленного тестового изображения"""
    # Создаем нормальное изображение
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(img, 'Noisy Test Image', (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Добавляем шум
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)

    os.makedirs('images', exist_ok=True)
    cv2.imwrite('images/noisy_image.jpg', noisy_img)
    print("Создано зашумленное изображение: images/noisy_image.jpg")


def create_objects_test_image():
    """Создание изображения с предметами на светлом фоне"""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # Белый фон

    # Рисуем различные геометрические фигуры
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)  # Красный квадрат
    cv2.circle(img, (300, 100), 50, (255, 0, 0), -1)  # Синий круг
    cv2.ellipse(img, (450, 100), (60, 30), 0, 0, 360, (0, 255, 0), -1)  # Зеленый эллипс

    # Треугольник
    pts = np.array([[50, 250], [150, 250], [100, 350]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 0))

    os.makedirs('images', exist_ok=True)
    cv2.imwrite('images/objects_image.jpg', img)
    print("Создано изображение с предметами: images/objects_image.jpg")


def main():
    """Главная функция"""
    print("ОБРАБОТКА ИЗОБРАЖЕНИЙ С OPENCV")
    print("=" * 60)

    # Создаем папки для результатов
    create_directories()

    # Выполняем все задания
    task3_histogram_equalization()
    task4_image_filtering()
    task5_morphological_operations()

    print("\n" + "=" * 60)
    print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ!")
    print("Результаты сохранены в папке 'results/'")
    print("=" * 60)


if __name__ == "__main__":
    main()
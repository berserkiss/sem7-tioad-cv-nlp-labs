import cv2
import numpy as np
import os

# Создаем директорию для выходных файлов если её нет
if not os.path.exists('output'):
    os.makedirs('output')

print("=== ОБРАБОТКА ВАШИХ РЕАЛЬНЫХ ИЗОБРАЖЕНИЙ ===")

# Загружаем ваши изображения
objects_img = cv2.imread('images/objects_image.jpg')
photo_img = cv2.imread('images/photo_real.jpg')

print(f"Размер objects_image.jpg: {objects_img.shape}")
print(f"Размер photo_real.jpg: {photo_img.shape}")

print("\n=== ЗАДАНИЕ 1: Применение операторов к вашему изображению ===")

# Работаем с вашим изображением photo_real.jpg
gray = cv2.cvtColor(photo_img, cv2.COLOR_BGR2GRAY)

# 1.1 Оператор Собеля
print("Применяем оператор Собеля...")
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_result = np.uint8(np.clip(sobel_combined, 0, 255))

# 1.2 Оператор Лапласа
print("Применяем оператор Лапласа...")
laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
laplacian_result = np.uint8(np.absolute(laplacian))

# 1.3 Детектор границ Кэнни (подбираем оптимальные параметры)
print("Применяем детектор Кэнни с разными параметрами...")
# Предварительная обработка
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Пробуем разные наборы параметров для получения наилучшего результата
canny1 = cv2.Canny(blurred, 50, 150, apertureSize=3)
canny2 = cv2.Canny(blurred, 100, 200, apertureSize=3)
canny3 = cv2.Canny(blurred, 30, 100, apertureSize=3)

# Выбираем лучший результат (средний вариант обычно работает хорошо)
canny = canny2

# Сохраняем результаты задания 1
cv2.imwrite('output/sobel_your_image.jpg', sobel_result)
cv2.imwrite('output/laplacian_your_image.jpg', laplacian_result)
cv2.imwrite('output/canny_your_image.jpg', canny)
cv2.imwrite('output/canny_variant1.jpg', canny1)
cv2.imwrite('output/canny_variant2.jpg', canny2)
cv2.imwrite('output/canny_variant3.jpg', canny3)

print("Результаты задания 1 сохранены:")
print("- output/sobel_your_image.jpg")
print("- output/laplacian_your_image.jpg")
print("- output/canny_your_image.jpg")

print("\n=== ЗАДАНИЕ 2: ПРАВИЛЬНЫЙ КОНТУРНЫЙ АНАЛИЗ ===")

# ЭТАП 1: Предварительная обработка изображения
print("Этап 1: Предварительная обработка...")
gray_obj = cv2.cvtColor(objects_img, cv2.COLOR_BGR2GRAY)

# Сглаживание и фильтрация помех
blurred_obj = cv2.GaussianBlur(gray_obj, (5, 5), 0)

# Увеличение контраста с помощью CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blurred_obj)

cv2.imwrite('output/step1_preprocessed.jpg', enhanced)
print("  ✓ Сглаживание, фильтрация и увеличение контраста применены")

# ЭТАП 2: Бинаризация изображения
print("Этап 2: Бинаризация...")
# Используем метод Отсу для автоматического выбора порога
_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imwrite('output/step2_binary.jpg', binary)
print("  ✓ Бинарное изображение получено методом Отсу")

# ЭТАП 3: Операции математической морфологии
print("Этап 3: Математическая морфология...")

# Создаем структурирующий элемент
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Операция "открытие" - убирает мелкие шумы
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Операция "закрытие" - заполняет мелкие дыры в объектах
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

cv2.imwrite('output/step3_morphology.jpg', opened)
print("  ✓ Применены операции открытия и закрытия")

# ЭТАП 4: Выделение контуров объектов (ТОЛЬКО ВНЕШНИЕ!)
print("Этап 4: Выделение только внешних контуров...")

# КРИТИЧЕСКИ ВАЖНО: Используем cv2.RETR_EXTERNAL для получения ТОЛЬКО внешних контуров
# Это исключает внутренние контуры, тени и текстуры, которые приводят к неверному подсчету
contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"  ✓ Найдено внешних контуров: {len(contours)}")

# ЭТАП 5: Первичная фильтрация контуров (по периметру, площади и т.п.)
print("Этап 5: Фильтрация контуров по критериям...")

# Параметры фильтрации
min_area = 50  # минимальная площадь
max_area = 10000  # максимальная площадь
min_perimeter = 20  # минимальный периметр

filtered_contours = []
for contour in contours:
    # Фильтрация по площади
    area = cv2.contourArea(contour)
    if not (min_area <= area <= max_area):
        continue

    # Фильтрация по периметру
    perimeter = cv2.arcLength(contour, True)
    if perimeter < min_perimeter:
        continue

    # Фильтрация по соотношению сторон
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    if not (0.1 <= aspect_ratio <= 10.0):
        continue

    filtered_contours.append(contour)

print(f"  ✓ После фильтрации осталось контуров: {len(filtered_contours)}")

# ЭТАП 6: Анализ контуров в соответствии с поставленной задачей
print("Этап 6: Анализ и подсчет предметов...")

# Создаем результирующее изображение
result_img = objects_img.copy()

# Рисуем прямоугольники, в которые вписаны предметы
for i, contour in enumerate(filtered_contours):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(result_img, str(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Выделяем контуры с наибольшей длиной и наибольшей площадью
if filtered_contours:
    # Находим контур с наибольшей длиной (периметром)
    contour_lengths = [cv2.arcLength(cnt, True) for cnt in filtered_contours]
    max_length_idx = np.argmax(contour_lengths)
    cnt_max_length = filtered_contours[max_length_idx]
    max_length = contour_lengths[max_length_idx]

    # Находим контур с наибольшей площадью
    contour_areas = [cv2.contourArea(cnt) for cnt in filtered_contours]
    max_area_idx = np.argmax(contour_areas)
    cnt_max_area = filtered_contours[max_area_idx]
    max_area = contour_areas[max_area_idx]

    # Выделяем их специальными цветами
    cv2.drawContours(result_img, [cnt_max_length], -1, (255, 0, 0), 3)  # синий - макс длина
    cv2.drawContours(result_img, [cnt_max_area], -1, (0, 0, 255), 3)  # красный - макс площадь

    print(f"  ✓ Контур с наибольшей длиной: №{max_length_idx + 1} (длина: {max_length:.1f})")
    print(f"  ✓ Контур с наибольшей площадью: №{max_area_idx + 1} (площадь: {max_area:.1f})")

# Сохраняем результаты всех этапов
cv2.imwrite('output/step6_final_result.jpg', result_img)

# Создаем изображение только с контурами для демонстрации
contours_only = np.zeros_like(objects_img)
cv2.drawContours(contours_only, filtered_contours, -1, (255, 255, 255), 2)
cv2.imwrite('output/contours_only.jpg', contours_only)

print(f"\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
print(f"Количество предметов: {len(filtered_contours)}")
print(f"Метод поиска: cv2.RETR_EXTERNAL (исключает внутренние контуры)")
print(f"Обработано бинарное изображение (исключает влияние теней и текстур)")

print(f"\nПоследовательность этапов контурного анализа выполнена полностью:")
print("✓ 1. Предварительная обработка (сглаживание, фильтрация, контраст)")
print("✓ 2. Бинаризация (метод Отсу)")
print("✓ 3. Математическая морфология (открытие + закрытие)")
print("✓ 4. Выделение ТОЛЬКО внешних контуров (cv2.RETR_EXTERNAL)")
print("✓ 5. Фильтрация по площади, периметру и соотношению сторон")
print("✓ 6. Анализ и выделение контуров с максимальными характеристиками")

print(f"\nФайлы результатов:")
print("- step1_preprocessed.jpg (после предобработки)")
print("- step2_binary.jpg (бинарное изображение)")
print("- step3_morphology.jpg (после морфологии)")
print("- step6_final_result.jpg (итоговый результат)")
print("- contours_only.jpg (визуализация контуров)")

print(f"\nОбработка закончена. Проверьте сохранённые изображения.")


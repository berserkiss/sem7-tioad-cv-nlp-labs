import cv2
import numpy as np
import os

# Создаем директорию для выходных файлов
if not os.path.exists('output'):
    os.makedirs('output')

print("=== ЗАДАНИЕ 3: ПРЕОБРАЗОВАНИЕ ХАФА С РЕАЛЬНЫМИ ИЗОБРАЖЕНИЯМИ ===")

# Загружаем ваши реальные изображения
lines_img = cv2.imread('images/lines_image.jpg')
circles_img = cv2.imread('images/circles_image.jpg')

print(f"Загружены реальные изображения:")
print(f"  - lines_image.jpg: {lines_img.shape}")
print(f"  - circles_image.jpg: {circles_img.shape}")

print("\n=== ЗАДАНИЕ 3A: ОБНАРУЖЕНИЕ ЛИНИЙ НА РЕАЛЬНОМ ИЗОБРАЖЕНИИ ===")

# ЭТАП 1: Предварительная обработка изображения с линиями (дорожная разметка)
gray_lines = cv2.cvtColor(lines_img, cv2.COLOR_BGR2GRAY)
print("✓ Преобразование в оттенки серого")

# Применяем размытие для уменьшения шума
blurred_lines = cv2.GaussianBlur(gray_lines, (5, 5), 0)
print("✓ Применено размытие по Гауссу")

# Применяем детектор границ Кэнни с оптимальными параметрами для дорожной разметки
edges_lines = cv2.Canny(blurred_lines, 80, 200, apertureSize=3)

# Сохраняем промежуточный результат
cv2.imwrite('output/real_edges_for_lines.jpg', edges_lines)
print("✓ Детектор Кэнни применен, результат сохранен в output/real_edges_for_lines.jpg")

# ЭТАП 2: Применение cv2.HoughLines() для поиска прямых линий
print("\nПрименяем cv2.HoughLines() к реальному изображению...")

# Параметры оптимизированы для дорожной разметки
rho = 1  # разрешение по расстоянию в пикселях
theta = np.pi / 180  # разрешение по углу в радианах
threshold = 150  # увеличенный порог для более четких линий

lines = cv2.HoughLines(edges_lines, rho, theta, threshold)

if lines is not None:
    print(f"Найдено линий: {len(lines)}")
else:
    print("Линии не найдены, попробуем с меньшим порогом...")
    # Попробуем с более чувствительными параметрами
    lines = cv2.HoughLines(edges_lines, rho, theta, 100)
    if lines is not None:
        print(f"С пониженным порогом найдено линий: {len(lines)}")
    else:
        print("Линии не обнаружены")

# ЭТАП 3: Визуализация найденных линий
result_lines = lines_img.copy()

if lines is not None:
    # Ограничиваем количество отображаемых линий для лучшей читаемости
    max_lines_to_show = min(20, len(lines))

    for i in range(max_lines_to_show):
        rho, theta = lines[i][0]

        # Преобразуем из полярных координат в декартовы
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # Вычисляем точки для рисования линии
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Рисуем линию красным цветом
        cv2.line(result_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    print(f"✓ Отображено {max_lines_to_show} наиболее выраженных линий")

# Сохраняем результат обнаружения линий
cv2.imwrite('output/real_detected_lines_result.jpg', result_lines)
print("✓ Результат сохранен в output/real_detected_lines_result.jpg")

print("\n=== ЗАДАНИЕ 3B: ОБНАРУЖЕНИЕ ОКРУЖНОСТЕЙ НА РЕАЛЬНОМ ИЗОБРАЖЕНИИ ===")

# ЭТАП 1: Предварительная обработка изображения с окружностями (монеты)
gray_circles = cv2.cvtColor(circles_img, cv2.COLOR_BGR2GRAY)
print("✓ Преобразование в оттенки серого")

# Применяем медианное размытие (оптимально для окружностей)
blurred_circles = cv2.medianBlur(gray_circles, 5)
print("✓ Применено медианное размытие")

# Дополнительное размытие по Гауссу для устранения мелких деталей
blurred_circles = cv2.GaussianBlur(blurred_circles, (9, 9), 2)

# Сохраняем предобработанное изображение
cv2.imwrite('output/real_preprocessed_circles.jpg', blurred_circles)
print("✓ Предобработка завершена, результат в output/real_preprocessed_circles.jpg")

# ЭТАП 2: Применение cv2.HoughCircles() для поиска окружностей
print("\nПрименяем cv2.HoughCircles() к реальному изображению...")

# Первая попытка с консервативными параметрами
circles = cv2.HoughCircles(
    blurred_circles,
    cv2.HOUGH_GRADIENT,
    dp=1,  # разрешение аккумулятора = разрешению изображения
    minDist=50,  # минимальное расстояние между центрами (для монет)
    param1=100,  # верхний порог для Кэнни
    param2=50,  # порог аккумулятора для центров
    minRadius=20,  # минимальный радиус монеты
    maxRadius=80  # максимальный радиус монеты
)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    num_circles_conservative = len(circles)
    print(f"С консервативными параметрами найдено: {num_circles_conservative} окружностей")
else:
    num_circles_conservative = 0
    print("С консервативными параметрами окружности не найдены")

# Вторая попытка с более чувствительными параметрами
print("Попытка с более чувствительными параметрами...")

circles_sensitive = cv2.HoughCircles(
    blurred_circles,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=30,  # уменьшили минимальное расстояние
    param1=50,  # уменьшили порог Кэнни
    param2=30,  # уменьшили порог аккумулятора
    minRadius=15,  # уменьшили минимальный радиус
    maxRadius=100  # увеличили максимальный радиус
)

if circles_sensitive is not None:
    circles_sensitive = np.round(circles_sensitive[0, :]).astype("int")
    num_circles_sensitive = len(circles_sensitive)
    print(f"С чувствительными параметрами найдено: {num_circles_sensitive} окружностей")

    # Используем более чувствительный результат для дальнейшей работы
    circles = circles_sensitive
    num_circles = num_circles_sensitive
else:
    num_circles_sensitive = 0
    num_circles = num_circles_conservative
    print("Окружности не найдены даже с чувствительными параметрами")

# ЭТАП 3: Анализ и фильтрация результатов
if circles is not None and len(circles) > 0:
    print(f"\nПроведем анализ найденных {len(circles)} окружностей...")

    # Фильтрация по размеру и качеству
    filtered_circles = []
    for i, (x, y, r) in enumerate(circles):
        # Проверяем, что центр находится в пределах изображения
        if 0 <= x < circles_img.shape[1] and 0 <= y < circles_img.shape[0]:
            # Проверяем разумный размер для монет
            if 15 <= r <= 100:
                filtered_circles.append((x, y, r))

    print(f"После фильтрации осталось: {len(filtered_circles)} окружностей")
    num_circles = len(filtered_circles)

    # ЭТАП 4: Визуализация результатов
    result_circles = circles_img.copy()

    for i, (x, y, r) in enumerate(filtered_circles):
        # Рисуем окружность зеленым цветом
        cv2.circle(result_circles, (x, y), r, (0, 255, 0), 3)
        # Рисуем центр красным цветом
        cv2.circle(result_circles, (x, y), 3, (0, 0, 255), -1)
        # Добавляем номер окружности
        cv2.putText(result_circles, str(i + 1), (x - 15, y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Сохраняем основной результат
    cv2.imwrite('output/real_detected_circles_result.jpg', result_circles)

    # Создаем дополнительную визуализацию только с контурами на черном фоне
    circles_overlay = np.zeros_like(circles_img)
    for x, y, r in filtered_circles:
        cv2.circle(circles_overlay, (x, y), r, (255, 255, 255), 2)
        cv2.circle(circles_overlay, (x, y), 2, (0, 0, 255), -1)

    cv2.imwrite('output/real_circles_overlay.jpg', circles_overlay)

    print("✓ Результаты сохранены:")
    print("  - output/real_detected_circles_result.jpg (основной результат)")
    print("  - output/real_circles_overlay.jpg (контуры на черном фоне)")

else:
    num_circles = 0
    # Создаем изображение с отметкой о том, что окружности не найдены
    result_circles = circles_img.copy()
    cv2.putText(result_circles, "No circles detected", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite('output/real_detected_circles_result.jpg', result_circles)

print(f"\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ЗАДАНИЯ 3 ===")
print(f"Обнаружение линий:")
print(f"  - Метод: cv2.HoughLines()")
print(f"  - Найдено линий: {len(lines) if lines is not None else 0}")
print(f"  - Результат: output/real_detected_lines_result.jpg")

print(f"\nОбнаружение окружностей:")
print(f"  - Метод: cv2.HoughCircles()")
print(f"  - Количество окружностей: {num_circles}")
print(f"  - Результат: output/real_detected_circles_result.jpg")

print(f"\nОбработка закончена. Проверьте сохранённые изображения в папке 'output'.")

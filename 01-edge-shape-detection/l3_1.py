# task1_improved.py
import cv2
import numpy as np

# Путь к вашему изображению (замените на реальный файл)
image_path = 'images/chess.jpeg'

# Загрузка изображения
img = cv2.imread(image_path)
if img is None:
    print(f"Ошибка: Не удалось загрузить изображение по пути '{image_path}'. Проверьте путь и попробуйте снова.")
    exit(1)  # Завершить выполнение, если изображение не загружено

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_float = np.float32(gray)

# Детектор Харриса: вычисляет отклик на углы на основе градиентов
harris_dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
harris_dst = cv2.dilate(harris_dst, None)  # Расширение для лучшей видимости
img_harris = img.copy()
img_harris[harris_dst > 0.01 * harris_dst.max()] = [0, 0, 255]  # Отметка красным

# Детектор Ши-Томаси: находит лучшие углы для трекинга
shi_tomasi_corners = cv2.goodFeaturesToTrack(np.uint8(gray), maxCorners=100, qualityLevel=0.01, minDistance=10)
img_shi_tomasi = img.copy()
if shi_tomasi_corners is not None:
    for corner in shi_tomasi_corners:
        x, y = corner.ravel()
        cv2.circle(img_shi_tomasi, (int(x), int(y)), 5, (0, 255, 0), -1)  # Отметка зелёным
else:
    print("Углы Ши-Томаси не найдены.")

# Количественное сравнение: вывод количества углов
num_harris = np.sum(harris_dst > 0.01 * harris_dst.max())
num_shi_tomasi = len(shi_tomasi_corners) if shi_tomasi_corners is not None else 0
print(f"Найдено углов методом Харриса: {num_harris}")
print(f"Найдено углов методом Ши-Томаси: {num_shi_tomasi}")

# Сохранение результатов
cv2.imwrite('harris_result.jpg', img_harris)
cv2.imwrite('shi_tomasi_result.jpg', img_shi_tomasi)
print("Результаты сохранены: harris_result.jpg и shi_tomasi_result.jpg")

# Визуальное сравнение: отображение изображений
cv2.imshow('Original Image', img)
cv2.imshow('Harris Corners', img_harris)
cv2.imshow('Shi-Tomasi Corners', img_shi_tomasi)
print("Нажмите любую клавишу в окне изображения, чтобы завершить...")
cv2.waitKey(0)
cv2.destroyAllWindows()

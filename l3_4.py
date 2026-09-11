import cv2
import numpy as np

# Функция для упорядочивания точек контура (левый верхний, правый верхний, правый нижний, левый нижний)
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Левый верхний (минимальная сумма)
    rect[2] = pts[np.argmax(s)]  # Правый нижний (максимальная сумма)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Правый верхний (минимальная разница)
    rect[3] = pts[np.argmax(diff)]  # Левый нижний (максимальная разница)
    return rect

# Загружаем изображение
image = cv2.imread('images/receipt.jpg')
if image is None:
    print("Ошибка: не удалось загрузить изображение.")
    exit()

# Предобработка: конвертируем в серый, размываем и находим края
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# Находим контуры
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # Берем 5 самых больших

# Ищем четырехугольный контур (документ)
screen_cnt = None
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        screen_cnt = approx
        break

if screen_cnt is None:
    print("Ошибка: не удалось найти контур документа. Попробуйте улучшить освещение или фон.")
    exit()

# Упорядочиваем точки
pts = screen_cnt.reshape(4, 2)
rect = order_points(pts)

# Вычисляем размеры выровненного изображения
width_top = np.linalg.norm(rect[0] - rect[1])
width_bottom = np.linalg.norm(rect[3] - rect[2])
max_width = max(int(width_top), int(width_bottom))

height_left = np.linalg.norm(rect[0] - rect[3])
height_right = np.linalg.norm(rect[1] - rect[2])
max_height = max(int(height_left), int(height_right))

# Точки назначения (ровный прямоугольник)
dst_pts = np.array([
    [0, 0],
    [max_width - 1, 0],
    [max_width - 1, max_height - 1],
    [0, max_height - 1]
], dtype="float32")

# Получаем матрицу преобразования и применяем
M = cv2.getPerspectiveTransform(rect, dst_pts)
warped = cv2.warpPerspective(image, M, (max_width, max_height))

# Сохраняем результат
cv2.imwrite('aligned_document_corrected.jpg', warped)
print("Выровненное изображение сохранено как 'aligned_document_corrected.jpg'.")

# Опционально: показываем результат (раскомментируйте, если нужно)
# cv2.imshow('Original', image)
# cv2.imshow('Aligned', warped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

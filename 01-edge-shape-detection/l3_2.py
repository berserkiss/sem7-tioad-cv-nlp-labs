# task2_optimized.py
import cv2

# Пути к XML-файлам каскадов Хаара (скачайте с GitHub OpenCV, укажите правильные пути)
face_cascade_path = 'haarcascades/haarcascade_frontalface_default.xml'
eyes_cascade_path = 'haarcascades/haarcascade_eye.xml'
smile_cascade_path = 'haarcascades/haarcascade_smile.xml'

# Загрузка классификаторов
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eyes_cascade = cv2.CascadeClassifier(eyes_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

# Проверка загрузки
if face_cascade.empty():
    print(f'Ошибка: не удалось загрузить каскад лиц {face_cascade_path}')
    exit(1)
if eyes_cascade.empty():
    print(f'Ошибка: не удалось загрузить каскад глаз {eyes_cascade_path}')
    exit(1)
if smile_cascade.empty():
    print(f'Ошибка: не удалось загрузить каскад улыбок {smile_cascade_path}')
    exit(1)

# Загрузка изображения
image_path = 'images/backstreet.jpg'  # Замените на свой путь
img = cv2.imread(image_path)
if img is None:
    print(f'Ошибка: не удалось загрузить изображение {image_path}')
    exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Детектирование лиц
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
print(f'Найдено лиц: {len(faces)}')

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # Детектирование глаз с более чувствительными параметрами
    eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20))
    print(f"Глаз найдено на лице в позиции ({x},{y}): {len(eyes)}")
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Детектирование улыбок — более строгие параметры уменьшают ложные срабатывания
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=30, minSize=(25, 25))
    print(f"Улыбок найдено на лице в позиции ({x},{y}): {len(smiles)}")
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

cv2.imwrite('detected_faces_optimized.jpg', img)
print('Результат сохранён: detected_faces_optimized.jpg')

cv2.imshow('Detected Faces, Eyes and Smiles Optimized', img)
print('Нажмите любую клавишу для выхода из окна...')
cv2.waitKey(0)
cv2.destroyAllWindows()

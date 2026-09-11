# Полностью готовый локальный скрипт для детекции лица, глаз и улыбки на видео с камеры с комментариями

import cv2

face_cascade_path = 'haarcascades/haarcascade_frontalface_default.xml'
eyes_cascade_path = 'haarcascades/haarcascade_eye.xml'
smile_cascade_path = 'haarcascades/haarcascade_smile.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eyes_cascade = cv2.CascadeClassifier(eyes_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

if face_cascade.empty():
    print(f"Ошибка: не удалось загрузить каскад лиц по пути '{face_cascade_path}'")
    exit(1)
if eyes_cascade.empty():
    print(f"Ошибка: не удалось загрузить каскад глаз по пути '{eyes_cascade_path}'")
    exit(1)
if smile_cascade.empty():
    print(f"Ошибка: не удалось загрузить каскад улыбок по пути '{smile_cascade_path}'")
    exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру")
    exit(1)

print("Нажмите 'q' для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось получить кадр с камеры")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    cv2.imshow('Video Face, Eyes and Smile Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Выход из видео по команде пользователя')
        break

cap.release()
cv2.destroyAllWindows()
print('Видео завершено')
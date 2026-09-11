import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# --- 1. Загрузка и подготовка данных (шаги a, b) ---

# Загружаем датасет Fashion-MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Имена классов
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Нормализация [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Добавляем канал цвета (для Conv2D)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("Данные подготовлены. Форма x_train:", x_train.shape)

# =========================================================================
# --- МОДЕЛЬ 1: БАЗОВАЯ (КАК РАНЬШЕ, С ПЕРЕОБУЧЕНИЕМ) ---
# =========================================================================

print("\n--- Обучение Базовой модели (без Dropout) ---")

model_base = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_base.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Обучаем 10 эпох (как в прошлый раз)
history_base = model_base.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=2  # verbose=2 сделает лог обучения чуть короче
)

# --- Оценка базовой модели (без Dropout) ---
print("\n--- Итоговая оценка базовой модели (без Dropout) ---")
test_loss_base, test_acc_base = model_base.evaluate(x_test, y_test, verbose=2)
print(f"\nТочность на тестовых данных (без Dropout): {test_acc_base:.4f}")

# Предсказание и отчет
y_pred_probs_base = model_base.predict(x_test)
y_pred_base = np.argmax(y_pred_probs_base, axis=1)
print("\n--- Отчет о классификации (без Dropout) ---")
print(classification_report(y_test, y_pred_base, target_names=class_names))


# =========================================================================
# --- МОДЕЛЬ 2 (С ДОБАВЛЕНИЕМ DROPOUT, OPTIMIZER: ADAM) ---
# =========================================================================

print("\n--- Обучение Регуляризованной модели (с Dropout, Adam) ---")

model_regularized = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(10, activation='softmax')
])

model_regularized.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Обучаем столько же эпох для честного сравнения
history_regularized = model_regularized.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=2
)

# =========================================================================
# --- ЭКСПЕРИМЕНТ: МОДЕЛЬ 3 (DROPOUT, OPTIMIZER: RMSPROP) ---
# =========================================================================

print("\n--- Обучение Экспериментальной модели (с Dropout, RMSprop) ---")

# Та же архитектура, что и у Модели 2, но другой оптимизатор
model_experimental = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# !!! ИЗМЕНЕНИЕ: Используем 'rmsprop' вместо 'adam'
model_experimental.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Обучаем столько же эпох
history_experimental = model_experimental.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=2
)


# =========================================================================
# --- СРАВНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ (ТРИ МОДЕЛИ) ---
# =========================================================================

print("\n--- Сравнение результатов ---")

# Устанавливаем размер фигуры (шире, т.к. 3 графика)
plt.figure(figsize=(24, 6))

# --- ГРАФИК 1: ФУНКЦИИ ПОТЕРЬ (LOSS) ---

# График 1 (Базовая)
plt.subplot(1, 3, 1)
plt.plot(history_base.history['loss'], label='Потери (обучение)')
plt.plot(history_base.history['val_loss'], label='Потери (валидация)')
plt.title('Модель БЕЗ Dropout (Переобучение)')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# График 2 (С Dropout, Adam)
plt.subplot(1, 3, 2)
plt.plot(history_regularized.history['loss'], label='Потери (обучение)')
plt.plot(history_regularized.history['val_loss'], label='Потери (валидация)')
plt.title('Модель С Dropout (Optimizer: Adam)')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# График 3 (С Dropout, RMSprop)
plt.subplot(1, 3, 3)
plt.plot(history_experimental.history['loss'], label='Потери (обучение)')
plt.plot(history_experimental.history['val_loss'], label='Потери (валидация)')
plt.title('Модель С Dropout (Optimizer: RMSprop)')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Показываем первый набор графиков (Потери)
plt.suptitle('Сравнение функций потерь (Loss) для трех моделей', fontsize=16)
plt.show()

# --- ГРАФИК 2: ТОЧНОСТЬ (ACCURACY) ---

plt.figure(figsize=(24, 6))

# График 1 (Базовая)
plt.subplot(1, 3, 1)
plt.plot(history_base.history['accuracy'], label='Точность (обучение)')
plt.plot(history_base.history['val_accuracy'], label='Точность (валидация)')
plt.title('Модель БЕЗ Dropout (Переобучение)')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# График 2 (С Dropout, Adam)
plt.subplot(1, 3, 2)
plt.plot(history_regularized.history['accuracy'], label='Точность (обучение)')
plt.plot(history_regularized.history['val_accuracy'], label='Точность (валидация)')
plt.title('Модель С Dropout (Optimizer: Adam)')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# График 3 (С Dropout, RMSprop)
plt.subplot(1, 3, 3)
plt.plot(history_experimental.history['accuracy'], label='Точность (обучение)')
plt.plot(history_experimental.history['val_accuracy'], label='Точность (валидация)')
plt.title('Модель С Dropout (Optimizer: RMSprop)')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Показываем второй набор графиков (Точность)
plt.suptitle('Сравнение метрик точности (Accuracy) для трех моделей', fontsize=16)
plt.show()

# =========================================================================
# --- ИТОГОВАЯ ОЦЕНКА ТРЕХ МОДЕЛЕЙ ---
# =========================================================================

# --- Оценка модели 1 (Базовая) ---
print("\n\n--- Итоговая оценка: Базовая модель (без Dropout) ---")
print(f"Точность на тестовых данных: {test_acc_base:.4f}")
print("Отчет о классификации (без Dropout):")
print(classification_report(y_test, y_pred_base, target_names=class_names))


# --- Оценка модели 2 (Dropout + Adam) ---
print("\n--- Итоговая оценка: Модель с Dropout (Adam) ---")
test_loss, test_acc = model_regularized.evaluate(x_test, y_test, verbose=2)
print(f"\nТочность на тестовых данных (с Dropout, Adam): {test_acc:.4f}")

y_pred_probs = model_regularized.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\n--- Отчет о классификации (с Dropout, Adam) ---")
print(classification_report(y_test, y_pred, target_names=class_names))

# --- Оценка модели 3 (Dropout + RMSprop) ---
print("\n--- Итоговая оценка: Модель с Dropout (RMSprop) ---")
test_loss_exp, test_acc_exp = model_experimental.evaluate(x_test, y_test, verbose=2)
print(f"\nТочность на тестовых данных (с Dropout, RMSprop): {test_acc_exp:.4f}")

y_pred_probs_exp = model_experimental.predict(x_test)
y_pred_exp = np.argmax(y_pred_probs_exp, axis=1)
print("\n--- Отчет о классификации (с Dropout, RMSprop) ---")
print(classification_report(y_test, y_pred_exp, target_names=class_names))
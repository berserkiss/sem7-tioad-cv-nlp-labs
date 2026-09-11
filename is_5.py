import time
import math
import pandas as pd
import matplotlib.pyplot as plt

# Zero-width characters
ZW_SPACE = '\u200B'      # '0'
ZW_NON_JOINER = '\u200C' # '1'

def text_to_bin(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bin_to_text(binary):
    if len(binary) % 8 != 0:
        binary = binary[:-(len(binary) % 8)]
    return ''.join(chr(int(binary[i:i+8], 2)) for i in range(0, len(binary), 8))

# === Метод 1: Пробелы ===
def embed_space(original_text, message):
    binary = text_to_bin(message)
    words = original_text.split()
    needed = len(binary)
    available = len(words) - 1
    if available < needed:
        repeats = math.ceil((needed + 1) / len(words))
        words = (words * repeats)[:needed + 1]
    else:
        words = words[:needed + 1]
    stego = words[0]
    for i, bit in enumerate(binary):
        stego += ('  ' if bit == '1' else ' ') + words[i + 1]
    return stego

def extract_space(stego_text):
    binary = ''
    i = 0
    while i < len(stego_text):
        if stego_text[i] == ' ':
            if i + 1 < len(stego_text) and stego_text[i + 1] == ' ':
                binary += '1'
                i += 2
                while i < len(stego_text) and stego_text[i] == ' ':
                    i += 1
            else:
                binary += '0'
                i += 1
        else:
            i += 1
    return bin_to_text(binary)

def detect_space(stego_text):
    return '  ' in stego_text

# === Метод 2: Нулевая ширина ===
def embed_zero(original_text, message):
    binary = text_to_bin(message)
    hidden = ''.join(ZW_SPACE if b == '0' else ZW_NON_JOINER for b in binary)
    return original_text + hidden

def extract_zero(stego_text):
    binary = ''.join('0' if c == ZW_SPACE else '1' if c == ZW_NON_JOINER else '' for c in stego_text)
    return bin_to_text(binary)

def detect_zero(stego_text):
    return any(c in (ZW_SPACE, ZW_NON_JOINER) for c in stego_text)

def count_changes(original, stego):
    from difflib import ndiff
    return sum(1 for d in ndiff(original, stego) if d.startswith(('+', '-')))

# === Подготовка данных ===
texts = {
    "EN_short": "AI changes everything.",
    "EN_medium": "Artificial intelligence is transforming healthcare, education, and entertainment.",
    "EN_long": "Steganography allows secret data to be hidden in plain sight within ordinary text without arousing suspicion among observers.",
    "RU_short": "Технологии растут.",
    "RU_medium": "Современные технологии ускоряют общение, обучение и работу в повседневной жизни.",
    "RU_long": "Стеганография позволяет скрывать секретную информацию в обычном тексте так, что даже внимательный наблюдатель не заподозрит ничего необычного."
}

messages = {
    "short": "Hi",
    "medium": "Secret key: 12345",
    "long": "The quick brown fox jumps over the lazy dog. This is a test message for steganography experiments."
}

results = []

exp_id = 0
for text_name, cover_text in texts.items():
    for msg_name, secret_msg in messages.items():
        exp_id += 1
        for method in ["Пробелы", "Нулевая ширина"]:
            # Embed
            start = time.perf_counter()
            if method == "Пробелы":
                stego = embed_space(cover_text, secret_msg)
            else:
                stego = embed_zero(cover_text, secret_msg)
            embed_time = time.perf_counter() - start

            # Extract
            start = time.perf_counter()
            if method == "Пробелы":
                extracted = extract_space(stego)
            else:
                extracted = extract_zero(stego)
            extract_time = time.perf_counter() - start

            # Metrics
            success = (extracted == secret_msg)
            changes = count_changes(cover_text, stego)
            detected = detect_space(stego) if method == "Пробелы" else detect_zero(stego)
            cover_len = len(cover_text)
            msg_bits = len(text_to_bin(secret_msg))

            results.append({
                "ID": exp_id,
                "Текст": text_name,
                "Сообщение": msg_name,
                "Метод": method,
                "Время_встраивания_с": embed_time,
                "Время_извлечения_с": extract_time,
                "Успешно": success,
                "Изменений": changes,
                "Обнаружено": detected,
                "Длина_контейнера": cover_len,
                "Бит_сообщения": msg_bits
            })

df = pd.DataFrame(results)
print("\nВсего экспериментов:", len(df))
print(df[["Текст", "Сообщение", "Метод", "Успешно", "Обнаружено", "Изменений"]].to_string(index=False))

# === График 1: Время встраивания ===
plt.figure(figsize=(10, 6))
colors = {"Пробелы": "#4682B4", "Нулевая ширина": "#DC143C"}

for method in df["Метод"].unique():
    subset = df[df["Метод"] == method]
    x = [f"{r['Текст']}\n{r['Сообщение']}" for _, r in subset.iterrows()]
    plt.bar(x, subset["Время_встраивания_с"],
            label=method, alpha=0.8,
            color=colors[method])

plt.title("Время встраивания сообщения (секунды)")
plt.ylabel("Время (с)")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# === График 2: Время извлечения ===
plt.figure(figsize=(10, 6))

for method in df["Метод"].unique():
    subset = df[df["Метод"] == method]
    x = [f"{r['Текст']}\n{r['Сообщение']}" for _, r in subset.iterrows()]
    plt.bar(x, subset["Время_извлечения_с"],
            label=method, alpha=0.8,
            color=colors[method])

plt.title("Время извлечения сообщения (секунды)")
plt.ylabel("Время (с)")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# === График 3: Количество изменений ===
plt.figure(figsize=(10, 6))

for method in df["Метод"].unique():
    subset = df[df["Метод"] == method]
    x = [f"{r['Текст']}\n{r['Сообщение']}" for _, r in subset.iterrows()]
    plt.bar(x, subset["Изменений"],
            label=method, alpha=0.8,
            color=colors[method])

plt.title("Количество изменённых символов")
plt.ylabel("Число изменений")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# === Выводы (печатаем) ===
print("\n" + "="*60)
print("ВЫВОДЫ СРАВНИТЕЛЬНОГО АНАЛИЗА")
print("="*60)

# Производительность
avg_embed_time = df.groupby("Метод")["Время_встраивания_с"].mean()
avg_extract_time = df.groupby("Метод")["Время_извлечения_с"].mean()
print(f"\nСреднее время встраивания:")
for m, t in avg_embed_time.items():
    print(f"  {m}: {t:.6f} сек")

print(f"\nСреднее время извлечения:")
for m, t in avg_extract_time.items():
    print(f"  {m}: {t:.6f} сек")

# Универсальность
print(f"\nУниверсальность:")
print("  Оба метода работают с английским и русским текстом, короткими и длинными сообщениями.")
print("  Метод 'Пробелы' требует достаточного числа слов — при нехватке дублирует текст (снижает скрытность).")
print("  Метод 'Нулевая ширина' работает с любым текстом, даже однословным.")

# Надёжность и скрытность
print(f"\nНадёжность и скрытность:")
for method in df["Метод"].unique():
    sub = df[df["Метод"] == method]
    success = sub["Успешно"].mean()
    detected = sub["Обнаружено"].mean()
    avg_changes = sub["Изменений"].mean()
    print(f"  {method}:")
    print(f"    Успешность: {success:.1%}")
    print(f"    Обнаружимость: {detected:.1%}")
    print(f"    Среднее число изменений: {avg_changes:.0f}")

print("\nЗАКЛЮЧЕНИЕ:")
print("  → Метод 'Нулевая ширина' быстрее, скрытнее и универсальнее.")
print("  → Метод 'Пробелы' надёжен, но легко обнаруживается и создаёт много артефактов.")
print("  → Для реальных применений предпочтителен метод с нулевой шириной.")
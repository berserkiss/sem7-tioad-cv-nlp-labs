import difflib
import math
import pandas as pd
import matplotlib.pyplot as plt

# Zero-width characters
ZW_SPACE = '\u200B'      # '0'
ZW_NON_JOINER = '\u200C' # '1'

def text_to_bin(text):
    """Convert text message to binary string."""
    return ''.join(format(ord(c), '08b') for c in text)

def bin_to_text(binary):
    """Convert binary string back to text."""
    if len(binary) % 8 != 0:
        binary = binary[:-(len(binary) % 8)]  # truncate to multiple of 8
    return ''.join(chr(int(binary[i:i + 8], 2)) for i in range(0, len(binary), 8))

def embed_space(original_text, message):
    """Embed message using single vs double spaces."""
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
    """Extract binary from space patterns."""
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

def embed_zero(original_text, message):
    """Embed message using zero-width characters."""
    binary = text_to_bin(message)
    hidden = ''.join(ZW_SPACE if b == '0' else ZW_NON_JOINER for b in binary)
    return original_text + hidden

def extract_zero(stego_text):
    """Extract message from zero-width chars."""
    binary = ''.join(
        '0' if c == ZW_SPACE else '1' if c == ZW_NON_JOINER else ''
        for c in stego_text
    )
    return bin_to_text(binary)

def detect_space(stego_text):
    return '  ' in stego_text

def detect_zero(stego_text):
    return any(c in (ZW_SPACE, ZW_NON_JOINER) for c in stego_text)

def count_changes(original, stego):
    diff = difflib.ndiff(original, stego)
    return sum(1 for d in diff if d.startswith(('+', '-')))

# === Main Experiment ===
if __name__ == "__main__":
    english_text = "Artificial intelligence is transforming the world, from healthcare and education to business and entertainment."
    russian_text = "Современные технологии изменяют нашу жизнь, делая общение, работу и обучение быстрее и удобнее."

    short_msg = "ok"
    medium_msg = "hidden code"
    long_msg = "Steganography allows secret information to be embedded into normal text without raising suspicion."

    experiments = [
        ("Английский", "Короткое", english_text, short_msg),
        ("Английский", "Среднее", english_text, medium_msg),
        ("Английский", "Длинное", english_text, long_msg),
        ("Русский", "Короткое", russian_text, short_msg),
        ("Русский", "Среднее", russian_text, medium_msg),
        ("Русский", "Длинное", russian_text, long_msg),
    ]

    results = []

    for lang, length, text, msg in experiments:
        # Space-based
        space_stego = embed_space(text, msg)
        space_extracted = extract_space(space_stego)
        space_ok = (space_extracted == msg)
        space_changes = count_changes(text, space_stego)
        space_detected = detect_space(space_stego)

        # Zero-width
        zero_stego = embed_zero(text, msg)
        zero_extracted = extract_zero(zero_stego)
        zero_ok = (zero_extracted == msg)
        zero_changes = count_changes(text, zero_stego)
        zero_detected = detect_zero(zero_stego)

        results.append({
            "Язык": lang,
            "Длина": length,
            "Метод": "Пробелы",
            "Успешно": space_ok,
            "Изменений": space_changes,
            "Обнаружено": space_detected
        })
        results.append({
            "Язык": lang,
            "Длина": length,
            "Метод": "Нулевая ширина",
            "Успешно": zero_ok,
            "Изменений": zero_changes,
            "Обнаружено": zero_detected
        })

    df = pd.DataFrame(results)
    print("\nСводная таблица результатов:")
    print(df.to_string(index=False))

    # === Визуализация ===
    fig, ax = plt.subplots(figsize=(10, 6))

    # Цвета
    colors = {'Пробелы': 'skyblue', 'Нулевая ширина': 'salmon'}

    for method in df['Метод'].unique():
        subset = df[df['Метод'] == method]
        x = [f"{row['Язык']}\n{row['Длина']}" for _, row in subset.iterrows()]
        y = subset['Изменений']
        detected = subset['Обнаружено']
        success = subset['Успешно']

        # Отображаем столбцы
        bars = ax.bar(x, y, label=method, color=colors[method], alpha=0.8)

        # Добавляем метки: ✅ / ❌ / ⚠️
        for i, bar in enumerate(bars):
            txt = ''
            if not success.iloc[i]:
                txt += '❌'
            elif detected.iloc[i]:
                txt += '⚠️'
            else:
                txt += '✅'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(y)*0.02,
                    txt, ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel("Количество изменённых символов")
    ax.set_title("Сравнение методов стеганографии: изменения и надёжность")
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
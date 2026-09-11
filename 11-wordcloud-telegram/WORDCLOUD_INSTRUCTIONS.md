# ☁️ WORD CLOUD FROM A TELEGRAM CHAT

## 📋 Description

This script builds a nice-looking word cloud from an exported Telegram chat. You
can pick one person's messages (mom, a friend, a partner) and build a word cloud
that reflects what they talk about most.

**Uses:**
- A personalized card
- Analyzing someone's communication style
- A birthday or holiday gift
- Visualizing conversations with people close to you

---

## 🚀 Quick start

### 1. Install dependencies

```bash
pip install -r requirements_wordcloud.txt
```

Or manually:
```bash
pip install wordcloud matplotlib pandas nltk pillow beautifulsoup4
```

### 2. Export the chat from Telegram

#### Option 1: Via Telegram Desktop

1. Open Telegram Desktop
2. Open the chat you want
3. Click the three-dot menu → **Export chat history**
4. Pick a format: **JSON** (recommended) or HTML
5. Save the file next to the script

#### Option 2: Use the sample

The script generates a sample chat automatically if no export file is found.

### 3. Run the script

```bash
python wordcloud_telegram.py
```

---

## 📁 Export formats

The script supports several formats:

### JSON (recommended)
- Filenames: `result.json` or `telegram_export.json`
- Structure: Telegram's standard export format

### HTML
- Filename: `.html`
- Parsed with BeautifulSoup

### Plain text
- Filename: `.txt`
- Format: `Name: Message text`

---

## 🎨 Configuring the word cloud

### Basic parameters

In `main()` you can change:

```python
generator = WordCloudGenerator(
    width=1200,           # image width
    height=600,           # image height
    background_color='white'  # background color
)

generator.generate(
    processed_text,
    output_path='wordcloud.png',
    max_words=100,        # max word count
    colormap='viridis'    # color scheme
)
```

### Available color schemes:

- `'viridis'` — blue-green (default)
- `'plasma'` — purple-pink
- `'inferno'` — fiery
- `'magma'` — magma-toned
- `'coolwarm'` — blue-red
- `'autumn'` — autumn tones
- `'winter'` — winter tones
- `'spring'` — spring tones
- `'summer'` — summer tones

---

## 🎭 Making a shaped word cloud

### Step 1: Prepare a mask

Create a black-and-white image:
- **White** — where words should go
- **Black** — where words shouldn't go

**Example shapes:**
- A heart ❤️
- A star ⭐
- A circle ⭕
- Any shape you like

### Step 2: Use the mask

```python
generator.generate(
    processed_text,
    output_path='wordcloud_heart.png',
    mask_path='heart_mask.png',  # path to the mask
    max_words=100,
    colormap='plasma'
)
```

### Where to get a mask:

1. **Make your own:**
   - Draw it in Paint/GIMP/Photoshop
   - Black-and-white image
   - Save as PNG

2. **Download one:**
   - Search "word cloud mask" online
   - Use a silhouette image

3. **Use an emoji:**
   - Convert an emoji to an image
   - Use it as the mask

---

## 📊 Usage examples

### Example 1: Word cloud for "Mom"

```python
# In the code, pick the sender "Mom"
selected_sender = "Mom"
```

**Result:** A word cloud of that person's most frequent words (e.g. "health",
"school", "family", "love")

### Example 2: A heart-shaped word cloud

```python
generator.generate(
    processed_text,
    output_path='wordcloud_heart.png',
    mask_path='heart.png',  # a heart image
    max_words=150,
    colormap='plasma'
)
```

### Example 3: Different color schemes

```python
# For a birthday gift
generator.generate(..., colormap='autumn')

# For a romantic gift
generator.generate(..., colormap='plasma')

# For a gift to a friend
generator.generate(..., colormap='viridis')
```

---

## 🎁 Making a card

### Step 1: Generate the word cloud

Run the script and get the word-cloud image.

### Step 2: Add text

Use any image editor (Canva, Photoshop, GIMP) to add:
- A greeting
- The recipient's name
- The date
- Any extra elements

### Step 3: Print it

- Size: A4 or whatever you like
- Quality: 300 DPI (already set in the script)
- Paper: matte or glossy

---

## 🔧 More settings

### Changing the size

```python
generator = WordCloudGenerator(
    width=2000,   # bigger = higher quality
    height=1000
)
```

### Changing the word count

```python
generator.generate(..., max_words=200)  # more words
```

### Changing relative scaling

In the code you can adjust `relative_scaling`:
- `0.5` — a more even distribution
- `1.0` — more contrast in word sizes

---

## 📝 File layout

```
.
├── wordcloud_telegram.py      # Main script
├── requirements_wordcloud.txt  # Dependencies
├── telegram_export.json       # Your Telegram export
├── wordcloud_Mom.png           # Result (plain)
└── wordcloud_freq_Mom.png      # Result (with frequencies)
```

---

## ⚠️ Important notes

### Stop-word removal

✅ **Already handled!** The script automatically:
- Removes stop words (and, in, on, with, etc.)
- Removes short words (under 3 characters)
- Keeps only meaningful words

### Result quality

For a good result you need:
- **At least 50-100 messages** from the chosen person
- **Varied text** (not just one-word replies)
- **A proper export** from Telegram

### Issues with Russian text

If words render incorrectly:
1. Install a font with Cyrillic support
2. Point the code at it:
   ```python
   font_path='path/to/russian_font.ttf'
   ```

---

## 💡 Tips

1. **For the best result:**
   - Use chats with plenty of messages
   - Pick someone who writes longer messages

2. **For a card:**
   - Use a heart-shaped (or other) mask
   - Pick a fitting color scheme
   - Add a personal note

3. **For analysis:**
   - Compare word clouds across different people
   - See which topics come up most often

---

## 🎯 Example results

### A word cloud for "Mom" might include:
- health
- school
- family
- love
- food
- sleep
- support

### A word cloud for a friend might include:
- meetup
- fun
- plans
- help
- support

---

## 🚀 Ready to go!

Just run:
```bash
python wordcloud_telegram.py
```

And get a nice-looking word cloud! ☁️✨

---

**Have fun making unique cards! 🎁**

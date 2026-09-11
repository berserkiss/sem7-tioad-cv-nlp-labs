# 📤 HOW TO EXPORT A CHAT FROM TELEGRAM

## 🚀 STEP-BY-STEP GUIDE

### Step 1: Install Telegram Desktop

If you don't have it yet:
1. Download it from the official site: https://desktop.telegram.org/
2. Install it and sign in

### Step 2: Open the chat you want

1. Launch Telegram Desktop
2. Find the chat you want (mom, a friend, a partner)
3. Open it

### Step 3: Export the chat history

1. **Click the three-dot menu** in the chat's top-right corner
2. Choose **"Export chat history"**
3. In the dialog that opens:
   - ✅ Pick a format: **JSON** (recommended) or HTML
   - ✅ Pick a range: **Entire history** or a specific period
   - ✅ Pick the type: **Personal chats**
   - ✅ Click **Export**

### Step 4: Save the file

1. Choose a folder to save it in
2. **IMPORTANT:** save it next to the `wordcloud_telegram.py` script
3. The file will be named `result.json` (or something else, if you set that)

### Step 5: Run the script

```bash
python wordcloud_telegram.py
```

The script will find `result.json` automatically and process it!

---

## 📁 WHERE DOES THE EXPORT FILE END UP?

After exporting, the file usually lands in:
- **Windows:** `C:\Users\YourName\Downloads\` or wherever you picked
- **Mac:** `~/Downloads/` or wherever you picked
- **Linux:** `~/Downloads/` or wherever you picked

**Filename:**
- `result.json` (the default name)
- Or whatever you set during export

---

## 🔧 POINTING AT THE FILE MANUALLY

If the file ends up somewhere else:

### Option 1: Move the file

Just copy `result.json` next to the script:
```
D:\Sem7\tioad\L1\11-wordcloud-telegram\
├── wordcloud_telegram.py
├── result.json  ← copy the file here
```

### Option 2: Edit the code

Open `wordcloud_telegram.py` and find (around line 480):

```python
export_files = [
    'telegram_export_sample.json',
    'telegram_export.json',
    'result.json',  # Telegram's default export filename
]
```

Add your file's path:
```python
export_files = [
    'telegram_export_sample.json',
    'telegram_export.json',
    'result.json',
    'C:/Users/YourName/Downloads/result.json',  # your path
]
```

---

## 📋 EXPORT FORMATS

### JSON (recommended) ✅

**Advantages:**
- Structured format
- Easy to parse
- Contains all the information

**Structure:**
```json
{
  "name": "Chat name",
  "type": "personal_chat",
  "messages": [
    {
      "date": "2024-01-15T10:00:00",
      "from": "Sender name",
      "text": "Message text"
    }
  ]
}
```

### HTML

**Advantages:**
- Opens in a browser
- Human-readable

**Drawbacks:**
- Harder to parse
- May need the parser adapted

---

## 🎯 PICKING A SPECIFIC PERSON

### Automatic

The script:
1. Finds every sender in the chat
2. Picks the first one (can be changed)

### Manual

Open `wordcloud_telegram.py` and find (around line 520):

```python
# Picks the first sender for the demo (could be made interactive)
selected_sender = list(senders)[0]
```

Change it to the name you want:
```python
# Pick a specific person
selected_sender = "Mom"  # or "Friend", "Partner", etc.
```

Or make it interactive:
```python
print("Available senders:")
for i, sender in enumerate(senders, 1):
    print(f"  {i}. {sender}")

choice = input("\nEnter the sender's number: ")
selected_sender = list(senders)[int(choice) - 1]
```

---

## ✅ VERIFYING THE EXPORT

### How to check the export is correct:

1. **Open `result.json`** in a text editor
2. **Check the structure:**
   - There should be a `"messages"` key
   - Each message should have `"from"` and `"text"` fields
   - The text should be there in whatever language you chatted in

### Example of a correct export:

```json
{
  "name": "Chat with Mom",
  "messages": [
    {
      "date": "2024-01-15T10:00:00",
      "from": "Mom",
      "text": "Hi, how are you?"
    },
    {
      "date": "2024-01-15T10:05:00",
      "from": "Me",
      "text": "All good!"
    }
  ]
}
```

---

## 🔍 TROUBLESHOOTING

### Issue 1: File not found

**Fix:**
- Make sure the file is next to the script
- Check the filename (should be `result.json` or one of the other names in
  the list)
- Or point the code at the full path

### Issue 2: Messages don't parse

**Fix:**
- Check the file format (should be JSON)
- Make sure it has a `"messages"` field
- Check the structure of each message

### Issue 3: Sender names look wrong

**Fix:**
- Telegram names can vary
- Check how the name is actually written in the export
- Use the exact name from the file

### Issue 4: Too few messages

**Fix:**
- A good word cloud needs at least 50-100 messages
- Export the entire chat history
- Pick a chat with plenty of messages

---

## 💡 TIPS

1. **For the best result:**
   - Export the entire chat history
   - Pick a chat with longer messages
   - At least 50-100 messages from the chosen person

2. **For a card:**
   - Pick a chat with someone close to you
   - Use a heart-shaped mask
   - Choose a fitting color scheme

3. **For analysis:**
   - Compare word clouds across different people
   - See which topics come up most often
   - Build several clouds to compare

---

## 🎯 QUICK CHECK

After exporting, check:

1. ✅ `result.json` is next to the script
2. ✅ It has a `"messages"` field
3. ✅ Messages have `"from"` and `"text"` fields
4. ✅ There are messages from the person you want

Then just run:
```bash
python wordcloud_telegram.py
```

---

## 📸 VISUAL GUIDE

### In Telegram Desktop:

```
[Chat] → [⋮] → [Export chat history] → [JSON] → [Export]
```

### Result:

```
📁 Script folder
├── wordcloud_telegram.py
└── result.json  ← the export file
```

---

**Done! Now you can use a real conversation! 🚀**

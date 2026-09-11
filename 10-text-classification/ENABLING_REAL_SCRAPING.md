# 🌐 ENABLING REAL WEB SCRAPING

## ❓ Why does it use the sample reviews?

**Reason:** No real URLs are set in the code, so the script falls back to the
sample reviews.

### The current code:

```python
review_urls = [
    # 'https://otzovik.com/reviews/...',  # COMMENTED OUT!
    # 'https://irecommend.ru/content/...',  # COMMENTED OUT!
]
```

**Result:** since the list is empty (or every URL is commented out), the script
falls back to the sample reviews.

---

## ✅ HOW TO ENABLE REAL SCRAPING

### Step 1: Find the relevant lines

Open `text_classification.py` and find lines **897-901**:

```python
review_urls = [
    # 'https://otzovik.com/reviews/...',  # replace with a real URL
    # 'https://irecommend.ru/content/...',  # replace with a real URL
    # 'https://market.yandex.ru/product/.../reviews',  # replace with a real URL
]
```

### Step 2: Uncomment and add real URLs

#### Option A: Scraping Otzovik

1. Go to https://otzovik.com
2. Find a page with reviews (e.g. a product's review page)
3. Copy the page's URL
4. Paste it into the code:

```python
review_urls = [
    'https://otzovik.com/reviews/12345_product_review/',  # replace with a real URL
]
```

**Example of a real URL:**
```
https://otzovik.com/reviews/iphone_15_pro_max_review/
```

#### Option B: Scraping Irecommend

1. Go to https://irecommend.ru
2. Find a page with reviews
3. Copy the URL
4. Paste it into the code:

```python
review_urls = [
    'https://irecommend.ru/content/product-review',  # replace with a real URL
]
```

**Example of a real URL:**
```
https://irecommend.ru/content/iphone-15-pro-max-review
```

#### Option C: Scraping Yandex.Market

1. Go to https://market.yandex.ru
2. Find a product with reviews
3. Open the "Reviews" tab
4. Copy the URL
5. Paste it into the code:

```python
review_urls = [
    'https://market.yandex.ru/product/12345678/reviews',  # replace with a real URL
]
```

**Example of a real URL:**
```
https://market.yandex.ru/product/iphone-15/12345678/reviews
```

---

## 🔧 STEP-BY-STEP

### 1. Open `text_classification.py`

### 2. Find lines 897-901:

```python
review_urls = [
    # 'https://otzovik.com/reviews/...',  # <-- HERE
    # 'https://irecommend.ru/content/...',
    # 'https://market.yandex.ru/product/.../reviews',
]
```

### 3. Uncomment and swap in real URLs:

**Before:**
```python
review_urls = [
    # 'https://otzovik.com/reviews/...',  # commented out
]
```

**After:**
```python
review_urls = [
    'https://otzovik.com/reviews/12345_product_review/',  # a real URL
]
```

### 4. Save and run:

```bash
python text_classification.py
```

---

## 📋 EXAMPLES OF REAL URLS TO SCRAPE

### Otzovik (otzovik.com):

```
https://otzovik.com/reviews/iphone_15_pro_max_review/
https://otzovik.com/reviews/samsung_galaxy_s24_review/
https://otzovik.com/reviews/xiaomi_redmi_note_13_review/
```

**How to find one:**
1. Go to otzovik.com
2. Search for a product
3. Open a review
4. Copy the URL from the address bar

### Irecommend (irecommend.ru):

```
https://irecommend.ru/content/iphone-15-pro-max-review
https://irecommend.ru/content/samsung-galaxy-s24-review
https://irecommend.ru/content/xiaomi-redmi-note-13-review
```

**How to find one:**
1. Go to irecommend.ru
2. Find a product
3. Copy the URL of its review page

### Yandex.Market (market.yandex.ru):

```
https://market.yandex.ru/product/iphone-15/12345678/reviews
https://market.yandex.ru/product/samsung-galaxy-s24/87654321/reviews
```

**How to find one:**
1. Go to market.yandex.ru
2. Find a product
3. Open the "Reviews" tab
4. Copy the URL

---

## ⚠️ IMPORTANT NOTES

### 1. Site structure can change

- Sites may change their HTML structure
- The scraper may need updating
- Some sites may block scraping

### 2. Play by the rules

- ✅ Check the site's `robots.txt`
- ✅ Don't hammer it with requests
- ✅ Use it for coursework purposes
- ✅ Respect the site's terms of use

### 3. If scraping fails

The script automatically falls back to sample reviews if:
- The URL is unreachable
- The site blocks the request
- The site's structure has changed
- There's a network error

---

## 🎯 ALTERNATIVE: Scraping a single site

To scrape just one specific site, use option 3:

### Find lines 920-925:

```python
# OPTION 3: Scraping a specific site (alternative):
# Uncomment to scrape a single specific site:
# new_texts = scraper.scrape_from_url('https://otzovik.com/reviews/12345', site_type='otzovik')
# if not new_texts or len(new_texts) == 0:
#     print("Scraping failed, falling back to sample reviews")
#     new_texts = scraper.scrape_reviews_from_text("", num_reviews=15)
```

### Uncomment and set the URL:

```python
# OPTION 3: Scraping a specific site
new_texts = scraper.scrape_from_url(
    'https://otzovik.com/reviews/12345_product_review/',
    site_type='otzovik'
)
if not new_texts or len(new_texts) == 0:
    print("Scraping failed, falling back to sample reviews")
    new_texts = scraper.scrape_reviews_from_text("", num_reviews=15)
```

**Site types:**
- `'otzovik'` — for Otzovik
- `'irecommend'` — for Irecommend
- `'yandex_market'` — for Yandex.Market
- `'generic'` — for any other site

---

## 📊 WHAT HAPPENS DURING REAL SCRAPING

### If scraping succeeds:

```
[6/7] Scraping new data...
Trying the configured URLs...
Successfully pulled 15 reviews from the sites
Preprocessing the new data...
```

### If scraping fails:

```
[6/7] Scraping new data...
Trying the configured URLs...
Error scraping https://...: ...
URL scraping failed, falling back to sample reviews
```

---

## 🔍 VERIFYING IT WORKED

### After enabling real scraping:

1. **Run the script:**
   ```bash
   python text_classification.py
   ```

2. **Check the output:**
   - Should say: "Trying the configured URLs..."
   - Should say: "Successfully pulled X reviews from the sites"

3. **Check the results:**
   - The reviews should be real (not from the sample set)
   - The text should match the site's subject matter

---

## 💡 TIPS

### For the best results:

1. **Use several URLs:**
   ```python
   review_urls = [
       'https://otzovik.com/reviews/...',
       'https://otzovik.com/reviews/...',
       'https://irecommend.ru/content/...',
   ]
   ```

2. **Check the URLs work:**
   - Open each one in a browser
   - Make sure the page loads
   - Make sure it actually has reviews

3. **Start with easier sites:**
   - Otzovik usually scrapes cleanly
   - Yandex.Market can be trickier

---

## 🎯 BOTTOM LINE

**Right now:**
- ❌ No URLs set → sample reviews are used

**To enable real scraping:**
- ✅ Uncomment lines 897-901
- ✅ Add real URLs
- ✅ Save and run

**Result:**
- ✅ The script scrapes real reviews from the sites
- ✅ If scraping fails, it falls back to the samples

---

**Happy scraping! 🚀**

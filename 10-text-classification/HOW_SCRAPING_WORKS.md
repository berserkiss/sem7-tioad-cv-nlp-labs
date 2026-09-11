# 🔍 HOW THE REVIEW SCRAPER WORKS

## ❓ Do you need to copy a link to every single review?

### ✅ NO! One link to the reviews page is enough

The scraper automatically pulls **every review** off the page you point it at.

---

## 📋 HOW IT WORKS

### Option 1: One page with several reviews

**What to do:**
1. Find a product/service page that has reviews
2. Copy that page's URL
3. Paste it into the code

**Example:**
```python
review_urls = [
    'https://otzovik.com/reviews/contact_lens_solution_alcon_opti_fri_pure_moist/',
]
```

**What happens:**
- The scraper opens that page
- Finds every review on it
- Extracts each review's text
- Returns the full list

**Result:** Can pull 10-50+ reviews from a single page!

---

### Option 2: Several pages (for more reviews)

**What to do:**
1. Find several review pages
2. Copy each page's URL
3. Add them all to the list

**Example:**
```python
review_urls = [
    'https://otzovik.com/reviews/contact_lens_solution_alcon_opti_fri_pure_moist/',
    'https://otzovik.com/reviews/another_product/',
    'https://irecommend.ru/content/face-cream-bioderma-sebium-hydra',
]
```

**What happens:**
- The scraper processes each page
- Collects reviews from all of them
- Returns them combined

**Result:** Can pull 30-100+ reviews from a few pages!

---

## 🎯 REAL EXAMPLES

### Example 1: Otzovik

**Product page:**
```
https://otzovik.com/reviews/contact_lens_solution_alcon_opti_fri_pure_moist/
```

**On this page:**
- There may be 20-50 reviews
- All visible on a single page
- The scraper pulls them all automatically

**In the code:**
```python
review_urls = [
    'https://otzovik.com/reviews/contact_lens_solution_alcon_opti_fri_pure_moist/',
]
```

✅ **One link is enough!**

---

### Example 2: Irecommend

**Product page:**
```
https://irecommend.ru/content/face-cream-bioderma-sebium-hydra
```

**On this page:**
- There may be 10-30 reviews
- All on a single page
- The scraper pulls them all automatically

**In the code:**
```python
review_urls = [
    'https://irecommend.ru/content/face-cream-bioderma-sebium-hydra',
]
```

✅ **One link is enough!**

---

### Example 3: Several products

**For more reviews:**
```python
review_urls = [
    'https://otzovik.com/reviews/product_1/',
    'https://otzovik.com/reviews/product_2/',
    'https://irecommend.ru/content/product_3',
]
```

**Result:**
- Reviews from product 1
- Reviews from product 2
- Reviews from product 3
- All combined!

---

## ⚠️ IMPORTANT NOTES

### 1. You don't need links to individual reviews

❌ **WRONG:**
```python
review_urls = [
    'https://otzovik.com/reviews/12345/',  # review 1
    'https://otzovik.com/reviews/12346/',  # review 2
    'https://otzovik.com/reviews/12347/',  # review 3
]
```

✅ **RIGHT:**
```python
review_urls = [
    'https://otzovik.com/reviews/product/',  # the page with every review
]
```

---

### 2. Make sure the page actually has reviews

**Check:**
- Open the URL in a browser
- Confirm reviews are visible on the page
- If there are no reviews, the scraper won't find anything

---

### 3. Review count depends on the page

**Different sites show different amounts:**
- Otzovik: usually 20-50 reviews per page
- Irecommend: usually 10-30 reviews per page
- Yandex.Market: usually 10-20 reviews per page

**Need more?**
- Point it at several review pages
- Or use pagination (if the site supports it)

---

## 🔧 HOW THE SCRAPER EXTRACTS REVIEWS

### The process:

1. **Opens the page** at the given URL
2. **Looks for review blocks** (by HTML tags/classes)
3. **Extracts the text** from each block
4. **Filters out** overly short snippets
5. **Returns the list** of every review found

### Example run:

```
URL: https://otzovik.com/reviews/product/

The scraper finds:
- Review 1: "The product is good..."
- Review 2: "Excellent quality..."
- Review 3: "Wouldn't recommend..."
- ... (and so on)

Result: [review1, review2, review3, ...]
```

---

## 📊 THE CURRENT CODE

### What's already there:

```python
review_urls = [
    'https://otzovik.com/reviews/contact_lens_solution_alcon_opti_fri_pure_moist/',
    'https://irecommend.ru/content/face-cream-bioderma-sebium-hydra'
]
```

### This is correct! ✅

**What happens:**
1. The scraper opens the first page (Otzovik)
2. Extracts every review from it
3. Opens the second page (Irecommend)
4. Extracts every review from it
5. Combines everything

**Expected result:**
- 20-50 reviews from Otzovik
- 10-30 reviews from Irecommend
- **Total: 30-80 reviews!** 🎉

---

## 💡 TIPS

### For the best results:

1. **Use pages with lots of reviews**
   - More reviews = more data to classify

2. **Point it at several pages**
   - Different products = more varied reviews
   - More data = a better test of the model

3. **Check the pages are reachable**
   - Open each URL in a browser before running the script
   - Confirm the page actually loads

4. **Mix sites**
   - Otzovik + Irecommend = a mix of review styles
   - Better for testing how general the model is

---

## 🎯 BOTTOM LINE

### The short answer:

**No, you don't need a link to every review!**

✅ **All you need:**
- A link to the product page with its reviews
- The scraper pulls every review on that page automatically

✅ **You can add:**
- Several pages, for more reviews
- Different sites, for variety

✅ **The current code is already right:**
- Two pages = plenty of reviews automatically!

---

**Happy scraping! 🚀**

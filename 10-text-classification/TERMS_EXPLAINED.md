# 📚 TERMS EXPLAINED

## 1. 🔤 LEMMATIZATION

### What is it?

**Lemmatization** is the process of reducing a word to its **normal (dictionary)
form** — its **lemma**.

### In plain terms:

Lemmatization turns different forms of the same word into one base form.

### Examples:

| Original word | Lemma (base form) |
|----------------|--------------------------|
| products | product |
| product's | product |
| bought | buy |
| buying | buy |
| buys | buy |
| quicker | quick |
| quickest | quick |

### Why it matters:

#### Without lemmatization:
```
"The product is good"
"The products are good"
"Bought a good product"
```
The computer sees this as **3 different sets of words**!

#### With lemmatization:
```
"the product is good"
"the product is good"
"buy a good product"
```
The computer sees the shared words as **the same word**! ✅

### Benefits:

1. ✅ **Lower dimensionality** — fewer unique words
2. ✅ **Better understanding of meaning** — different forms get merged
3. ✅ **Better accuracy** — the model sees the connections between words
4. ✅ **Less memory** — fewer features to train on

### How this works in the code:

```python
# Without lemmatization:
"The products are good, quality is excellent"
→ ["products", "are", "good", "quality", "is", "excellent"]

# With lemmatization (pymorphy2):
"The products are good, quality is excellent"
→ ["product", "be", "good", "quality", "be", "excellent"]
```

### Visualization:

```
Original text:
"Buyers bought products, the product was good"

After lemmatization:
"buyer buy product product be good"
```

### Technical details:

- Uses **morphological analysis** of the word
- Accounts for **part of speech** (noun, verb, etc.)
- Reduces to the **dictionary form** (nominative case, infinitive)

---

## 2. 📦 BAG OF WORDS

### What is it?

**Bag of Words (BoW)** represents text as a **set of words with their
frequencies**, **ignoring word order**.

### In plain terms:

Imagine dumping every word from a text into a bag, shaking it, and now you just
count how many times each word shows up.

### Example:

#### Source texts:
```
Text 1: "Product is good, quality is great"
Text 2: "Product is bad, quality is terrible"
Text 3: "Product is good, delivery is fast"
```

#### Building the vocabulary:
```
All unique words: ["product", "good", "quality", "great",
                    "bad", "terrible", "delivery", "fast"]
```

#### Representing texts as vectors:

| Word | Text 1 | Text 2 | Text 3 |
|-------|---------|---------|---------|
| product | 1 | 1 | 1 |
| good | 1 | 0 | 1 |
| quality | 1 | 1 | 0 |
| great | 1 | 0 | 0 |
| bad | 0 | 1 | 0 |
| terrible | 0 | 1 | 0 |
| delivery | 0 | 0 | 1 |
| fast | 0 | 0 | 1 |

**Vectors:**
- Text 1: `[1, 1, 1, 1, 0, 0, 0, 0]`
- Text 2: `[1, 0, 1, 0, 1, 1, 0, 0]`
- Text 3: `[1, 1, 0, 0, 0, 0, 1, 1]`

### Visualization:

```
Source text:
"Product is good, quality is great"

Bag of Words:
{
  "product": 1,
  "good": 1,
  "quality": 1,
  "great": 1
}
```

### Advantages:

1. ✅ **Simple** — easy to understand and implement
2. ✅ **Fast** — cheap to compute
3. ✅ **Effective** — works well for many tasks
4. ✅ **Intuitive** — easy to see what's happening

### Drawbacks:

1. ❌ **Loses word order** — "not good" and "good, not" look the same
2. ❌ **Loses context** — ignores a word's surroundings
3. ❌ **Dimensionality** — the vocabulary can get very large
4. ❌ **Rare words** — can be important but get little weight

### How this works in the code:

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
texts = [
    "Product is good",
    "Product is bad"
]

vectors = vectorizer.fit_transform(texts)
# Result: a matrix of word counts
```

### A worked example:

**Text:** "Product is good, product is high-quality"

**Bag of Words:**
- product: 2 (appears twice)
- good: 1
- high-quality: 1

**Vector:** `[2, 1, 1]` (for the vocabulary ["product", "good", "high-quality"])

---

## 3. 📊 TF-IDF (Term Frequency - Inverse Document Frequency)

### What is it?

**TF-IDF** scores how **important a word** is to a document relative to the
whole document collection.

### In plain terms:

TF-IDF shows how **important** a word is in a specific text, weighted by how
**rare** it is across the whole collection.

### Two components:

---

### 3.1. TF (Term Frequency)

**What it is:** How often a word appears in a document.

#### Formula:

```
TF(t, d) = (number of times term t appears in document d) /
           (total number of words in document d)
```

#### Alternative formulas:

1. **Raw frequency:**
   ```
   TF = term count / total word count
   ```

2. **Log-scaled:**
   ```
   TF = log(1 + term count)
   ```

3. **Normalized:**
   ```
   TF = term count / max term count in the document
   ```

#### Example:

**Document:** "Product is good, product is high-quality, product is excellent"

- Total words: 6
- "product" appears: 3 times
- "good" appears: 1 time

**TF:**
- TF("product") = 3/6 = 0.5
- TF("good") = 1/6 = 0.167

#### Visualization:

```
Document: "Product is good, product is high-quality"

TF("product") = 2/4 = 0.5      (2 of 4 words)
TF("good") = 1/4 = 0.25         (1 of 4 words)
TF("high-quality") = 1/4 = 0.25
```

---

### 3.2. IDF (Inverse Document Frequency)

**What it is:** How **rare** a word is across the whole document collection.

#### Formula:

```
IDF(t, D) = log(total number of documents /
                number of documents containing term t)
```

#### In plain terms:

- A word in **many** documents → **low IDF** (uninformative)
- A word in **few** documents → **high IDF** (informative)

#### Example:

**A collection of 1000 documents:**

| Word | In how many documents | IDF |
|-------|----------------------|-----|
| "product" | 800 | log(1000/800) = 0.097 |
| "good" | 400 | log(1000/400) = 0.916 |
| "quantum" | 5 | log(1000/5) = 5.298 |

**Takeaway:**
- "product" — common word → low IDF (not very informative)
- "good" — medium frequency → medium IDF
- "quantum" — rare word → high IDF (very informative!)

#### Visualization:

```
Collection: 100 documents

Word "product": appears in 90 documents
IDF = log(100/90) = 0.046  (low - common word)

Word "excellent": appears in 10 documents
IDF = log(100/10) = 2.303  (high - rare word)
```

---

### 3.3. TF-IDF — combining the two

**Formula:**

```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
```

**Meaning:**
- High TF-IDF = the word is **frequent in this document** AND **rare in the
  collection**
- Low TF-IDF = the word is **rare in this document** OR **frequent in the
  collection**

---

### A full worked TF-IDF example:

#### Document collection:
```
Document 1: "Product is good, quality is great"
Document 2: "Product is bad, quality is terrible"
Document 3: "Product is good, delivery is fast"
```

#### Step 1: Compute TF for each word

**Document 1:**
- Total words: 4
- TF("product") = 1/4 = 0.25
- TF("good") = 1/4 = 0.25
- TF("quality") = 1/4 = 0.25
- TF("great") = 1/4 = 0.25

#### Step 2: Compute IDF

**Total documents: 3**

| Word | In how many documents | IDF |
|-------|----------------------|-----|
| product | 3 | log(3/3) = 0 |
| good | 2 | log(3/2) = 0.405 |
| quality | 2 | log(3/2) = 0.405 |
| great | 1 | log(3/1) = 1.099 |
| bad | 1 | log(3/1) = 1.099 |
| terrible | 1 | log(3/1) = 1.099 |
| delivery | 1 | log(3/1) = 1.099 |
| fast | 1 | log(3/1) = 1.099 |

#### Step 3: Compute TF-IDF

**Document 1:**

| Word | TF | IDF | TF-IDF |
|-------|----|-----|--------|
| product | 0.25 | 0 | 0.25 × 0 = **0** |
| good | 0.25 | 0.405 | 0.25 × 0.405 = **0.101** |
| quality | 0.25 | 0.405 | 0.25 × 0.405 = **0.101** |
| great | 0.25 | 1.099 | 0.25 × 1.099 = **0.275** |

**Takeaway:**
- "product" → TF-IDF = 0 (in every document, uninformative)
- "great" → TF-IDF = 0.275 (high — rare and important word!)

---

### TF-IDF visualization:

```
Document: "Product is good, quality is great"

TF (frequency in this document):
product: ████ (0.25)
good: ████ (0.25)
quality: ████ (0.25)
great: ████ (0.25)

IDF (rarity in the collection):
product: ░░░░ (0.0) - in every document
good: ████ (0.4) - medium rarity
quality: ████ (0.4) - medium rarity
great: ████████ (1.1) - very rare!

TF-IDF (importance):
product: ░░░░ (0.0) - unimportant
good: ██ (0.1) - somewhat important
quality: ██ (0.1) - somewhat important
great: ███ (0.28) - very important! ⭐
```

---

### Advantages of TF-IDF:

1. ✅ **Accounts for word importance** — rare words get more weight
2. ✅ **Downweights common words** — "the", "a", "is" get low weight
3. ✅ **Better than BoW** — accounts for the whole collection's context
4. ✅ **Effective** — works well for search and classification

### Drawbacks of TF-IDF:

1. ❌ **Still ignores word order** — same as BoW
2. ❌ **Ignores semantics** — doesn't understand synonyms
3. ❌ **Depends on the collection** — IDF shifts as documents are added

---

### BoW vs. TF-IDF

#### Example text: "Product is good, product is high-quality"

**Bag of Words:**
```
product: 2
good: 1
high-quality: 1
```
Every word carries the same weight.

**TF-IDF:**
```
product: 0.0 (if it's a common word)
good: 0.3 (if it's a rare word)
high-quality: 0.5 (if it's a very rare word)
```
Rare words get more weight! ⭐

---

### How this works in the code:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
texts = [
    "Product is good",
    "Product is bad"
]

# Computes TF-IDF automatically
vectors = vectorizer.fit_transform(texts)
```

**What happens under the hood:**
1. Compute TF for each word in each document
2. Compute IDF for each word across the collection
3. Multiply: TF × IDF = TF-IDF
4. Build a vector per document

---

## 📊 COMPARISON TABLE

| Property | Bag of Words | TF-IDF |
|----------------|--------------|--------|
| Accounts for frequency | ✅ Yes | ✅ Yes |
| Accounts for importance | ❌ No | ✅ Yes |
| Accounts for order | ❌ No | ❌ No |
| Complexity | Simple | Medium |
| Speed | Fast | Medium |
| Accuracy | Good | Better |
| Dimensionality | High | High |

---

## 🎯 PRACTICAL GUIDANCE

### When to use BoW:
- ✅ Simple classification tasks
- ✅ Speed matters
- ✅ Small datasets
- ✅ All words are equally important

### When to use TF-IDF:
- ✅ Document search
- ✅ Accuracy matters
- ✅ Large datasets
- ✅ Rare words matter
- ✅ **This project's case** (review classification) ✅

---

## 💡 KEY TAKEAWAYS

1. **Lemmatization** — reduces words to their dictionary form
   - "products" → "product"
   - Improves model quality

2. **Bag of Words** — counts word frequency
   - Simple and fast
   - Ignores order and importance

3. **TF-IDF** — accounts for word importance
   - TF — frequency within a document
   - IDF — rarity across the collection
   - TF-IDF = TF × IDF
   - Better than BoW for most tasks

---

**Good luck learning NLP! 🚀**

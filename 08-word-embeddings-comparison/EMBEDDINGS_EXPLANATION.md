# How the Embeddings and Code Work

## 1. How the chosen models produce embeddings

### Word2Vec (Google News)

**How it works:**
Word2Vec trains a neural network to learn word representations from the words'
context in text.

**Architecture:**
```
┌─────────────┐
│   Input     │  "king" (one-hot vector)
│    word     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Hidden     │  Weight matrix W (300 dimensions)
│   layer     │  → 300-dimensional vector
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Output     │  Predicted context words
│   layer     │  "man", "queen", "royal"...
└─────────────┘
```

**Two main approaches:**

1. **CBOW (Continuous Bag of Words)**:
   - Input: the context words around a target word
   - Output: predict the target word
   - Example: "The ___ is powerful" → predict "king"

2. **Skip-gram** (used by the Google News model):
   - Input: a single word
   - Output: predict its context words
   - Example: "king" → predict "man", "queen", "royal", "crown"

**Key idea:** Words that occur in similar contexts get similar vectors.

**Formula:**
```
P(context|word) = softmax(W_out × W_in × word_vector)
```

### GloVe (Global Vectors for Word Representation)

**How it works:**
GloVe combines global corpus statistics with local word context.

**Core idea:**
It uses a co-occurrence matrix that records how often words appear together
within a window of a given size.

**Algorithm:**
```
1. Build the co-occurrence matrix X:
   X[i,j] = how many times word j appears in the context of word i

2. Minimize the loss function:
   J = Σ f(X_ij) (w_i^T w_j + b_i + b_j - log(X_ij))²

   where:
   - w_i, w_j - word vectors
   - b_i, b_j - bias terms
   - f(X_ij) - a weighting function (accounts for frequency)

3. Result: word vectors that encode relationships between words
```

**Advantages:**
- Uses the whole corpus's global statistics
- Captures semantic relationships better (e.g. "king - man + woman = queen")
- Efficient on large corpora

**Diagram:**
```
Text corpus
    │
    ▼
┌─────────────────────┐
│ Co-occurrence       │  X[i,j] = co-occurrence frequency
│   matrix            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Optimize via       │  Minimize the loss function
│  gradient descent   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Word vectors       │  w_i ∈ R^d (d = dimensionality)
│  (embeddings)       │
└─────────────────────┘
```

## 2. Measuring semantic similarity between words

### Cosine similarity

**Main measure:** The cosine of the angle between two word vectors.

**Formula:**
```
cos(θ) = (A · B) / (||A|| × ||B||)

where:
- A · B - the dot product of the vectors
- ||A||, ||B|| - the vectors' norms (lengths)
```

**Value range:**
- **1.0** — the words are identical (or very similar)
- **0.0** — the words are orthogonal (unrelated)
- **-1.0** — the words are opposites

**Example from the code:**
```python
# find_similar_words uses:
similar = model.most_similar(word, topn=10)
# internally gensim computes:
similarity = cosine_similarity(word_vector, other_word_vector)
```

**Visualization:**
```
        word1 (vector A)
         /
        /
       / θ (angle)
      /
     /___________ word2 (vector B)

The smaller the angle θ, the larger cos(θ), the closer the words are semantically
```

### Alternative metrics:

1. **Euclidean distance:**
   ```
   d = √(Σ(a_i - b_i)²)
   ```
   Smaller distance = more similar

2. **Dot product:**
   ```
   similarity = A · B
   ```
   Used by some models

## 3. Training your own text embeddings

### Libraries

#### 1. **Gensim** (used in this code)
```python
from gensim.models import Word2Vec, FastText

# Prepare the data
sentences = [
    ['king', 'lives', 'in', 'palace'],
    ['queen', 'lives', 'in', 'palace'],
    # ... more sentences
]

# Train Word2Vec
model = Word2Vec(
    sentences=sentences,
    vector_size=300,      # vector dimensionality
    window=5,             # context window size
    min_count=2,          # minimum word frequency
    workers=4,            # thread count
    sg=1                  # 1 = Skip-gram, 0 = CBOW
)

# Save it
model.save("my_word2vec.model")
```

#### 2. **FastText** (from Facebook)
```python
from gensim.models import FastText

model = FastText(
    sentences=sentences,
    vector_size=300,
    window=5,
    min_count=2,
    workers=4
)
# FastText can handle out-of-vocabulary (OOV) words
```

#### 3. **TensorFlow / Keras**
```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding

# Create an embedding layer
embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    input_length=max_length
)
```

#### 4. **PyTorch**
```python
import torch
import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)
```

### Training process:

```
1. Prepare the data
   │
   ├─ Tokenize the text
   ├─ Remove stop words
   └─ Build the vocabulary

2. Build training pairs
   │
   ├─ For Skip-gram: (word, context_word)
   └─ For CBOW: (context_words, word)

3. Train the neural network
   │
   ├─ Initialize weights
   ├─ Forward pass
   ├─ Compute loss
   └─ Backpropagation

4. Extract the embeddings
   │
   └─ Hidden-layer weights = word vectors
```

## 4. How this project's code works

### Code structure:

```
embeddings_comparison.py
│
├─ EmbeddingModel (class)
│  │
│  ├─ __init__() - initializes the model
│  ├─ find_similar_words() - finds similar words
│  ├─ analogy() - runs an analogy
│  └─ get_vectors_for_words() - fetches vectors
│
├─ EmbeddingComparator (class)
│  │
│  ├─ load_word2vec() - loads Word2Vec
│  ├─ load_fasttext() - loads FastText/GloVe
│  ├─ compare_similar_words() - compares similar-word search
│  ├─ compare_analogies() - compares analogies
│  ├─ visualize_vectors() - visualization (t-SNE/PCA)
│  └─ test_analogies_batch() - batch testing
│
└─ main() - entry point
   │
   ├─ Load the models
   ├─ Test similar words
   ├─ Test analogies
   └─ Visualize
```

### Function walkthrough:

#### 1. Loading the models (`load_word2vec`, `load_fasttext`)

```
# How it works:
Check the cache
    │
    ├─ Model cached? → load from cache (fast)
    └─ Not cached? → download from the internet (slow)

Load into memory
    │
    ├─ Read the file from disk
    ├─ Parse the data
    └─ Build the model object

Initialize EmbeddingModel
    │
    └─ Compute vocab_size (vocabulary size)
```

#### 2. Finding similar words (`find_similar_words`)

```
# Algorithm:
1. Get the word's vector: word_vector = model[word]
2. Compute cosine similarity against every word:
   for each_word in vocabulary:
       similarity = cosine_similarity(word_vector, model[each_word])
3. Sort by similarity, descending
4. Return the top N
```

**Diagram:**
```
Input word: "king"
    │
    ▼
┌─────────────────┐
│  Word vector    │  [0.2, -0.1, 0.5, ...] (300 dimensions)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Compute cosine similarity       │
│  against every word              │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Sort by similarity,             │
│  descending                      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Top 10 results:                 │
│  1. kings (0.7138)               │
│  2. queen (0.6511)               │
│  3. monarch (0.6413)             │
│  ...                             │
└─────────────────────────────────┘
```

#### 3. Running analogies (`analogy`)

**Idea:** Vector arithmetic in embedding space.

**Formula:**
```
result_vector = word3_vector + word2_vector - word1_vector
result = most_similar(result_vector)
```

**Example:** "king - man + woman = ?"

```
1. Get the vectors:
   king_vec = [0.2, -0.1, 0.5, ...]
   man_vec = [0.1, 0.2, -0.3, ...]
   woman_vec = [0.15, 0.25, -0.2, ...]

2. Compute:
   result = woman_vec + (king_vec - man_vec)
          = woman_vec + [0.1, -0.3, 0.8, ...]
          = [0.25, -0.05, 0.6, ...]

3. Find the nearest vector:
   → "queen" (similarity: 0.7609)
```

**Diagram:**
```
Analogy: king - man = queen - ?
    │
    ├─ king_vec ────┐
    ├─ man_vec ─────┤
    └─ queen_vec ───┤
                    │
                    ▼
        ┌───────────────────────┐
        │  Vector               │
        │  arithmetic:          │
        │  queen + (king - man) │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Find nearest         │
        │  vector               │
        └───────────┬───────────┘
                    │
                    ▼
            Result: "woman"
```

#### 4. Visualization (`visualize_vectors`)

**t-SNE (t-Distributed Stochastic Neighbor Embedding):**
```
High-dimensional vectors (300D)
    │
    ▼
┌──────────────────────┐
│  t-SNE algorithm:     │
│  1. Compute           │
│     probabilities     │
│     in high-dim       │
│     space             │
│  2. Minimize          │
│     divergence        │
│     from the          │
│     low-dim           │
│     space             │
└──────────┬────────────┘
           │
           ▼
Low-dimensional vectors (2D)
    │
    ▼
┌──────────────────────┐
│  Plot on a chart      │
└──────────────────────┘
```

**PCA (Principal Component Analysis):**
```
Word vectors (300D)
    │
    ▼
┌──────────────────────┐
│  Compute the          │
│  principal            │
│  components:          │
│  1. Covariance         │
│     matrix             │
│  2. Eigenvectors       │
│  3. Project onto       │
│     the first 2        │
│     components         │
└──────────┬────────────┘
           │
           ▼
2D vectors - projection
    │
    ▼
┌──────────────────────┐
│  Plot it              │
└──────────────────────┘
```

### main()'s execution flow:

```
1. Setup
   │
   ├─ Create the EmbeddingComparator
   └─ Print caching info

2. Load the models
   │
   ├─ load_word2vec()
   │  ├─ Check the cache
   │  ├─ Load the model
   │  └─ Build an EmbeddingModel
   │
   └─ load_fasttext()
      ├─ Try loading a lightweight model
      ├─ Check the cache
      └─ Load the model

3. Test similar words
   │
   ├─ For each test word:
   │  ├─ compare_similar_words()
   │  │  ├─ find_similar_words() for Word2Vec
   │  │  └─ find_similar_words() for GloVe
   │  └─ Print the results

4. Test analogies
   │
   ├─ For each analogy:
   │  ├─ compare_analogies()
   │  │  ├─ analogy() for Word2Vec
   │  │  └─ analogy() for GloVe
   │  └─ Print the results
   │
   └─ test_analogies_batch()
      └─ Compute accuracy

5. Visualize
   │
   ├─ visualize_vectors(..., method='tsne')
   │  ├─ Fetch the word vectors
   │  ├─ Apply t-SNE
   │  └─ Save the plot
   │
   └─ visualize_vectors(..., method='pca')
      ├─ Fetch the word vectors
      ├─ Apply PCA
      └─ Save the plot

6. Conclusions
   └─ Compare the models and give recommendations
```

## Conclusions per model

### Word2Vec:
- ✅ Fast to load and use
- ✅ Works well on frequent words
- ❌ Can't handle rare (OOV) words

### GloVe:
- ✅ Uses global corpus statistics
- ✅ Better at semantic analogies
- ✅ More stable results

### FastText:
- ✅ Handles rare words
- ✅ Can handle OOV words
- ✅ Better for morphologically rich languages

## Recommendations

1. **For English:** Word2Vec or GloVe
2. **For rare words:** FastText
3. **For morphologically rich languages:** FastText
4. **For semantic analogies:** GloVe gives the best results

# Embedding Quality Comparison

This script compares different embedding models (Word2Vec and FastText) on semantic
similarity search and analogy tasks.

## Install dependencies

```bash
pip install -r requirements_embeddings.txt
```

Or manually:
```bash
pip install gensim matplotlib scikit-learn numpy
```

## Usage

```bash
python embeddings_comparison.py
```

## What the script does

1. **Loads pretrained models:**
   - Word2Vec (Google News) — ~3M words
   - FastText (Wiki News) — ~1M words

2. **Finds semantically similar words:**
   - For a given word, finds the top 10 most similar words
   - Compares results between the two models

3. **Runs analogies:**
   - Examples: "Paris - France = Minsk - ?" → "Belarus"
   - "Italy - pizza = Japan - ?" → "sushi"
   - Tests several analogies and computes accuracy

4. **Visualizes vectors:**
   - t-SNE for 2D dimensionality reduction
   - PCA as an alternative visualization
   - Plots 20-30 words from different categories

5. **Compares the models:**
   - Prints analogy-accuracy statistics
   - Summarizes each model's strengths

## Output

Running the script produces:
- `embeddings_tsne.png` — t-SNE visualization
- `embeddings_pca.png` — PCA visualization

## Notes

- On first run the models are downloaded automatically (can take a few minutes)
- Models are cached by gensim (typically under `~/.gensim/`)
- English models are used since they're the most available and best tested

## Example analogies in the script

1. `king - man = queen - ?` → `woman`
2. `paris - france = minsk - ?` → `belarus`
3. `italy - pizza = japan - ?` → `sushi`
4. `good - better = bad - ?` → `worse`
5. `computer - keyboard = car - ?` → `steering_wheel`

## Code structure

- `EmbeddingModel` — wraps a single model
- `EmbeddingComparator` — compares models against each other
- `main()` — runs the tests and produces the visualizations

## Conclusions

The script prints a comparison of the two models:
- **Word2Vec**: better for frequent words, faster, but can't handle out-of-vocabulary words
- **FastText**: handles rare/OOV words, better for morphologically rich languages

# Text Classification: Preprocessing and Model Training

A full pipeline for classifying Russian-language text: preprocessing, vectorization,
model training, and classifying new data.

## What it does

1. ✅ Load/download a text classification dataset
2. ✅ Preprocess text (tokenization, lemmatization, stop-word removal)
3. ✅ Vectorize (Bag of Words, TF-IDF)
4. ✅ Train several classifiers (Naive Bayes, SVM, Random Forest)
5. ✅ Pick the best model by its metrics
6. ✅ Scrape new data from websites
7. ✅ Classify the new data

## Install dependencies

```bash
pip install -r requirements_text_classification.txt
```

Or manually:
```bash
pip install nltk scikit-learn pandas numpy requests beautifulsoup4 pymorphy2
```

For Russian-language support you may also need:
```bash
pip install pymorphy2-dicts-ru
```

## Where to get a dataset

### Option 1: Use the built-in dataset (default)
The script generates a synthetic product-review dataset for demonstration.

### Option 2: Download a dataset from Kaggle

**Recommended datasets for Russian-language classification:**

1. **Russian Sentiment Analysis Dataset**
   - URL: https://www.kaggle.com/datasets/cryptexcode/russian-sentiment-analysis-dataset
   - A dataset of reviews labeled positive/negative

2. **Russian Reviews Dataset**
   - URL: https://www.kaggle.com/datasets/artemkonevskoy/russian-reviews-dataset
   - A larger dataset of Russian-language reviews

3. **Lenta.ru News Dataset**
   - URL: https://www.kaggle.com/datasets/yutkin/corpus-of-russian-news-articles-from-lenta
   - News with categories (for topic classification)

**How to download from Kaggle:**
1. Sign up on Kaggle (https://www.kaggle.com)
2. Get API credentials (Settings → API → Create New Token)
3. Install the client: `pip install kaggle`
4. Put `kaggle.json` under `~/.kaggle/` (or `C:\Users\YourUsername\.kaggle\` on Windows)
5. Download the dataset:
   ```bash
   kaggle datasets download -d dataset-name
   unzip dataset-name.zip
   ```

### Option 3: Use your own CSV

Create a CSV with columns:
- `text` — the text data
- `label` — the class label (0 or 1 for binary classification)

Then adjust `DatasetLoader.load_from_file()` to load your file.

### Option 4: Scrape data from websites

The script includes a `WebScraper` class you can adapt to scrape:
- Reviews from Ozon, Wildberries, Yandex.Market
- News from Lenta.ru, RIA Novosti
- Social media comments

**Example of adapting it for review scraping:**
```python
def scrape_ozon_reviews(product_url):
    # your scraping code
    pass
```

## Usage

### Basic usage

```bash
python text_classification.py
```

The script will:
1. Load the dataset
2. Preprocess it
3. Train the models
4. Pick the best one
5. Scrape and classify new data

### Using your own dataset

Adjust `main()` in the script:

```python
# Instead of:
df = loader.download_sentiment_dataset()

# Use:
df = loader.load_from_file('your_dataset.csv')
```

## Project layout

```
.
├── text_classification.py               # Main script
├── requirements_text_classification.txt  # Dependencies
├── text_classifier_model.pkl             # Saved model (created after training)
└── README_text_classification.md         # This file
```

## Preprocessing steps

1. **Tokenization** — splitting text into words
2. **Lemmatization** — reducing words to their dictionary form (via pymorphy2)
3. **Stop-word removal**
4. **Punctuation/special-character removal**
5. **Lowercasing**

## Vectorization methods

1. **Bag of Words (BoW)** — word-frequency counts
2. **TF-IDF** — Term Frequency-Inverse Document Frequency

## Classifiers

1. **Naive Bayes**
2. **SVM (Support Vector Machine)**
3. **Random Forest**

## Evaluation metrics

Used to pick the best model:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

## Output

Running the script prints:
- Metrics for each model
- The best model and vectorization method
- Classification results on the new (scraped) data

## Saving and loading the model

The model is saved automatically to `text_classifier_model.pkl`.

To load a saved model:
```python
classifier = TextClassifier()
classifier.load_model('text_classifier_model.pkl')
predictions = classifier.predict(new_texts)
```

## Notes

- The script uses a synthetic dataset by default, for demonstration
- For real projects, use a larger dataset from Kaggle or elsewhere
- Web scraping should respect each site's terms (robots.txt, rate limiting)
- For best results, use a dataset with 1000+ examples

## Possible improvements

1. Support multi-class classification
2. Add cross-validation for a more reliable evaluation
3. Add hyperparameter search (GridSearchCV)
4. Implement scraping for real sites (Ozon, Wildberries)
5. Add result visualizations (confusion matrix, ROC curve)

## License

Written for coursework purposes.

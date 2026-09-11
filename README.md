# TIOAD — Computer Vision & NLP Labs

Coursework labs covering classical computer vision and NLP/ML pipelines: edge and shape
detection, corner detection, face detection and embedding comparison, object counting
in video with YOLO, Russian-language text classification, and word cloud generation
from a Telegram chat export.

## Topics

| Area | Scripts | Notes |
| --- | --- | --- |
| Edge / shape detection | `l1_1.py`, `l1_2.py`, `l2.py`, `l2_3.py`, `l3_1.py`–`l3_4.py` | Canny, Sobel, Laplacian, Hough circles/lines |
| Corner detection | `L4.py`, `l5.py`, `l5_2.py` | Harris, Shi-Tomasi |
| Face detection & embeddings | `is_4.py`, `is_5.py`, `embeddings_comparison.py` | Haar cascades, Word2Vec/FastText comparison — see `EMBEDDINGS_EXPLANATION.md` |
| Vehicle counting | `task1_yolo_car_count.py` | YOLOv8 on traffic video |
| Text classification | `text_classification.py` | Russian NLP pipeline (tokenize → lemmatize → TF-IDF → classify) — see `README_text_classification.md` |
| Word cloud | `wordcloud_telegram.py` | Parses a Telegram export into a word cloud — see `КАК_ЭКСПОРТИРОВАТЬ_ИЗ_TELEGRAM.md` |

Each area with its own setup notes has a matching `README_*.md` / `ИНСТРУКЦИЯ*` file
alongside it.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # or `source .venv/bin/activate` on macOS/Linux
pip install -r requirements_embeddings.txt
pip install -r requirements_text_classification.txt
pip install -r requirements_wordcloud.txt
```

## Not tracked in this repo

A few inputs are downloaded/generated rather than committed (see `.gitignore`):

- `datasets/sentiment_dataset.csv` — download via `download_dataset.py`
- Haar cascade XML files — ship with `opencv-python` (`cv2.data.haarcascades`)
- `yolov8n.pt` — auto-downloaded by `ultralytics` on first run
- Traffic/pedestrian `.mp4` test videos and trained `.pkl` models — regenerate locally

Sample outputs (`output/`, `images/`, `*_result.jpg`, `wordcloud_*.png`) are kept small
enough to commit and show what each script produces.

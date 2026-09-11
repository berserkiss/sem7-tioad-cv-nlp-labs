# TIOAD — Computer Vision & NLP Labs

Coursework labs covering classical computer vision and NLP/ML pipelines: edge and shape
detection, corner detection, face detection and embedding comparison, object counting
in video with YOLO, Russian-language text classification, and word cloud generation
from a Telegram chat export.

## Structure

| Folder | Topic | Notes |
| --- | --- | --- |
| `01-edge-shape-detection/` | Canny, Sobel, Laplacian, Hough circles/lines | `l1_1.py`–`l3_4.py` |
| `02-corner-detection/` | Harris, Shi-Tomasi | `L4.py`, `l5.py`, `l5_2.py` |
| `03-face-detection-embeddings/` | Haar-cascade face detection, Word2Vec vs FastText | see its `README_embeddings.md` |
| `04-yolo-vehicle-counting/` | Vehicle counting in traffic video with YOLOv8 | |
| `05-text-classification/` | Russian NLP pipeline: tokenize → lemmatize → TF-IDF → classify | see its `README_text_classification.md` |
| `06-wordcloud-telegram/` | Parses a Telegram export into a word cloud | see its `ИНСТРУКЦИЯ_ОБЛАКО_СЛОВ.md` |

Each folder is self-contained: its script(s), its own `requirements*.txt` where it
needs one, its sample input images, and its result output.

## Setup

Each topic folder installs independently, e.g.:

```bash
python -m venv .venv
.venv\Scripts\activate   # or `source .venv/bin/activate` on macOS/Linux
pip install -r 03-face-detection-embeddings/requirements_embeddings.txt
```

## Not tracked in this repo

A few inputs are downloaded/generated rather than committed (see `.gitignore`):

- The sentiment dataset CSV — download via `05-text-classification/download_dataset.py`
- Haar cascade XML files — ship with `opencv-python` (`cv2.data.haarcascades`)
- `yolov8n.pt` — auto-downloaded by `ultralytics` on first run
- Traffic/pedestrian `.mp4` test videos and trained `.pkl` models — regenerate locally

Sample result images kept in each folder are small enough to commit and show what
each script produces.

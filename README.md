# TIOAD — Computer Vision & NLP Labs

Coursework labs covering classical computer vision, deep learning, and NLP/ML pipelines.

## Structure

| Folder | Topic |
| --- | --- |
| `01-image-fundamentals/` | Grayscale/binary thresholding, histogram equalization, noise filtering, morphology |
| `02-edge-shape-detection/` | Canny/Sobel/Laplacian edges, Hough lines/circles, document perspective alignment |
| `03-corner-detection/` | Harris and Shi-Tomasi corner detection |
| `04-face-detection/` | Haar-cascade face/eye/smile detection — static image and webcam |
| `05-motion-object-tracking/` | Contour-based motion detection from a video/camera feed |
| `06-cnn-image-classification/` | Fashion-MNIST CNN classifier, with and without Dropout |
| `07-zerowidth-steganography/` | Hiding text in zero-width Unicode characters (two iterations of the same idea) |
| `08-word-embeddings-comparison/` | Word2Vec vs FastText on similarity and analogy tasks |
| `09-yolo-vehicle-counting/` | Vehicle counting in traffic video with YOLOv8 |
| `10-text-classification/` | Russian NLP pipeline: tokenize → lemmatize → TF-IDF → classify |
| `11-wordcloud-telegram/` | Parses a Telegram export into a word cloud |

Each folder is self-contained: its script(s), a `requirements*.txt` where it needs one
beyond the base CV stack, its sample input images, and its result output.

## Setup

```bash
pip install opencv-python numpy matplotlib scikit-learn tensorflow ultralytics
```

Folders with extra dependencies carry their own `requirements*.txt`
(`08-word-embeddings-comparison/`, `10-text-classification/`, `11-wordcloud-telegram/`).

## Known gaps (pre-existing, not fixed here)

- `01-image-fundamentals/binarization_thresholding.py` points at a hardcoded personal
  path (`scottish-fold-2.jpg`) that isn't part of this repo — point `image_path` at any
  local image to run it.
- `04-face-detection/` expects Haar cascade XML files under a `haarcascades/` folder
  next to the scripts. They ship with `opencv-python`
  (`cv2.data.haarcascades` gives you the install path) — copy the ones you need
  (`haarcascade_frontalface_default.xml`, `haarcascade_eye.xml`, `haarcascade_smile.xml`)
  in rather than downloading them separately.

## Not tracked in this repo

- The sentiment dataset CSV — download via `10-text-classification/download_dataset.py`
- Haar cascade XML files (see above)
- `yolov8n.pt` — auto-downloaded by `ultralytics` on first run
- Traffic/pedestrian `.mp4` test videos and trained `.pkl` models — regenerate locally

Sample result images kept in each folder are small enough to commit and show what
each script produces.

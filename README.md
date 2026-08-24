# Webnovel Tag / Genre Predictor

Predict webnovel genres and tags from a title and description using different machine learning models.

## Structure

```
webnovel_predictor/
├── app.py                # FastAPI app (routes + serves the frontend)
├── src/
│   ├── preprocessing.py  # text cleaning / tag parsing (shared by train + predict)
│   ├── train.py          # CLI training script -> saves a .pkl bundle
│   ├── predict.py        # runs predictions given a loaded model bundle
│   └── registry.py       # auto-discovers models/*.pkl, loads + caches them
├── models/                # .pkl model bundles go here
├── data/                  # put training CSVs here (optional)
├── notebooks/             # colab notebooks
├── static/
│   └── index.html         # single-page frontend
└── requirements.txt
```

## Models

**Model 1**: TF-IDF + One-vs-Rest Logistic Regression multi-label classification

Dataset:

Metadata from over 35,000 webnovels hosted on RoyalRoad.
Scraped with [novel-metadata-scraper](https://github.com/kevbot404/novel-metadata-scraper).

Pipeline:

1. **Text preprocessing**: Titles and descriptions are cleaned (lowercased, URLs removed, punctuation stripped). The title is repeated 3x during feature construction to give it more weight than the description.
2. **Feature extraction**: TF-IDF vectorization with n-grams (1-3), sublinear TF scaling, and a vocabulary cap of 100,000 features.
3. **Classification**: Each tag/genre gets its own binary classifier. A threshold (default 0.30) controls how confident the model must be before predicting a tag.
4. **Model bundle**: The trained vectorizer, classifier, label binarizer, and genre list are saved together as a single `.pkl` file.

## Setup

1. Put your existing `.pkl` file in `models/`
   (e.g. `models/webnovel_title_description_model.pkl`)
2. Install deps:
   ```
   pip install -r requirements.txt
   ```
3. Run:
   ```
   uvicorn app:app --reload
   ```
4. Open http://127.0.0.1:8000

## Training from a CSV

```
python -m src.train --csv data/novels.csv --out models/webnovel_v2.pkl --min-tag-count 100
```

The CSV needs `title`, `description`, `tags` columns (tags pipe-separated,
e.g. `Romance|Fantasy|Isekai`).

### Training Options

| Flag              | Default      | Description                                             |
| ----------------- | ------------ | ------------------------------------------------------- |
| `--csv`           | _(required)_ | Path to the training CSV                                |
| `--out`           | _(required)_ | Output path for the `.pkl` model bundle                 |
| `--test-size`     | `0.20`       | Fraction of data used for validation                    |
| `--seed`          | `42`         | Random seed for train/test split                        |
| `--min-tag-count` | `100`        | Minimum novel count a tag must appear in to be included |

### Training Output

Training prints:

- Row counts after each filtering step
- Tag frequency stats (unique tags before/after filtering, removed tags)
- A `classification_report` (precision/recall/F1) on the held-out test set
- The path where the model bundle was saved

## Frontend

The single-page frontend at `http://127.0.0.1:8000` provides:

- **Model selector**: Choose any model placed in `models/`
- **Title input**: The novel's title (optional but recommended)
- **Description input**: The novel's blurb/summary (optional but recommended)
- **Threshold slider**: Adjust prediction confidence cutoff from 0.05 to 0.90
- **Results panel**: Predicted tags shown with their probability percentages, sorted highest-first

### Tips for Good Results

- **Provide both title and description** for the most accurate predictions
- **Lower the threshold** if no tags are returned (try 0.15-0.20)
- **Raise the threshold** if too many low-confidence tags are shown
- Train on a **larger, high-quality dataset** for better genre coverage
- Filter `min-tag-count` higher for a **focused genre set**

## API

### `GET /`

Serves the frontend.

### `GET /api/models`

Returns a list of available model names:

```json
{ "models": ["webnovel_v1", "webnovel_v2"] }
```

### `POST /api/predict`

Request body:

```json
{
  "model": "webnovel_v1",
  "title": "The Reincarnated Villainess",
  "description": "A young woman is reborn in a fantasy world...",
  "threshold": 0.3
}
```

Response:

```json
{
  "predictions": [
    { "tag": "Isekai", "probability": 0.89 },
    { "tag": "Fantasy", "probability": 0.76 },
    { "tag": "Romance", "probability": 0.42 }
  ]
}
```

## Adding a New Model

1. Train or obtain a `.pkl` bundle
2. Drop it into the `models/` directory
3. Restart the uvicorn server
4. The new model appears automatically in the frontend selector

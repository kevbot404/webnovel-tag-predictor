# Webnovel Tag / Genre Predictor

Predict webnovel genres and tags from a title and description using different machine learning models.

## Structure

```
webnovel_predictor/
├── app.py                # FastAPI app (routes + serves the frontend)
├── src/
│   ├── preprocessing.py  # text cleaning / tag parsing (shared by train + predict)
│   ├── train.py          # CLI training script -> saves a .pkl bundle
│   ├── predict.py        # runs inference given a loaded model bundle
│   └── registry.py       # auto-discovers models/*.pkl, loads + caches them
├── models/                # put your .pkl model bundles here
├── data/                  # put training CSVs here (optional)
├── notebooks/             # colab notebooks
├── static/
│   └── index.html         # single-page frontend
└── requirements.txt
```

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

## Retraining from a CSV

```
python -m src.train --csv data/novels.csv --out models/webnovel_v2.pkl
```

The CSV needs `title`, `description`, `tags` columns (tags pipe-separated,
e.g. `Romance|Fantasy|Isekai`).

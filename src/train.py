"""
Train a webnovel title+description -> tags/genres model and save it as a
single .pkl bundle that predict.py can load.

Usage:
    python -m src.train --csv data/novels.csv --out models/webnovel_v1.pkl

The output filename becomes the model's name in the webapp (minus .pkl)
"""

import argparse
import joblib
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from .preprocessing import clean_dataframe


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    print("\n================================================")
    print("BASE CSV")
    print("================================================")
    print("Path:", csv_path)
    print("Rows in raw CSV:", len(df))
    print("Columns in raw CSV:", list(df.columns))

    return df


def train(
    csv_path: str,
    out_path: str,
    test_size: float = 0.20,
    seed: int = 42,
    min_tag_count: int = 100,
):
    df = load(csv_path)
    df = clean_dataframe(df, min_tag_count=min_tag_count)

    print(f"\nRows after cleaning: {len(df)}")

    all_tags = sorted(
        {
            tag
            for tags in df["tag_list"]
            for tag in tags
        }
    )

    print(f"Unique tags/genres: {len(all_tags)}")

    mlb = MultiLabelBinarizer(classes=all_tags)

    Y = mlb.fit_transform(df["tag_list"])

    X_all = df["clean_text"].values.reshape(-1, 1)

    # MultilabelStratifiedShuffleSplit (iterative-stratification package)
    # is a numpy-vectorized implementation so it stays fast on large datasets
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed
    )
    train_idx, test_idx = next(msss.split(X_all, Y))

    X_train = X_all[train_idx].ravel()
    X_test = X_all[test_idx].ravel()
    y_train = Y[train_idx]
    y_test = Y[test_idx]

    print("\n================================================")
    print("TRAIN / TEST SPLIT")
    print("================================================")
    print("Test size fraction:", test_size)
    print("Split method: MultilabelStratifiedShuffleSplit (multi-label stratified)")
    print("Random seed:", seed)
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\n================================================")
    print("VECTORIZATION")
    print("================================================")
    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print("Train matrix shape:", X_train_tfidf.shape)
    print("Test matrix shape:", X_test_tfidf.shape)

    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ), n_jobs=-1
    )

    print("\n================================================")
    print("TRAINING")
    print("================================================")
    print("Training...")

    model.fit(X_train_tfidf, y_train)

    print("Done.")

    predictions = model.predict(X_test_tfidf)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=mlb.classes_,
            zero_division=0
        )
    )

    model_data = {
        "model": model,
        "vectorizer": vectorizer,
        "mlb": mlb,
        "genres": all_tags,
        "min_tag_count": min_tag_count,
    }

    joblib.dump(
        model_data,
        out_path
    )

    print(f"Saved model bundle -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to training CSV (title, description, tags)"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Where to save the .pkl bundle, e.g. models/my_model.pkl"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--min-tag-count",
        type=int,
        default=100,
        help="Minimum number of novels a tag/genre must appear in"
    )

    args = parser.parse_args()

    train(args.csv, args.out, args.test_size, args.seed, args.min_tag_count)
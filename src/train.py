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

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from .preprocessing import (
    contains_non_english_characters,
    clean_title,
    clean_description,
    parse_tags,
)


def load_and_clean(csv_path: str, min_tag_count: int = 100,) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[["title", "description", "tags"]].copy()
    df = df.dropna(subset=["title", "description", "tags"])
    df = df[df["description"].str.strip().str.len() >= 10].copy()

    df["title"] = df["title"].astype(str)
    df["description"] = df["description"].astype(str)
    df["tags"] = df["tags"].astype(str)

    non_en_title = df["title"].apply(contains_non_english_characters)
    non_en_desc = df["description"].apply(contains_non_english_characters)
    df = df[~non_en_title & ~non_en_desc].copy()

    df["clean_title"] = df["title"].apply(clean_title)
    df["clean_description"] = df["description"].apply(clean_description)
    df["clean_text"] = (
        df["clean_title"] + " " + df["clean_title"] + " " + df["clean_title"]
        + " " + df["clean_description"]
    )

    df["tag_list"] = df["tags"].apply(parse_tags)
    df = df[df["tag_list"].apply(len) > 0].copy()

    print(f"Rows with usable tags before frequency filtering: {len(df)}")

    # remove tags/genres with fewer than min_tag_count mentions
    tag_counts = {}

    for tags in df["tag_list"]:
        # count each tag only once per novel
        for tag in set(tags):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    allowed_tags = {
        tag
        for tag, count in tag_counts.items()
        if count >= min_tag_count
    }

    print("\n================================================")
    print("TAG / GENRE FREQUENCY FILTER")
    print("================================================")

    print("Minimum amount required:", min_tag_count)
    print("Unique tags/genres before filtering:", len(tag_counts))
    print("Unique tags/genres after filtering:", len(allowed_tags))

    removed_tags = sorted(
        tag
        for tag, count in tag_counts.items()
        if count < min_tag_count
    )

    print("\nRemoved tags/genres:")

    for tag in removed_tags:
        print(f"{tag:30} {tag_counts[tag]}")

    # remove low-frequency tags from each novel
    df["tag_list"] = df["tag_list"].apply(
        lambda tags: [
            tag
            for tag in tags
            if tag in allowed_tags
        ]
    )

    # remove rows with no tags after frequency filtering
    df = df[df["tag_list"].apply(len) > 0].copy()

    print("\nRows after frequency filtering:", len(df))

    return df


def train(
    csv_path: str,
    out_path: str,
    test_size: float = 0.20,
    seed: int = 42,
    min_tag_count: int = 100,
):
    df = load_and_clean(csv_path, min_tag_count=min_tag_count,)

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

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        Y,
        test_size=test_size,
        random_state=seed
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000
        ), n_jobs=-1
    )

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
"""
Text cleaning + tag parsing helpers.
"""

import re

import pandas as pd


def contains_non_english_characters(text: str) -> bool:
    for char in str(text):
        if char.isalpha() and not ("a" <= char.lower() <= "z"):
            return True
    return False


def _clean(text: str) -> str:
    text = text.lower()
    # remove URLs
    text = re.sub(r"https?://\S+", " ", text)
    # remove HTML tags (e.g. leftover <div>, <br>, decorative <----- bars)
    text = re.sub(r"<[^>]+>", " ", text)
    # remove HTML entities (e.g. &amp; &quot; &#39;)
    text = re.sub(r"&\w+;|&#\d+;", " ", text)
    # strip punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_title(title: str) -> str:
    return _clean(title)


def clean_description(description: str) -> str:
    return _clean(description)


def build_clean_text(title: str, description: str, title_repeat: int = 3) -> str:
    """Combine title (repeated) + description, same weighting used at train time."""
    ct = clean_title(title)
    cd = clean_description(description)
    return " ".join([ct] * title_repeat + [cd])


def parse_tags(tags: str) -> list[str]:
    """'Romance|Fantasy|Isekai' -> ['Romance', 'Fantasy', 'Isekai']"""
    return [t.strip() for t in str(tags).split("|") if t.strip()]


def clean_dataframe(df: pd.DataFrame, min_tag_count: int = 100, min_description_length: int = 40,) -> pd.DataFrame:
    """Apply column selection + text cleaning + tag parsing/frequency filtering
    to a raw, just-loaded dataframe."""

    # keep only the columns the model needs
    df = df[["title", "description", "tags"]].copy()

    # drop rows with missing title/description/tags
    before = len(df)
    df = df.dropna(subset=["title", "description", "tags"]).copy()
    print(f"Rows dropped for missing title/description/tags: {before - len(df)}")

    df["title"] = df["title"].astype(str)
    df["description"] = df["description"].astype(str)
    df["tags"] = df["tags"].astype(str)

    # drop exact duplicate (title, description) pairs
    before = len(df)
    df = df.drop_duplicates(subset=["title", "description"]).copy()
    print(f"Rows dropped as duplicate title+description: {before - len(df)}")

    # drop placeholder/junk descriptions ("Please delete, thx.", ".", "Dropped.", etc.)
    before = len(df)
    df = df[df["description"].str.strip().str.len() >= min_description_length].copy()
    print(f"Rows dropped for too-short/placeholder description: {before - len(df)}")

    # drop non-English titles/descriptions
    before = len(df)
    is_non_english = df["title"].apply(contains_non_english_characters) | df[
        "description"
    ].apply(contains_non_english_characters)
    df = df[~is_non_english].copy()
    print(f"Rows dropped for non-English title/description: {before - len(df)}")

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

    return df
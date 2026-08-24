"""
Text cleaning + tag parsing helpers.
"""

import re


def contains_non_english_characters(text: str) -> bool:
    for char in str(text):
        if char.isalpha() and not ("a" <= char.lower() <= "z"):
            return True
    return False


def _clean(text: str) -> str:
    text = text.lower()
    # remove URLs
    text = re.sub(r"https?://\S+", " ", text)
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

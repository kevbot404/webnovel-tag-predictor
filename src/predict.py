"""
Runs prediction using a loaded model bundle (from registry.get_model).
"""

from .preprocessing import clean_title, clean_description, build_clean_text


def predict_tags(bundle: dict, title: str, description: str, threshold: float = 0.30):
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    mlb = bundle["mlb"]

    cleaned_text = build_clean_text(title, description, 3)
    X = vectorizer.transform([cleaned_text])

    probabilities = model.predict_proba(X)[0]

    results = sorted(zip(mlb.classes_, probabilities), key=lambda x: x[1], reverse=True)

    return [
        {"tag": tag, "probability": float(prob)}
        for tag, prob in results
        if prob >= threshold
    ]

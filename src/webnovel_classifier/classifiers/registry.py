from .tfidf_logreg import TfidfLogisticPredictor


def get_predictors(model_dir):
    return {
        "tfidf-logistic": TfidfLogisticPredictor(
            model_dir / "tfidf-logistic.pkl"
        ),
    }
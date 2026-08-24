"""
Auto-discovers model bundles in models/*.pkl so the webapp can list them
and let the user pick one. Adding a new model = dropping a new .pkl in
models/ and restarting the server. No code changes required.

Loaded bundles are cached in memory after first use.
"""

from pathlib import Path
import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# cache so it doesn't repeatedly deserialize the potentially large model from disk.
_cache: dict[str, dict] = {}


def list_models() -> list[str]:
    """Return available model names (filename without .pkl), sorted."""
    if not MODELS_DIR.exists():
        return []
    return sorted(p.stem for p in MODELS_DIR.glob("*.pkl"))


def get_model(name: str) -> dict:
    """Load (and cache) a model bundle by name."""
    if name in _cache:
        return _cache[name]

    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No model named '{name}' found in {MODELS_DIR}")

    bundle = joblib.load(path)
    _cache[name] = bundle
    return bundle

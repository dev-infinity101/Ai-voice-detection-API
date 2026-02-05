import numpy as np


LANGUAGE_NORMALIZATION: dict[str, dict[str, float]] = {
    "Tamil": {"mean": -35.0, "std": 15.0},
    "English": {"mean": -34.0, "std": 14.5},
    "Hindi": {"mean": -36.0, "std": 15.5},
    "Malayalam": {"mean": -35.5, "std": 15.2},
    "Telugu": {"mean": -35.8, "std": 15.3},
}


def normalize_feature_vector(values: np.ndarray, language: str) -> np.ndarray:
    stats = LANGUAGE_NORMALIZATION.get(language)
    if stats is None:
        raise ValueError(f"Unsupported language: {language}")

    mean = float(stats["mean"])
    std = float(stats["std"])
    if std <= 0:
        std = 1.0

    return ((values - mean) / std).astype(np.float32, copy=False)


from dataclasses import dataclass

import librosa
import numpy as np


SUPPORTED_LANGUAGES: tuple[str, ...] = ("Tamil", "English", "Hindi", "Malayalam", "Telugu")


@dataclass(frozen=True)
class PreprocessParams:
    trim_top_db: int


LANGUAGE_PREPROCESS: dict[str, PreprocessParams] = {
    "Tamil": PreprocessParams(trim_top_db=28),
    "English": PreprocessParams(trim_top_db=32),
    "Hindi": PreprocessParams(trim_top_db=30),
    "Malayalam": PreprocessParams(trim_top_db=26),
    "Telugu": PreprocessParams(trim_top_db=29),
}


def preprocess_waveform(waveform: np.ndarray, sample_rate: int, language: str) -> np.ndarray:
    if language not in LANGUAGE_PREPROCESS:
        raise ValueError(f"Unsupported language: {language}")

    params = LANGUAGE_PREPROCESS[language]

    y = waveform.astype(np.float32, copy=False)

    y = librosa.util.normalize(y)

    y, _ = librosa.effects.trim(y, top_db=params.trim_top_db)

    if y.size == 0:
        raise ValueError("Audio is silent after preprocessing")

    return y

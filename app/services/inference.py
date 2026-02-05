import time
from dataclasses import dataclass

import numpy as np
import torch

from app.services.features import MelFeatureExtractor
from app.services.normalization import normalize_feature_vector
from app.services.preprocess import preprocess_waveform


@dataclass(frozen=True)
class InferenceResult:
    classification: str
    confidence_score: float
    probabilities: dict[str, float]
    language_detected: str
    duration_seconds: float
    processing_ms: float
    explanation: str


@torch.inference_mode()
def run_inference(
    *,
    model: torch.nn.Module,
    device: torch.device,
    feature_extractor: MelFeatureExtractor,
    waveform: np.ndarray,
    sample_rate: int,
    language: str,
    duration_seconds: float,
) -> InferenceResult:
    start = time.perf_counter()

    y = preprocess_waveform(waveform, sample_rate, language=language)
    features = feature_extractor.extract(y).values
    normed = normalize_feature_vector(features, language=language)

    x = torch.from_numpy(normed).to(device=device, dtype=torch.float32).unsqueeze(0)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1).squeeze(0).detach().to("cpu").numpy()

    p_human = float(probs[0])
    p_ai = float(probs[1])

    if p_ai >= 0.5:
        classification = "AI_GENERATED"
        confidence = p_ai
        explanation = "Model indicates synthetic speech characteristics in mel-spectral profile"
    else:
        classification = "HUMAN"
        confidence = p_human
        explanation = "Model indicates natural speech characteristics in mel-spectral profile"

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    confidence = float(min(max(confidence, 0.0), 1.0))

    return InferenceResult(
        classification=classification,
        confidence_score=confidence,
        probabilities={"human": p_human, "ai": p_ai},
        language_detected=language,
        duration_seconds=float(duration_seconds),
        processing_ms=float(elapsed_ms),
        explanation=explanation,
    )


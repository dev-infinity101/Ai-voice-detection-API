from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
from scipy import signal

from app.services.preprocess import preprocess_waveform


@dataclass(frozen=True)
class HeuristicResult:
    classification: str
    confidence_score: float
    probabilities: dict[str, float]
    explanation: str
    features: dict[str, float]


class HeuristicVoiceDetector:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.ai_indicators: dict[str, float] = {
            "pitch_variance_threshold": 0.2,
            "spectral_flatness_threshold": 0.18,
            "spectral_flatness_low_threshold": 0.05,
            "formant_stability_threshold": 0.7,
            "jitter_threshold": 0.03,
            "envelope_smoothness_threshold": 0.03,
            "decision_threshold": 0.5,
        }

    def extract_audio_features(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        features: dict[str, float] = {}

        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0_valid = f0[~np.isnan(f0)]
        if f0_valid.size > 10:
            features["pitch_mean_hz"] = float(np.mean(f0_valid))
            features["pitch_variance"] = float(np.std(f0_valid) / (np.mean(f0_valid) + 1e-8))
            features["jitter"] = float(np.mean(np.abs(np.diff(f0_valid))) / (np.mean(f0_valid) + 1e-8))
            features["voiced_ratio"] = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
            features["voiced_prob_mean"] = float(np.nanmean(voiced_prob)) if voiced_prob is not None else 0.0
        else:
            features["pitch_mean_hz"] = 0.0
            features["pitch_variance"] = 0.0
            features["jitter"] = 0.0
            features["voiced_ratio"] = 0.0
            features["voiced_prob_mean"] = 0.0

        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        features["spectral_flatness"] = float(np.mean(spectral_flatness))

        zcr = librosa.feature.zero_crossing_rate(audio)
        features["zero_crossing_rate"] = float(np.mean(zcr))

        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features["spectral_rolloff"] = float(np.mean(rolloff))

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features["mfcc_mean"] = float(np.mean(mfccs))
        features["mfcc_std"] = float(np.std(mfccs))

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features["formant_stability"] = float(1.0 - np.std(contrast) / (np.mean(contrast) + 1e-8))

        envelope = np.abs(signal.hilbert(audio))
        envelope_diff = np.diff(envelope)
        features["envelope_smoothness"] = float(np.std(envelope_diff))

        return features

    def calculate_ai_probability(self, features: dict[str, float]) -> tuple[float, str]:
        ai_score = 0.0
        indicators: list[str] = []

        if features["pitch_variance"] < self.ai_indicators["pitch_variance_threshold"]:
            ai_score += 0.35
            indicators.append("unnaturally consistent pitch")

        if features["spectral_flatness"] > self.ai_indicators["spectral_flatness_threshold"]:
            ai_score += 0.2
            indicators.append("flat spectral characteristics")

        if features["spectral_flatness"] < self.ai_indicators["spectral_flatness_low_threshold"]:
            ai_score += 0.25
            indicators.append("overly tonal spectrum (low noise floor)")

        if features["formant_stability"] > self.ai_indicators["formant_stability_threshold"]:
            ai_score += 0.25
            indicators.append("overly stable formants")

        if features["jitter"] < self.ai_indicators["jitter_threshold"]:
            ai_score += 0.15
            indicators.append("minimal pitch variation (jitter)")

        if features["envelope_smoothness"] < self.ai_indicators["envelope_smoothness_threshold"]:
            ai_score += 0.1
            indicators.append("unusually smooth temporal envelope")

        if ai_score >= self.ai_indicators["decision_threshold"]:
            explanation = f"AI-generated indicators detected: {', '.join(indicators)}"
        else:
            human_indicators: list[str] = []
            if features["pitch_variance"] >= self.ai_indicators["pitch_variance_threshold"]:
                human_indicators.append("natural pitch variation")
            if features["jitter"] >= self.ai_indicators["jitter_threshold"]:
                human_indicators.append("human-like voice quality")
            if features["formant_stability"] <= self.ai_indicators["formant_stability_threshold"]:
                human_indicators.append("natural formant dynamics")
            explanation = (
                f"Human voice characteristics: {', '.join(human_indicators)}"
                if human_indicators
                else "Human voice characteristics: natural speech patterns detected"
            )

        return float(ai_score), explanation

    def detect(self, waveform: np.ndarray, sr: int, language: str) -> HeuristicResult:
        y = preprocess_waveform(waveform, sr, language=language)
        features = self.extract_audio_features(y, sr)
        ai_probability, explanation = self.calculate_ai_probability(features)

        if ai_probability >= self.ai_indicators["decision_threshold"]:
            classification = "AI_GENERATED"
            confidence = ai_probability
        else:
            classification = "HUMAN"
            confidence = 1.0 - ai_probability

        confidence = float(min(max(confidence, 0.0), 1.0))
        return HeuristicResult(
            classification=classification,
            confidence_score=confidence,
            probabilities={"human": float(1.0 - ai_probability), "ai": float(ai_probability)},
            explanation=explanation,
            features=features,
        )

    def debug_payload(self, result: HeuristicResult) -> dict[str, Any]:
        return {
            "classification": result.classification,
            "confidenceScore": result.confidence_score,
            "probabilities": result.probabilities,
            "explanation": result.explanation,
            "features": result.features,
        }

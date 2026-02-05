import base64
import io
from typing import Dict, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


class VoiceDetector:
    def __init__(self) -> None:
        self.sample_rate = 16000
        self.ai_indicators = {
            "pitch_variance_threshold": 0.15,
            "spectral_flatness_threshold": 0.25,
            "zero_crossing_threshold": 0.08,
            "formant_stability_threshold": 0.9,
        }

    def decode_base64_audio(self, audio_base64: str) -> Tuple[np.ndarray, int]:
        try:
            audio_bytes = base64.b64decode(audio_base64)
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            if sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            return audio_data.astype(np.float32, copy=False), self.sample_rate
        except Exception as e:
            raise ValueError(f"Failed to decode audio: {str(e)}")

    def extract_audio_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        features: Dict[str, float] = {}

        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) > 0:
            features["pitch_variance"] = float(np.std(pitch_values) / (np.mean(pitch_values) + 1e-8))
        else:
            features["pitch_variance"] = 0.0

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

        if len(pitch_values) > 1:
            jitter = float(np.mean(np.abs(np.diff(pitch_values))) / (np.mean(pitch_values) + 1e-8))
            features["jitter"] = jitter
        else:
            features["jitter"] = 0.0

        return features

    def calculate_ai_probability(self, features: Dict[str, float]) -> Tuple[float, str]:
        ai_score = 0.0
        indicators = []

        if features["pitch_variance"] < self.ai_indicators["pitch_variance_threshold"]:
            ai_score += 0.25
            indicators.append("unnaturally consistent pitch")

        if features["spectral_flatness"] > self.ai_indicators["spectral_flatness_threshold"]:
            ai_score += 0.2
            indicators.append("flat spectral characteristics")

        if features["formant_stability"] > self.ai_indicators["formant_stability_threshold"]:
            ai_score += 0.25
            indicators.append("overly stable formants")

        if features["jitter"] < 0.01:
            ai_score += 0.15
            indicators.append("minimal pitch variation (jitter)")

        if features["envelope_smoothness"] < 0.05:
            ai_score += 0.15
            indicators.append("unusually smooth temporal envelope")

        if ai_score > 0.5:
            explanation = f"AI-generated indicators detected: {', '.join(indicators)}"
        else:
            human_indicators = []
            if features["pitch_variance"] >= self.ai_indicators["pitch_variance_threshold"]:
                human_indicators.append("natural pitch variation")
            if features["jitter"] >= 0.01:
                human_indicators.append("human-like voice quality")
            if features["formant_stability"] <= self.ai_indicators["formant_stability_threshold"]:
                human_indicators.append("natural formant dynamics")
            explanation = (
                f"Human voice characteristics: {', '.join(human_indicators)}"
                if human_indicators
                else "Human voice characteristics: natural speech patterns detected"
            )

        return float(ai_score), explanation

    def detect(self, audio_base64: str, language: str) -> Dict:
        try:
            audio, sr = self.decode_base64_audio(audio_base64)
            features = self.extract_audio_features(audio, sr)
            ai_probability, explanation = self.calculate_ai_probability(features)

            if ai_probability > 0.5:
                classification = "AI_GENERATED"
                confidence = ai_probability
            else:
                classification = "HUMAN"
                confidence = 1.0 - ai_probability

            confidence = float(min(max(confidence, 0.0), 1.0))
            return {
                "status": "success",
                "language": language,
                "classification": classification,
                "confidenceScore": round(confidence, 2),
                "explanation": explanation,
            }
        except Exception as e:
            return {"status": "error", "message": f"Detection failed: {str(e)}"}


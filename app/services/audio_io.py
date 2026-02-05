import io
import os
import tempfile
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class DecodedAudio:
    waveform: np.ndarray
    sample_rate: int
    duration_seconds: float


class AudioDecodeError(ValueError):
    pass


def decode_audio_bytes(audio_bytes: bytes, *, filename: str, target_sample_rate: int = 16000) -> DecodedAudio:
    try:
        audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    except Exception as e:
        suffix = os.path.splitext(filename)[1].lower() or ".bin"
        try:
            with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                audio_data, sr = librosa.load(tmp.name, sr=None, mono=False)
        except Exception as e2:
            raise AudioDecodeError(f"Failed to decode audio: {e2}") from e2

    if audio_data is None or (hasattr(audio_data, "size") and audio_data.size == 0):
        raise AudioDecodeError("Decoded audio is empty")

    if len(audio_data.shape) > 1:
        if audio_data.shape[0] <= 2:
            audio_data = np.mean(audio_data, axis=0)
        else:
            audio_data = np.mean(audio_data, axis=1)

    if sr != target_sample_rate:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sample_rate)
        sr = target_sample_rate

    duration = float(len(audio_data)) / float(sr)
    return DecodedAudio(waveform=audio_data.astype(np.float32, copy=False), sample_rate=sr, duration_seconds=duration)

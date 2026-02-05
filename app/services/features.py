from dataclasses import dataclass

import numpy as np
import torch
import torchaudio


@dataclass(frozen=True)
class FeatureVector:
    values: np.ndarray


class MelFeatureExtractor:
    def __init__(self, sample_rate: int = 16000, n_mels: int = 256, device: torch.device | None = None) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.device = device if device is not None else torch.device("cpu")

        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=320,
            f_min=20.0,
            f_max=float(sample_rate // 2),
            n_mels=n_mels,
            power=2.0,
            center=True,
        ).to(self.device)

        self._amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(self.device)

    @torch.inference_mode()
    def extract(self, waveform: np.ndarray) -> FeatureVector:
        if waveform.size == 0:
            raise ValueError("Empty waveform")

        x = torch.from_numpy(waveform).to(self.device, dtype=torch.float32)
        if x.dim() != 1:
            x = x.view(-1)

        x = x.unsqueeze(0)
        mel = self._mel(x)
        mel_db = self._amplitude_to_db(mel)

        vec = mel_db.mean(dim=-1).squeeze(0)
        values = vec.detach().to("cpu").numpy().astype(np.float32, copy=False)
        if values.shape != (self.n_mels,):
            values = values.reshape(self.n_mels)
        return FeatureVector(values=values)


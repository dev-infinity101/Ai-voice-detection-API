import os
from dataclasses import dataclass

import torch


class AiVoiceDetectorModel(torch.nn.Module):
    def __init__(self, input_dim: int = 256) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    device: torch.device


def resolve_device(device_setting: str) -> torch.device:
    if device_setting == "cpu":
        return torch.device("cpu")
    if device_setting == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str, device: torch.device) -> LoadedModel:
    model: torch.nn.Module

    if os.path.exists(model_path):
        try:
            model = torch.jit.load(model_path, map_location=device)
        except Exception:
            state = torch.load(model_path, map_location="cpu")
            model = AiVoiceDetectorModel(input_dim=256)
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                model.load_state_dict(state["state_dict"], strict=False)
            elif isinstance(state, dict):
                model.load_state_dict(state, strict=False)
    else:
        torch.manual_seed(0)
        model = AiVoiceDetectorModel(input_dim=256)

    model.to(device)
    model.eval()
    return LoadedModel(model=model, device=device)


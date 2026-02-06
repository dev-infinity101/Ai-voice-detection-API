import os
import sys
import base64

import requests


def main() -> int:
    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000")
    api_key = os.getenv("API_KEY", "SleeplessDev3")

    r = requests.get(f"{base_url}/api/v1/health", timeout=10)
    print("GET /health", r.status_code, r.text)
    r.raise_for_status()

    r = requests.get(f"{base_url}/api/v1/languages", timeout=10)
    print("GET /languages", r.status_code, r.text)
    r.raise_for_status()

    default_sample = "sample voice 1.mp3"
    target_sample = sys.argv[1] if len(sys.argv) > 1 else default_sample

    if os.path.isabs(target_sample):
        sample_path = target_sample
    else:
        sample_path = os.path.join(os.path.dirname(__file__), "..", "samples", target_sample)
    sample_path = os.path.abspath(sample_path)
    if not os.path.exists(sample_path):
        print(f"Sample not found: {sample_path}")
        return 2

    with open(sample_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        headers = {"x-api-key": api_key}
        payload = {"language": "English", "audioFormat": "mp3", "audioBase64": audio_b64}
        r = requests.post(f"{base_url}/api/voice-detection", json=payload, headers=headers, timeout=60)
        print("POST /api/voice-detection", r.status_code, r.text)
        r.raise_for_status()

    with open(sample_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        headers = {"x-api-key": api_key}
        payload = {"language": "English ", "audioFormat": " mp3", "audioBase64Format": f"data:audio/mp3;base64,{audio_b64}\n"}
        r = requests.post(f"{base_url}/api/voice-detection", json=payload, headers=headers, timeout=60)
        print("POST /api/voice-detection (noisy input)", r.status_code, r.text)
        r.raise_for_status()

    run_multipart = os.getenv("RUN_MULTIPART", "false").strip().lower() in {"1", "true", "yes", "on"}
    if run_multipart:
        with open(sample_path, "rb") as f:
            files = {"file": (os.path.basename(sample_path), f, "audio/mpeg")}
            data = {"language": "English"}
            headers = {"x-api-key": api_key}
            r = requests.post(f"{base_url}/api/v1/classify", files=files, data=data, headers=headers, timeout=180)
            print("POST /classify", r.status_code, r.text)

        with open(sample_path, "rb") as f:
            files = {"file": (os.path.basename(sample_path), f, "audio/mpeg")}
            data = {"language": "English"}
            headers = {"x-api-key": api_key}
            r = requests.post(f"{base_url}/api/v1/_debug/features", files=files, data=data, headers=headers, timeout=180)
            print("POST /_debug/features", r.status_code, r.text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

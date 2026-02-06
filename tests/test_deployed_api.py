import json
import os
import ssl
import unittest
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "https://ai-voice-detection-api-production-0bc9.up.railway.app"


def _read_sample_base64() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sample_path = os.path.join(repo_root, "samples", "sample-voice-1_base64.txt")
    with open(sample_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _request_json(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    payload: dict | None = None,
    timeout_s: float = 60.0,
) -> tuple[int, dict | None, str]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method.upper())
    req.add_header("Accept", "application/json")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=context) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            parsed = json.loads(text) if text else None
            return resp.status, parsed, text
    except urllib.error.HTTPError as e:
        raw = e.read()
        text = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text) if text else None
        except Exception:
            parsed = None
        return e.code, parsed, text


class TestDeployedApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = os.getenv("BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        cls.api_key = os.getenv("API_KEY", "SleeplessDev3")
        cls.sample_audio_b64 = _read_sample_base64()
        try:
            status, _, _ = _request_json(method="GET", url=f"{cls.base_url}/api/v1/health", timeout_s=10)
        except urllib.error.URLError as e:
            raise unittest.SkipTest(f"Network/DNS unavailable for {cls.base_url}: {e}") from e
        if status != 200:
            raise unittest.SkipTest(f"Health check failed for {cls.base_url} (status={status})")

    def test_health_endpoint(self) -> None:
        status, body, text = _request_json(method="GET", url=f"{self.base_url}/api/v1/health", timeout_s=20)
        self.assertEqual(status, 200, msg=text)
        self.assertIsInstance(body, dict, msg=text)
        self.assertEqual(body.get("status"), "healthy", msg=text)

    def test_languages_endpoint(self) -> None:
        status, body, text = _request_json(method="GET", url=f"{self.base_url}/api/v1/languages", timeout_s=20)
        self.assertEqual(status, 200, msg=text)
        self.assertIsInstance(body, dict, msg=text)
        self.assertEqual(
            set(body.get("languages", [])),
            {"Tamil", "English", "Hindi", "Malayalam", "Telugu"},
            msg=text,
        )

    def test_voice_detection_missing_api_key(self) -> None:
        payload = {"language": "English", "audioFormat": "mp3", "audioBase64": self.sample_audio_b64}
        status, body, text = _request_json(
            method="POST",
            url=f"{self.base_url}/api/voice-detection",
            payload=payload,
            timeout_s=60,
        )
        self.assertEqual(status, 401, msg=text)
        self.assertIsInstance(body, dict, msg=text)
        self.assertEqual(body.get("status"), "error", msg=text)

    def test_voice_detection_invalid_api_key(self) -> None:
        payload = {"language": "English", "audioFormat": "mp3", "audioBase64": self.sample_audio_b64}
        status, body, text = _request_json(
            method="POST",
            url=f"{self.base_url}/api/voice-detection",
            headers={"x-api-key": "invalid"},
            payload=payload,
            timeout_s=60,
        )
        self.assertEqual(status, 403, msg=text)
        self.assertIsInstance(body, dict, msg=text)
        self.assertEqual(body.get("status"), "error", msg=text)

    def test_voice_detection_invalid_body(self) -> None:
        status, body, text = _request_json(
            method="POST",
            url=f"{self.base_url}/api/voice-detection",
            headers={"x-api-key": self.api_key},
            payload={"language": "English"},
            timeout_s=20,
        )
        self.assertEqual(status, 422, msg=text)
        self.assertIsInstance(body, dict, msg=text)
        self.assertEqual(body.get("status"), "error", msg=text)

    def test_voice_detection_unsupported_language(self) -> None:
        payload = {"language": "French", "audioFormat": "mp3", "audioBase64": self.sample_audio_b64}
        status, body, text = _request_json(
            method="POST",
            url=f"{self.base_url}/api/voice-detection",
            headers={"x-api-key": self.api_key},
            payload=payload,
            timeout_s=20,
        )
        self.assertEqual(status, 422, msg=text)
        self.assertIsInstance(body, dict, msg=text)
        self.assertEqual(body.get("status"), "error", msg=text)

    def test_voice_detection_success(self) -> None:
        payload = {"language": "English", "audioFormat": "mp3", "audioBase64": self.sample_audio_b64}
        status, body, text = _request_json(
            method="POST",
            url=f"{self.base_url}/api/voice-detection",
            headers={"x-api-key": self.api_key},
            payload=payload,
            timeout_s=90,
        )
        self.assertEqual(status, 200, msg=text)
        self.assertIsInstance(body, dict, msg=text)
        self.assertEqual(body.get("status"), "success", msg=text)
        self.assertEqual(body.get("language"), "English", msg=text)
        self.assertIn(body.get("classification"), {"AI_GENERATED", "HUMAN"}, msg=text)
        self.assertIsInstance(body.get("confidenceScore"), (int, float), msg=text)
        self.assertGreaterEqual(float(body["confidenceScore"]), 0.0, msg=text)
        self.assertLessEqual(float(body["confidenceScore"]), 1.0, msg=text)
        self.assertIsInstance(body.get("explanation"), str, msg=text)
        self.assertTrue(body["explanation"].strip(), msg=text)

    def test_voice_detection_noisy_base64_inputs(self) -> None:
        payload = {
            "language": "English ",
            "audioFormat": " mp3",
            "audioBase64Format": f"data:audio/mp3;base64,{self.sample_audio_b64}\n",
        }
        status, body, text = _request_json(
            method="POST",
            url=f"{self.base_url}/api/voice-detection",
            headers={"x-api-key": self.api_key},
            payload=payload,
            timeout_s=90,
        )
        self.assertEqual(status, 200, msg=text)
        self.assertIsInstance(body, dict, msg=text)
        self.assertEqual(body.get("status"), "success", msg=text)


if __name__ == "__main__":
    unittest.main()

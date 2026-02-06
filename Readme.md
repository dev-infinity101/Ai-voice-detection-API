# AI Voice Detection API 🎙️

A production-ready FastAPI service that classifies speech audio as either **AI-generated** (synthetic) or **Human** speech using advanced heuristic feature analysis. This system supports multiple Indian languages and English, providing real-time detection with detailed explanations.

---

## 1. Project Overview & Key Features

The AI Voice Detection API is designed to combat deepfakes and synthetic audio by analyzing acoustic properties that differ between biological speech production and algorithmic synthesis.

- **Dual-Mode Classification**: Distinguishes between `AI_GENERATED` and `HUMAN` speech.
- **Multilingual Support**: Optimized for Tamil, English, Hindi, Malayalam, and Telugu.
- **Heuristic Engine**: Uses pitch variance, spectral flatness, formant stability, and temporal envelope analysis.
- **Flexible Integration**: Supports both JSON/Base64 and Multipart/Form-data uploads.
- **Developer First**: Comprehensive Swagger/OpenAPI documentation and detailed error reporting.
- **Reliable Performance**: Built-in rate limiting and asynchronous processing.

---

## 2. Installation & Setup

### Prerequisites
- Python 3.10+
- FFmpeg (required for audio decoding via librosa/soundfile)

### Local Installation

1. **Clone and Navigate**:
   ```bash
   git clone <repository-url>
   cd Ai-voice-detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```env
   API_KEY=SleeplessDev3
   PORT=8000
   ENVIRONMENT=dev
   LOG_LEVEL=INFO
   ENABLE_DEBUG_ROUTES=false
   RATE_LIMIT_RPM=60
   MAX_UPLOAD_MB=25
   ```

4. **Start the Server**:
   ```bash
   python main.py
   ```

---

## 3. API Reference

### Base URL
- Local: `http://localhost:8000`
- Production: `https://ai-voice-detection-api-production-0bc9.up.railway.app`

### Endpoints

#### `POST /api/voice-detection`
The primary endpoint for hackathon testers and simple integrations.

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `language` | string | Yes | One of: `Tamil`, `English`, `Hindi`, `Malayalam`, `Telugu` |
| `audioFormat` | string | Yes | `mp3`, `wav`, or `flac` |
| `audioBase64` | string | Yes | Base64 encoded audio string (raw or data-URI) |

**Response (200 OK)**:
```json
{
  "status": "success",
  "language": "English",
  "classification": "HUMAN",
  "confidenceScore": 0.85,
  "explanation": "Human voice characteristics: natural pitch variation, natural formant dynamics"
}
```

#### `POST /api/v1/classify`
Advanced endpoint supporting direct file uploads.

- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **Fields**:
    - `file`: Binary audio file (WAV/MP3/FLAC)
    - `language`: Target language

**Response (200 OK)**:
```json
{
  "status": "success",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.92,
  "languageDetected": "English",
  "probabilities": { "human": 0.08, "ai": 0.92 },
  "audioDurationSeconds": 4.5,
  "processingMs": 120.5,
  "explanation": "AI-generated indicators detected: unnaturally consistent pitch, overly stable formants"
}
```

---

## 4. Code Examples

### Python (Asynchronous with `httpx`)
```python
import httpx
import asyncio
import base64

async def detect_voice(file_path: str):
    url = "https://ai-voice-detection-api-production-0bc9.up.railway.app/api/voice-detection"
    headers = {"x-api-key": "SleeplessDev3"}
    
    with open(file_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_b64
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=30.0)
            resp.raise_for_status()
            print(resp.json())
        except httpx.HTTPStatusError as e:
            print(f"Error {e.response.status_code}: {e.response.text}")

asyncio.run(detect_voice("sample.mp3"))
```

### JavaScript (Fetch API)
```javascript
const fs = require('fs');

async function detectVoice(filePath) {
  const audioBase64 = fs.readFileSync(filePath, { encoding: 'base64' });
  
  const response = await fetch('https://ai-voice-detection-api-production-0bc9.up.railway.app/api/voice-detection', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': 'SleeplessDev3'
    },
    body: JSON.stringify({
      language: 'English',
      audioFormat: 'mp3',
      audioBase64: audioBase64
    })
  });

  const data = await response.json();
  console.log(data);
}
```

### cURL (JSON/Base64)
```bash
curl -X POST "https://ai-voice-detection-api-production-0bc9.up.railway.app/api/voice-detection" \
  -H "Content-Type: application/json" \
  -H "x-api-key: SleeplessDev3" \
  -d "{\"language\":\"English\",\"audioFormat\":\"mp3\",\"audioBase64\":\"$(base64 -w 0 audio.mp3)\"}"
```

---

## 5. How It Works: The Detection Workflow

1.  **Ingestion**: The API receives audio via Base64 or Multipart upload.
2.  **Normalization**: Waveforms are resampled to 16kHz and peak-normalized.
3.  **Language-Specific Preprocessing**: Silence is trimmed using dynamic thresholds tailored for the phonetic characteristics of the selected language (e.g., different `top_db` for Tamil vs English).
4.  **Feature Extraction**:
    - **Pitch Variance**: Analyzes F0 stability (AI voices often lack natural jitter).
    - **Spectral Flatness**: Detects robotic "tonal" signatures vs. natural noise floors.
    - **Formant Stability**: Measures the consistency of vocal tract resonance.
    - **Temporal Envelope**: Examines the smoothness of the signal amplitude.
5.  **Heuristic Scoring**: Features are weighted to produce a probability score. If the score exceeds 0.5, it is classified as `AI_GENERATED`.

---

## 6. Configuration & Authentication

### Authentication
All requests must include the `x-api-key` header.
```http
x-api-key: YOUR_SECRET_KEY
```

### Settings
- `MAX_UPLOAD_MB`: Default 25MB.
- `ENABLE_DEBUG_ROUTES`: Enables `/api/v1/_debug/features` for deep analysis.
- `SUPPORTED_LANGUAGES`: `Tamil`, `English`, `Hindi`, `Malayalam`, `Telugu`.

---

## 7. Rate Limiting & Guidelines

- **Limit**: 60 requests per minute per IP (configurable).
- **Audio Duration**: 0.5s minimum, 60s maximum.
- **Recommended Quality**: 128kbps+ MP3 or 16-bit WAV for optimal feature extraction.
- **Noise**: Excessive background noise may reduce classification confidence.

---

## 8. Error Handling

| Status Code | Message | Solution |
| :--- | :--- | :--- |
| `400` | "Invalid base64 audio" | Check Base64 encoding and padding. |
| `401` | "Missing API key" | Include `x-api-key` header. |
| `403` | "Invalid API key" | Verify your key matches the server config. |
| `413` | "File too large" | Reduce file size below `MAX_UPLOAD_MB`. |
| `422` | "Invalid request body" | Ensure all required fields are present and valid. |
| `429` | "Rate limit exceeded" | Wait before sending more requests. |

---

## 9. Testing

### Sample Audio
Samples are located in the `/samples` directory. Use these to verify your integration.

### Automated Tests
```bash
# Test the deployed production API
python -m unittest tests/test_deployed_api.py

# Test a specific local instance
set BASE_URL=http://localhost:8000
python -m unittest tests/test_deployed_api.py
```

---

## 10. Contributing & License

### Contributing
1. Fork the repository.
2. Create a feature branch.
3. Submit a Pull Request with tests for any new features.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Created for GUVI x HCL Hackathon 2026**

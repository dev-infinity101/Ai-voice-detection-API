# AI Voice Detection API 🎙️

A REST API that detects whether a voice sample is AI-generated or spoken by a real human, supporting Tamil, English, Hindi, Malayalam, and Telugu languages.

## 🚀 Features

- ✅ Detects AI-generated vs Human voices
- ✅ Supports 5 languages: Tamil, English, Hindi, Malayalam, Telugu
- ✅ Base64 MP3 audio input
- ✅ API key authentication
- ✅ JSON response with confidence scores
- ✅ Fast and reliable detection

## 📋 Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## 🛠️ Local Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd voice-detection-api
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file:
```env
API_KEY=your_secret_api_key_here
PORT=8000
```

### 4. Run the application
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📡 API Usage

### Authentication
All requests require an API key in the header:
```
x-api-key: your_secret_api_key
```

### Endpoint
```
POST /api/voice-detection
```

### Request Format
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
}
```

### Response Format (Success)
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

### Response Format (Error)
```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

### Supported Languages
- Tamil
- English
- Hindi
- Malayalam
- Telugu

### Classification Types
- `AI_GENERATED` - Voice created using AI or synthetic systems
- `HUMAN` - Voice spoken by a real human

## 🧪 Testing

Run the test script:
```bash
python test_api.py
```

Test with a real audio file:
```python
from test_api import test_with_real_audio

test_with_real_audio("path/to/audio.mp3", "English")
```

## 🐳 Docker Deployment

Build the Docker image:
```bash
docker build -t voice-detection-api .
```

Run the container:
```bash
docker run -p 8000:8000 -e API_KEY=your_secret_key voice-detection-api
```

## ☁️ Cloud Deployment

### Railway
1. Connect your GitHub repository to Railway
2. Set environment variable: `API_KEY=your_secret_key`
3. Deploy automatically

### Render
1. Create a new Web Service
2. Connect your repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python main.py`
5. Add environment variable: `API_KEY`

### Heroku
```bash
heroku create your-app-name
heroku config:set API_KEY=your_secret_key
git push heroku main
```

## 🔧 How It Works

The detection system uses multiple audio feature analysis techniques:

1. **Pitch Variance Analysis**: AI voices often have unnaturally consistent pitch
2. **Spectral Flatness**: Measures the tonality vs. noise-like characteristics
3. **Zero Crossing Rate**: Analyzes the rate of signal sign changes
4. **Formant Stability**: AI voices have overly stable formants
5. **Jitter & Shimmer**: Voice quality measures that differ between AI and human
6. **Temporal Envelope**: Smoothness patterns in the audio signal
7. **MFCC Statistics**: Mel-frequency cepstral coefficients analysis

These features are combined to calculate a confidence score and determine if the voice is AI-generated or human.

## 📊 Performance Tips

1. **Audio Quality**: Higher quality MP3 files (128kbps+) give better results
2. **Audio Length**: 3-10 seconds of audio is optimal
3. **Background Noise**: Clean audio improves accuracy
4. **Language Specificity**: While the core algorithm works across languages, accuracy may vary

## 🎯 Evaluation Criteria

The system is evaluated on:
- ✅ Accuracy of AI vs Human detection
- ✅ Consistency across all 5 languages
- ✅ Correct request & response format
- ✅ API reliability and response time
- ✅ Quality of explanation

## 🔒 Security

- API key authentication required for all requests
- Input validation for all parameters
- Error handling without exposing internal details
- Rate limiting recommended for production

## 📝 Example cURL Request

```bash
curl -X POST https://your-domain.com/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
  }'
```

## 🐛 Troubleshooting

### Issue: "Failed to decode audio"
- Ensure the audio is valid MP3 format
- Check that base64 encoding is correct
- Verify the audio isn't corrupted

### Issue: "Invalid API key"
- Check the `x-api-key` header is included
- Verify the API key matches the server configuration

### Issue: "Language must be one of..."
- Ensure language is exactly one of: Tamil, English, Hindi, Malayalam, Telugu
- Check for typos and case sensitivity

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Librosa Audio Analysis](https://librosa.org/)
- [Audio Feature Extraction Guide](https://en.wikipedia.org/wiki/Audio_signal_processing)

## 🤝 Contributing

This is a hackathon project. Feel free to fork and improve!

## 📄 License

MIT License

## 👥 Team

Created for GUVI x HCL Hackathon 2026

---

**Note**: This system uses feature-based detection. For production use, consider fine-tuning with labeled datasets of AI and human voices in all supported languages.
# 🚀 QUICK START GUIDE - 1 DAY HACKATHON

## ⏰ Time Breakdown (8 hours total)

1. **Setup (30 mins)** - Install dependencies, set up environment
2. **Testing (1 hour)** - Test locally with sample audio
3. **Tuning (2 hours)** - Adjust thresholds based on sample data
4. **Deployment (1 hour)** - Deploy to cloud platform
5. **Documentation (30 mins)** - Finalize README and API docs
6. **Final Testing (1 hour)** - End-to-end testing with deployed API
7. **Buffer (2 hours)** - Handle issues, improvements, presentation prep

---

## 🎯 STEP-BY-STEP EXECUTION

### STEP 1: Initial Setup (30 mins)

```bash
# Create project directory
mkdir voice-detection-hackathon
cd voice-detection-hackathon

# Copy all provided files into this directory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### STEP 2: Get Sample Audio (Provided by Organizers)

The hackathon will provide sample audio files. Save them in a `samples/` folder:
```
samples/
  ├── ai_tamil.mp3
  ├── human_tamil.mp3
  ├── ai_english.mp3
  ├── human_english.mp3
  └── ...
```

### STEP 3: Test Locally (30 mins)

```bash
# Start the server
python main.py

# In another terminal, run tests
python test_api.py

# Test with real audio
python -c "from test_api import test_with_real_audio; test_with_real_audio('samples/ai_english.mp3', 'English')"
```

### STEP 4: Tune Detection Thresholds (2 hours)

This is THE MOST IMPORTANT STEP for winning!

1. **Test with provided samples**:
   ```python
   # Create a tuning script
   python tune_detector.py
   ```

2. **Analyze results and adjust thresholds in `voice_detector.py`**:
   ```python
   self.ai_indicators = {
       'pitch_variance_threshold': 0.15,  # ADJUST THIS
       'spectral_flatness_threshold': 0.25,  # ADJUST THIS
       'zero_crossing_threshold': 0.08,  # ADJUST THIS
       'formant_stability_threshold': 0.9  # ADJUST THIS
   }
   ```

3. **Iterate**: Test → Adjust → Test → Adjust

### STEP 5: Deploy to Cloud (1 hour)

#### Option A: Railway (Recommended - Fastest)

1. Create account at railway.app
2. Create new project
3. Connect GitHub repo OR deploy from CLI:
   ```bash
   npm i -g @railway/cli
   railway login
   railway init
   railway up
   ```
4. Set environment variable: `API_KEY=sk_test_123456789`
5. Get your deployment URL

#### Option B: Render

1. Go to render.com
2. New → Web Service
3. Connect repo
4. Settings:
   - Build: `pip install -r requirements.txt`
   - Start: `python main.py`
   - Environment: Add `API_KEY`

#### Option C: Hugging Face Spaces

1. Create Space at huggingface.co
2. Upload files
3. Add `app.py` that imports from `main.py`

### STEP 6: Final Testing (1 hour)

```bash
# Test deployed API
curl -X POST https://your-app.railway.app/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d @test_request.json

# Test all languages
# Test error cases
# Verify response format
```

### STEP 7: Submit (30 mins)

1. ✅ Verify API endpoint is public
2. ✅ Test API key authentication
3. ✅ Confirm all 5 languages work
4. ✅ Check response format matches requirements
5. ✅ Submit API endpoint and key to hackathon portal

---

## 🎯 WINNING STRATEGIES

### 1. **Focus on Accuracy with Sample Data**
- The organizers will test with specific audio samples
- Spend most time tuning thresholds on these samples
- Don't over-engineer; simple features work well

### 2. **Reliable API is Better than Complex ML**
- A working 80% accurate API beats a broken 95% model
- Feature-based detection is fast and reliable
- ML models can fail in production

### 3. **Handle Edge Cases**
- Very short audio (< 1 second)
- Very long audio (> 30 seconds)
- Poor quality audio
- Background noise

### 4. **Perfect the Response Format**
- Match the exact JSON format required
- Include meaningful explanations
- Confidence scores between 0.0 and 1.0

### 5. **Fast Response Time**
- Optimize audio processing
- Keep API response under 2 seconds
- Use async processing if needed

---

## 🔧 TUNING SCRIPT

Create `tune_detector.py`:

```python
import os
from voice_detector import VoiceDetector
import base64

detector = VoiceDetector()

# Test all samples
samples = {
    'ai': [
        'samples/ai_tamil.mp3',
        'samples/ai_english.mp3',
        # ... add all AI samples
    ],
    'human': [
        'samples/human_tamil.mp3',
        'samples/human_english.mp3',
        # ... add all human samples
    ]
}

print("Testing AI samples...")
for file in samples['ai']:
    with open(file, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode()
        result = detector.detect(audio_b64, "English")
        print(f"{file}: {result['classification']} ({result['confidenceScore']})")

print("\nTesting Human samples...")
for file in samples['human']:
    with open(file, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode()
        result = detector.detect(audio_b64, "English")
        print(f"{file}: {result['classification']} ({result['confidenceScore']})")
```

---

## ⚠️ COMMON PITFALLS TO AVOID

1. ❌ **Don't hardcode results** - You'll be disqualified
2. ❌ **Don't ignore sample data** - Tune on what you'll be tested with
3. ❌ **Don't over-complicate** - Simple working solution > complex broken one
4. ❌ **Don't forget error handling** - Test edge cases
5. ❌ **Don't skip documentation** - Clear README helps judges

---

## ✅ PRE-SUBMISSION CHECKLIST

- [ ] API is publicly accessible
- [ ] API key authentication works
- [ ] All 5 languages tested
- [ ] Response format matches exactly
- [ ] Error handling works
- [ ] Response time < 3 seconds
- [ ] README is clear
- [ ] API endpoint submitted
- [ ] API key shared with organizers
- [ ] Tested with sample data

---

## 🏆 FINAL TIPS

1. **Start simple, iterate fast**
2. **Test early, test often**
3. **Document everything**
4. **Keep calm, stay focused**
5. **Have fun! 🎉**

Good luck with the hackathon! 🚀
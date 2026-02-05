# Railway Deployment Guide (Ai-voice-detection)

## 1) Prerequisites

- GitHub repo: https://github.com/dev-infinity101/Ai-voice-detection-API
- Railway account: https://railway.app

## 2) Create Railway Project (GitHub Deploy)

1. Railway → **New Project**
2. Choose **Deploy from GitHub repo**
3. Select `dev-infinity101/Ai-voice-detection-API`

## 3) Configure Build + Start

Railway Settings → **Deploy**:

- **Build Command**
  - `pip install -r requirements.txt`
- **Start Command**
  - `python -m uvicorn main:app --host 0.0.0.0 --port $PORT`

## 4) Set Environment Variables

Railway Settings → **Variables**:

- `API_KEY` = `SleeplessDev3`
- `ENVIRONMENT` = `prod`
- `LOG_LEVEL` = `INFO`
- `ENABLE_DEBUG_ROUTES` = `false`

## 5) Deploy

1. Trigger deploy (Railway auto-deploys on pushes to `main`)
2. Open the Railway **Public URL**

## 6) Verify Health

Open these in your browser:

- `https://<your-railway-domain>/api/v1/health`
- `https://<your-railway-domain>/api/v1/languages`
- `https://<your-railway-domain>/docs`

## 7) Hackathon Tester Endpoint (Base64 JSON)

Your hackathon tester UI expects a JSON request with base64 audio. This project supports:

- `POST https://<your-railway-domain>/api/voice-detection`
- Header `x-api-key: SleeplessDev3`

## 8) Common Fixes (If Railway Crashes)

- Ensure Start Command is **uvicorn** with `$PORT` (not `python main.py`)
- Ensure `API_KEY` is set in Railway Variables
- If you see 404 on debug endpoints, set `ENABLE_DEBUG_ROUTES=true` (only for troubleshooting)

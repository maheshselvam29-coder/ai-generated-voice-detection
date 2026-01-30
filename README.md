# AI or Human Voice Detector

An end-to-end app (FastAPI backend + minimal frontend) to detect whether a voice sample is AI-generated or Human-generated.

Features
- POST /detect-voice: accepts base64-encoded MP3 or WAV audio
- Feature extraction: MFCCs, pitch (F0), spectral features (centroid, bandwidth, rolloff, flatness), chroma, zero-crossing rate
- Model: scikit-learn RandomForest pipeline, with heuristic fallback if no trained model found
- Language detection (optional): uses Whisper if installed to transcribe and detect language (Tamil, English, Hindi, Malayalam, Telugu)
- Explanation: feature importances or heuristic explanation returned along with classification and confidence

Quick start (local)
1. Install dependencies (recommended: virtualenv):
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. (Optional) Train a model:
   - Prepare a CSV dataset `dataset.csv` with columns: `filepath,label` where label is "AI" or "Human"
   - Run:
     python scripts/train_model.py --csv dataset.csv --out models/voice_detector.pkl

   This will create `models/voice_detector.pkl` and `models/scaler.pkl`.

3. Run the server:
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

4. Open UI:
   Visit http://localhost:8000/static/index.html for a simple UI to upload and test audio.

API
- POST /detect-voice
  Request JSON:
  {
    "audio_base64": "<base64 string>",
    "filename": "sample.wav",   # optional (helps librosa infer format)
    "language_hint": "Tamil"    # optional
  }

  Response JSON:
  {
    "classification": "AI" | "Human",
    "confidence": 0.0-1.0,
    "language": "English" | "Tamil" | "Hindi" | "Malayalam" | "Telugu" | "unknown",
    "explanation": "human readable explanation with top features"
  }

Notes
- For better language detection/transcription, install `openai-whisper` and `torch` (CPU or GPU builds). If Whisper is not installed, the response will include `"language": "unknown"` unless the user provides a language_hint.
- The baseline model is a RandomForest classifier trained on extracted features; real performance depends on training data quality (AI samples vs real human recordings across languages and recording conditions).
- Use the training script to create your own labeled dataset and produce a saved model.

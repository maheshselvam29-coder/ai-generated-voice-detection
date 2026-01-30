import uvicorn
from fastapi import FastAPI, HTTPException
from app.schemas import DetectRequest, DetectResponse
from app.utils.audio import decode_base64_to_tempfile, load_audio, extract_features
from app.model.model import get_model_and_explainer, explain_prediction
import numpy as np
import os
import shutil

# Optional whisper import (for language detection/transcription)
try:
    import whisper
    whisper_model = whisper.load_model("small")
except Exception:
    whisper = None
    whisper_model = None

app = FastAPI(title="AI or Human Voice Detector")

@app.post("/detect-voice", response_model=DetectResponse)
async def detect_voice(req: DetectRequest):
    # 1. Decode file
    try:
        tmp_path = decode_base64_to_tempfile(req.audio_base64, req.filename or "sample.wav")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 audio: {str(e)}")

    try:
        y, sr = load_audio(tmp_path, sr=22050)
    except Exception as e:
        # cleanup
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Failed to load audio: {str(e)}")

    # 2. Extract features
    try:
        feature_vector, feature_dict = extract_features(y, sr)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    # 3. Language detection (try whisper if available)
    detected_language = "unknown"
    if req.language_hint:
        detected_language = req.language_hint
    else:
        if whisper_model is not None:
            try:
                # whisper expects a path or numpy array (float32)
                # save to temporary wav if needed (we already have path)
                result = whisper_model.transcribe(tmp_path, language=None, fp16=False)  # let it auto-detect
                # result contains language code and text
                lang_code = result.get("language", None)
                if lang_code:
                    # map to friendly name if needed (limited set)
                    code_map = {
                        "en": "English",
                        "ta": "Tamil",
                        "hi": "Hindi",
                        "ml": "Malayalam",
                        "te": "Telugu"
                    }
                    detected_language = code_map.get(lang_code, lang_code)
                else:
                    detected_language = "unknown"
            except Exception:
                detected_language = "unknown"
        else:
            detected_language = "unknown"

    # 4. Load model (or heuristic) and predict
    from app.model.model import load_trained_model
    model, scaler, model_type = None, None, None
    try:
        model, scaler, model_type = get_model_and_explainer()
    except Exception:
        model, scaler, model_type = get_model_and_explainer()

    X = feature_vector.reshape(1, -1)
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except Exception:
            X_scaled = X
    else:
        X_scaled = X

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            # Assuming classes are ordered as [Human, AI] for trained model; if model has classes_ attribute, align.
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                # find index for AI and Human
                if "AI" in classes and "Human" in classes:
                    ai_idx = classes.index("AI")
                    human_idx = classes.index("Human")
                    ai_conf = float(proba[ai_idx])
                    human_conf = float(proba[human_idx])
                else:
                    # fallback: assume second is AI
                    ai_conf = float(proba[-1])
                    human_conf = float(proba[0])
            else:
                # heuristic model returns [human_prob, ai_prob]
                human_conf = float(proba[0])
                ai_conf = float(proba[1])
        else:
            # model may be custom
            pred = model.predict(X_scaled)
            ai_conf = 0.9 if pred[0] == "AI" else 0.1
            human_conf = 1.0 - ai_conf
    except Exception:
        # fallback to heuristic probabilities from heuristic model if something fails
        try:
            probs = model.predict_proba(X_scaled)[0]
            human_conf = float(probs[0])
            ai_conf = float(probs[1])
        except Exception:
            human_conf = 0.5
            ai_conf = 0.5

    classification = "AI" if ai_conf >= human_conf else "Human"
    confidence = float(ai_conf if classification == "AI" else human_conf)

    # explanation
    explanation = explain_prediction(model_type if model_type else "heuristic", model, X_scaled.flatten(), feature_dict)

    # cleanup
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return DetectResponse(
        classification=classification,
        confidence=round(confidence, 4),
        language=detected_language,
        explanation=explanation
    )

# Serve static files (minimal UI)
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
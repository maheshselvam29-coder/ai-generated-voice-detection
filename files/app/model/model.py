import os
import joblib
import numpy as np
from typing import Tuple, Dict
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = os.path.join("models", "voice_detector.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

class HeuristicModel:
    """
    Simple heuristic model used as a fallback when no trained model is available.
    Uses spectral_flatness and pitch variance heuristics.
    """
    def predict_proba(self, X):
        # X: (n_samples, n_features) numpy
        # We expect spectral_flatness at a certain index; find it heuristically:
        # For our feature vector layout in utils.extract_features, spectral_flatness is near the end.
        # We'll assume spectral_flatness is the value that is small (~0-1) and present in features.
        sf_idx = None
        # find index with values mostly between 0 and 1
        sample = X[0]
        for i, v in enumerate(sample):
            if 0.0 <= v <= 1.0:
                sf_idx = i
                break
        if sf_idx is None:
            # fallback index
            sf_idx = -10
        sf_vals = X[:, sf_idx]
        # pitch variance near end: choose last column
        pv_vals = X[:, -1]
        probs = []
        for sf, pv in zip(sf_vals, pv_vals):
            # heuristic: very flat spectra (higher spectral_flatness) and very low pitch variance => AI
            score = 0.5 * (sf) + 0.5 * (1.0 - np.tanh(pv))  # scaled 0..1
            # clamp
            score = float(max(0.0, min(1.0, score)))
            probs.append([1.0 - score, score])  # [human_prob, ai_prob]
        return np.array(probs)

    def predict(self, X):
        proba = self.predict_proba(X)
        labels = ["Human" if p[0] >= p[1] else "AI" for p in proba]
        return np.array(labels)

def load_trained_model() -> Tuple[object, object]:
    """
    Returns (model, scaler) if found, else (None, None).
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None

def get_model_and_explainer():
    """
    Loads trained model if available, else returns heuristic model.
    """
    model, scaler = load_trained_model()
    if model is not None:
        return model, scaler, "trained"
    else:
        return HeuristicModel(), None, "heuristic"

def explain_prediction(model_type: str, model, feature_vector: np.ndarray, feature_dict: Dict[str, float]) -> str:
    if model_type == "trained":
        # If RandomForest or similar, use feature_importances_
        try:
            importances = model.feature_importances_
            # map importances to feature indices; build top-5 explanation
            inds = np.argsort(importances)[-5:][::-1]
            top = []
            for i in inds:
                top.append(f"feature_{i} (importance={importances[i]:.3f}, value={float(feature_vector[i]):.3f})")
            return "Top influencing features: " + "; ".join(top)
        except Exception:
            return "Model prediction (trained). Feature importance not available."
    else:
        # heuristic
        sf = feature_dict.get("spectral_flatness", None)
        pv = feature_dict.get("pitch_var", None)
        reasons = []
        if sf is not None:
            reasons.append(f"spectral_flatness={sf:.3f}")
        if pv is not None:
            reasons.append(f"pitch_variance={pv:.3f}")
        return "Heuristic decision using: " + ", ".join(reasons)
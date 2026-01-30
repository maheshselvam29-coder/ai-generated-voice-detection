import base64
import tempfile
import os
from typing import Tuple, Dict
import numpy as np
import librosa

def decode_base64_to_tempfile(b64: str, filename_hint: str = "sample.wav") -> str:
    header_cut = b64.find("base64,")
    if header_cut != -1:
        b64 = b64[header_cut + len("base64,"):]
    data = base64.b64decode(b64)
    suffix = os.path.splitext(filename_hint)[1] if "." in filename_hint else ".wav"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(data)
    tf.flush()
    tf.close()
    return tf.name

def load_audio(path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    # librosa load handles many formats if soundfile or audioread available
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def extract_features(y: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extract a compact feature vector for classification and a dict of named features for explanation.
    Returns: feature_vector (1D numpy), feature_dict
    """
    # Ensure audio length > small epsilon
    if y.size == 0:
        raise ValueError("Empty audio")

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # Spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    flatness = librosa.feature.spectral_flatness(y=y).mean()

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    # Pitch (fundamental frequency) - using YIN for f0 estimate
    try:
        f0 = librosa.yin(y, fmin=50, fmax=sr//2, sr=sr)
        # f0 may contain NaNs, ignore them
        f0_clean = f0[~np.isnan(f0)]
        if f0_clean.size > 0:
            f0_mean = float(np.mean(f0_clean))
            f0_var = float(np.var(f0_clean))
        else:
            f0_mean = 0.0
            f0_var = 0.0
    except Exception:
        # fallback if yin not available
        f0_mean = 0.0
        f0_var = 0.0

    # Aggregate into vector
    feat_list = []
    feat_list.extend(mfcc_mean.tolist())
    feat_list.extend(mfcc_std.tolist())
    feat_list.append(spec_centroid)
    feat_list.append(spec_bandwidth)
    feat_list.append(spec_rolloff)
    feat_list.append(zcr)
    feat_list.append(flatness)
    feat_list.extend(chroma_mean.tolist())
    feat_list.append(f0_mean)
    feat_list.append(f0_var)

    feature_vector = np.array(feat_list, dtype=np.float32)

    # feature dict for explanation
    feature_dict = {
        "mfcc_mean_0": float(mfcc_mean[0]),
        "mfcc_mean_1": float(mfcc_mean[1]) if mfcc_mean.size > 1 else 0.0,
        "spectral_centroid": float(spec_centroid),
        "spectral_bandwidth": float(spec_bandwidth),
        "spectral_rolloff": float(spec_rolloff),
        "zero_crossing_rate": float(zcr),
        "spectral_flatness": float(flatness),
        "pitch_mean": float(f0_mean),
        "pitch_var": float(f0_var),
    }

    return feature_vector, feature_dict
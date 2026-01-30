"""
Train a RandomForest classifier on a CSV dataset.

CSV format:
filepath,label
/path/to/audio1.wav,Human
/path/to/audio2.wav,AI
...

Produces:
- models/voice_detector.pkl  (trained sklearn pipeline)
- models/scaler.pkl
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from app.utils.audio import load_audio, extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def build_dataset(csv_path):
    df = pd.read_csv(csv_path)
    X = []
    y = []
    for idx, row in df.iterrows():
        fp = row['filepath']
        label = row['label']
        if not os.path.exists(fp):
            print(f"Warning: file {fp} not found, skipping.")
            continue
        try:
            y_audio, sr = load_audio(fp, sr=22050)
            feat_vec, feat_dict = extract_features(y_audio, sr)
            X.append(feat_vec)
            y.append(label)
        except Exception as e:
            print(f"Failed to process {fp}: {e}")
    return np.vstack(X), np.array(y)

def train(csv_path, out_dir):
    X, y = build_dataset(csv_path)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    clf.fit(Xs, y)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(out_dir, "voice_detector.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    print("Saved model and scaler to", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with columns filepath,label")
    p.add_argument("--out", default="models", help="Output directory for model files")
    args = p.parse_args()
    train(args.csv, args.out)
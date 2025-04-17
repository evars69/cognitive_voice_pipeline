# src/predict.py
import librosa
import numpy as np
import joblib
import sys
import os

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    mfcc_mean = np.mean(mfcc.T, axis=0)
    chroma_mean = np.mean(chroma.T, axis=0)

    print(f"MFCC shape: {mfcc_mean.shape}")    # Debug
    print(f"Chroma shape: {chroma_mean.shape}")  # Debug

    features = np.hstack([mfcc_mean, chroma_mean])
    print(f"Combined feature shape before padding: {features.shape}")  # Debug

    # Ensure feature vector has exactly 26 values
    if len(features) < 26:
        features = np.pad(features, (0, 26 - len(features)))
    elif len(features) > 26:
        features = features[:26]

    print(f"Final feature shape: {features.shape}")  # Debug

    return features.reshape(1, -1)  # Reshape to (1, 26) for prediction


def predict(audio_path):
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return

    features = extract_features(audio_path).reshape(1, -1)

    model = joblib.load("model/voice_model.pkl")
    prediction = model.predict(features)
    print(f"Predicted cognitive state: {prediction[0]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_audio_file>")
    else:
        predict(sys.argv[1])

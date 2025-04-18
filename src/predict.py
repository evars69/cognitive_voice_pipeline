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

    # Combine features
    features = np.hstack([mfcc_mean, chroma_mean])

    # Ensuring feature vector has exactly 26 values
    if len(features) < 26:
        features = np.pad(features, (0, 26 - len(features)))
    elif len(features) > 26:
        features = features[:26]

    return features.reshape(1, -1)  # Reshape to (1, 26) for prediction


def predict(audio_path):
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return

    features = extract_features(audio_path).reshape(1, -1)

    # Ensure model file exists
    model_path = "model/voice_model.pkl"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    model = joblib.load(model_path)
    prediction = model.predict(features)
    print(f"Predicted cognitive state: {prediction[0]}")

    return prediction[0]  # Return the predicted result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <data/Recording.m4a>")
    else:
        predict(sys.argv[1])

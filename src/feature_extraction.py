import os
import librosa
import numpy as np

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    
    feature_vector = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(zcr)
    ])
    
    return feature_vector

# Directory with all audio files
data_dir = "data"

features = []
labels = []

for filename in os.listdir(data_dir):
    if filename.endswith(".wav") or filename.endswith(".m4a"):
        file_path = os.path.join(data_dir, filename)
        print(f"Processing {file_path}")
        feature_vector = extract_features(file_path)
        features.append(feature_vector)

        # Manually assign label here
        label = input(f"Enter label for {filename}: ")
        labels.append(label)

features = np.array(features)
labels = np.array(labels)

# Save extracted features and labels
np.save("data/features.npy", features)
np.save("data/labels.npy", labels)

print("Feature extraction complete.")

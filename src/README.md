# 🧠 MemoTag | Voice-Based Cognitive Decline Detection

Welcome to  **MemoTag's Speech Intelligence Module** , a proof-of-concept system that leverages **audio feature engineering + machine learning** to detect **early signs of cognitive decline** — all from just a voice sample. 🎙️🧬

---

## 🔍 Problem Statement

Cognitive impairment often first reveals itself in subtle changes in speech: increased pauses, hesitation words, inconsistent pitch, or trouble recalling words.

The goal of this project is to **detect those indicators** from voice recordings and classify them as  **healthy** ,  **mild decline** , or **slow decline** using unsupervised + supervised ML.

---

## 🌟 Objectives

* Preprocess audio files and convert speech to text (planned)
* Extract cognitive-indicative features such as:
  * 🎵 MFCCs
  * 🎼 Chroma features
  * 🗣️ Speech rhythm
  * 🚧 Hesitation markers *(planned)*
  * 🧠 Word recall/naming tasks *(planned NLP features)*
* Use ML to detect early decline:
  * Supervised Random Forest classification
  * Unsupervised Isolation Forest anomaly detection *(optional)*

---

## 🛠️ Tech Stack

| Layer               | Tech Used                                        |
| ------------------- | ------------------------------------------------ |
| Audio Parsing       | `librosa`,`soundfile`,`audioread`          |
| ML Model            | `scikit-learn`(RandomForest, Isolation Forest) |
| Feature Engineering | MFCC, Chroma, Zero-Cross                         |
| Deployment Ready    | Python CLI scripts                               |
| Future Plan         | Flask API + Frontend UI                          |

---

## 🗂️ Project Structure

```
cognitive_voice_pipeline/
├── data/                  # Input audio files (m4a, wav, etc.)
│   └── Recording.m4a
├── model/                 # Saved ML models
│   └── voice_model.pkl
├── result/                # Output predictions
│
├── src/
│   ├── extract_features.py   # MFCC, Chroma, etc.
│   ├── train.py              # Train and save RandomForestClassifier
│   └── predict.py            # Load model and predict cognitive state
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/cognitive_voice_pipeline.git
cd cognitive_voice_pipeline
```

### 2. Set up your Python environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use

### 👉 Train the model

```bash
python src/train.py
```

### 👉 Predict cognitive state from new audio

```bash
python src/predict.py data/Recording.m4a
```

---

## 📊 Features Extracted

* **MFCC (Mel-Frequency Cepstral Coefficients):** Captures timbre and frequency details.
* **Chroma Features:** Harmonic content for identifying pitch variation.
* **Speech Rate & Pauses:** Derived from signal duration *(NLP features to come)*
* **Pitch Variability:** Using spectral features *(future enhancement)*

---

## 🧠 Machine Learning

* **Random Forest Classifier** for supervised classification.
* **Isolation Forest** for unsupervised outlier detection *(optional)*

> Future improvements: add NLP-based sentence analysis and validate against medical datasets.

---

## 📈 Sample Output

```
MFCC shape: (13,)
Chroma shape: (12,)
Combined feature shape: (25,)
Final feature shape: (26,)
Predicted cognitive state: slow_decline
```

---

## 🔮 Planned Enhancements

* Integrate speech-to-text transcription (e.g., via `whisper` or `SpeechRecognition`)
* NLP analysis of fillers, sentence lag, and task-based language recall
* Add MFCC/Chroma visualizations
* Build a frontend + REST API

---

## 👩‍⚕️ Disclaimer

This is a research prototype only. Do not use for diagnosis or clinical decision-making. Please consult licensed professionals.

---

## ✨ Contributors

* **Developer:** Varsha
* **Project Advisor:** MemoTag AI team

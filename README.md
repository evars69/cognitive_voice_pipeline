
# 🧠 MemoTag | Voice-Based Cognitive Decline Detection

Welcome to **MemoTag's Speech Intelligence Module**, a research prototype that leverages **audio feature engineering + machine learning** to detect **early signs of cognitive decline** — all from just a voice sample. 🎙️🧬

🌐 **Live API (Render)**: https://cognitive-voice-pipeline-1.onrender.com  
🖥️ **Local Server**: http://127.0.0.1:5000

---

## 🔍 Problem Statement

Cognitive impairment often first reveals itself in subtle changes in speech: increased pauses, hesitation words, inconsistent pitch, or trouble recalling words.

This system aims to **detect those indicators** from voice recordings and classify them into:

- ✅ **Healthy**
- ⚠️ **Mild Decline**
- 🔴 **Slow Decline**

---

## 🌟 Objectives

- 🎯 Preprocess audio files and (planned) convert speech to text
- 🎵 Extract cognitive-indicative features like:
  - MFCCs (Mel-Frequency Cepstral Coefficients)
  - Chroma features
  - Speech rhythm
  - 🚧 Hesitation markers *(planned)*
  - 🧠 Word recall & naming tasks *(planned NLP features)*
- 🧠 Train ML models to detect decline:
  - ✅ Random Forest Classifier
  - 🔍 Isolation Forest *(optional)*

---

## 🛠️ Tech Stack

| Layer               | Tech Used                                         |
| ------------------- | ------------------------------------------------- |
| Audio Parsing       | `librosa`, `soundfile`, `audioread`         |
| ML Model            | `scikit-learn` (RandomForest, Isolation Forest) |
| Feature Engineering | MFCC, Chroma, Zero-Cross                          |
| API Server          | `Flask`                                         |
| Deployment          | `Render`, `Postman` testing                   |
| Future Additions    | Speech-to-text, NLP, Frontend UI                  |

---

## 🗂️ Project Structure

cognitive_voice_pipeline/  
├── data/              # Input audio files (m4a, wav, etc.)  
├── model/             # Saved ML models  
├── result/            # Output predictions  
├── src/  
│   ├── extract_features.py  
│   ├── train.py  
│   ├── predict.py  
├── app.py             # Flask API server (root-level)  
├── requirements.txt  
└── README.md


---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/evars69/cognitive_voice_pipeline.git
cd cognitive_voice_pipeline
```

---

## 🚀 Usage Guide

### 🧠 Train the model

* python src/train.py

### 🔍 Predict from CLI

* python src/predict.py data/Recording.m4a

## 🌐 API Usage

### ▶️ Run Locally

* python app.py

Then in Postman or any REST client:

* **Method** : `POST`
* **URL** : `http://127.0.0.1:5000/predict`
* **Form-Data Key** : `file`
* **Value** : Upload your `.wav` or `.m4a` audio file

### 🌍 Deployed on Render

> Note: Render may cause cold-start delays. If it's inactive for a while, the first request can be slow.

**API URL: [https://cognitive-voice-pipeline-1.onrender.com](https://cognitive-voice-pipeline-1.onrender.com)**

---

## 📸 Postman Screenshot(Attached)

---

## 📊 Features Extracted

* 🎵  **MFCC** : Mel-Frequency Cepstral Coefficients  
* 🎼  **Chroma Features** : Pitch & harmony indicators  
* ⏱️  **Zero-Crossing Rate** : Speech texture/energy  
* 🧮  **Combined Vector Size** : 26 features total

---

## 🧠 ML Models Used

* ✅ **Random Forest Classifier** (supervised)  
* 🧪 **Isolation Forest** (optional anomaly detection)

---

## 🔮 Planned Enhancements

* 📜 Whisper-based speech-to-text integration  
* 🧠 NLP for fillers, pauses, sentence structure  
* 📊 MFCC/Chroma visualizations  
* 💻 Frontend with React + integrated Flask API

---

## 📈 Sample Output

- 🎵 MFCC shape: `(13,)`  
- 🎼 Chroma shape: `(12,)`  
- 🧮 Combined feature shape: `(25,)`  
- 🧠 Final feature shape: `(26,)`  
- 🩺 Predicted cognitive state: `slow_decline`

---

## ⚠️ Disclaimer

* This is a **research prototype only**. Not intended for clinical or diagnostic purposes. Always consult a qualified professional for healthcare decisions.

---

## ✨ Contributors

* 👩‍💻 **Developer:** Varsha  
* 🧪 **Advisors:** MemoTag AI Team

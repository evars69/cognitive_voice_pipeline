import whisper

# Load Whisper model
model = whisper.load_model("base")

# Path to audio file
audio_path = "data/Recording.m4a"  

# Load audio and prepare it for transcription
audio = whisper.load_audio(audio_path)
audio = whisper.pad_or_trim(audio)

# Make prediction
result = model.transcribe(audio)

# Print transcribed text
print("Transcription: ", result["text"])

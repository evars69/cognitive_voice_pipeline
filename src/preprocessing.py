import whisper

# Load Whisper model
model = whisper.load_model("base")

# Path to your audio file (replace with your actual file path)
audio_path = "data/Recording.m4a"  # Update with the actual path

# Load audio and prepare it for transcription
audio = whisper.load_audio(audio_path)
audio = whisper.pad_or_trim(audio)

# Make prediction
result = model.transcribe(audio)

# Print transcribed text
print("Transcription: ", result["text"])

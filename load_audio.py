import librosa

def load_audio(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None

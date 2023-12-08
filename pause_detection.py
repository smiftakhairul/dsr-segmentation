import librosa
import numpy as np

def detect_pauses(rms_energy, threshold_rms, min_pause_duration=0.5, sr=44100, hop_length=512):
    try:
        # Identify intervals where the RMS energy is below the threshold
        pause_intervals = []
        in_pause = False
        for i in range(len(rms_energy)):
            if rms_energy[i] < threshold_rms:
                if not in_pause:
                    start = librosa.frames_to_time(i, hop_length=hop_length)
                    in_pause = True
            elif in_pause:
                end = librosa.frames_to_time(i, hop_length=hop_length)
                pause_duration = end - start
                if pause_duration >= min_pause_duration:
                    pause_intervals.append((start, end))
                in_pause = False

        return pause_intervals
    except Exception as e:
        print(f"Error in pause detection: {e}")
        return None

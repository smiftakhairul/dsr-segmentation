import librosa
import numpy as np

def amplitude_analysis(audio_data, threshold_rms=0.01, frame_length=2048, hop_length=512):
    try:
        # Calculate the root mean square (RMS) energy
        rms_energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]

        # Normalize the RMS energy values
        rms_energy /= np.max(rms_energy)

        # Identify intervals where the RMS energy exceeds the threshold
        speech_intervals = []
        in_speech = False
        for i in range(len(rms_energy)):
            if rms_energy[i] > threshold_rms:
                if not in_speech:
                    start = librosa.frames_to_time(i, hop_length=hop_length)
                    in_speech = True
            elif in_speech:
                end = librosa.frames_to_time(i, hop_length=hop_length)
                speech_intervals.append((start, end))
                in_speech = False

        return speech_intervals, rms_energy
    except Exception as e:
        print(f"Error in amplitude analysis: {e}")
        return None, None

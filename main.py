import os
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment

from load_audio import load_audio
from amplitude_analysis import amplitude_analysis
from pause_detection import detect_pauses
from speech_segmentation import speech_segmentation

def plot_audio_waveform(audio_data, sr):
    plt.figure(figsize=(15, 4))
    librosa.display.waveshow(audio_data, sr=sr)
    plt.title('Audio Waveform')
    plt.show()

def plot_rms_energy(times, rms_energy, threshold_rms):
    plt.figure(figsize=(10, 4))
    plt.plot(times, rms_energy, label='RMS Energy')
    plt.axhline(threshold_rms, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Energy')
    plt.legend()
    plt.title('RMS Energy of the Audio Signal')
    plt.show()

def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def save_speech_segments(audio_data, speech_segments, sr, output_dir='./files/output_segments'):
    create_output_directory(output_dir)

    for i, segment in enumerate(speech_segments):
        start, end = segment
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        segment_data = audio_data[start_frame:end_frame]

        # Convert NumPy array to pydub AudioSegment
        segment_audio = AudioSegment(
            segment_data.tobytes(),
            frame_rate=sr,
            sample_width=segment_data.dtype.itemsize,
            channels=1  # Assuming mono audio
        )

        # Save the speech segment as a .wav file using pydub
        output_path = os.path.join(output_dir, f'segment_{i + 1}.wav')
        segment_audio.export(output_path, format="wav")

        print(f"Speech segment {i + 1} saved to {output_path}")


if __name__ == "__main__":
    audio_file_path = './files/audio.wav'
    audio_data, sr = load_audio(audio_file_path)
    
    if audio_data is not None:
        print(f"Audio loaded successfully. Shape: {audio_data.shape}, Sampling rate: {sr} Hz")
    else:
        print("Failed to load audio.")

    # Set parameters
    threshold_rms = 0.01
    hop_length = 512

    # Plot audio waveform
    plot_audio_waveform(audio_data, sr)

    if audio_data is not None:
        # Perform amplitude analysis
        speech_intervals, rms_energy = amplitude_analysis(audio_data, threshold_rms, hop_length)

        if speech_intervals is not None:
            print(f"Total Speech intervals: {len(speech_intervals)}")
            
            # Plot RMS energy
            times = librosa.times_like(rms_energy, sr=sr, hop_length=hop_length)
            plot_rms_energy(times, rms_energy, threshold_rms)

            # Detect pauses
            pause_intervals = detect_pauses(rms_energy, threshold_rms)
            if pause_intervals is not None:
                print(f"Total Pause intervals: {len(pause_intervals)}")

                # Segment the audio based on detected pauses
                audio_duration = librosa.get_duration(y=audio_data, sr=sr)
                speech_segments = speech_segmentation(pause_intervals, audio_duration)

                if speech_segments is not None:
                    print(f"Speech segments: {len(speech_segments)}")

                    # Save speech segments to .wav files
                    save_speech_segments(audio_data, speech_segments, sr)
                else:
                    print("No speech segments found.")
            else:
                print("No pauses detected.")
        else:
            print("No speech intervals found. Adjust the threshold_rms.")
    else:
        print("Failed to load audio.")

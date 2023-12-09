import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment
from sklearn.cluster import KMeans

from load_audio import load_audio
from amplitude_analysis import amplitude_analysis
from pause_detection import detect_pauses
from speech_segmentation import speech_segmentation

def auto_adjust_parameters(audio_data, sr):
    # Extract features from the audio data
    rms_energy = librosa.feature.rms(y=audio_data)
    features = np.array([np.mean(rms_energy), np.std(rms_energy)])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=2, n_init=10)  # Explicitly set the value of n_init
    kmeans.fit(features.reshape(-1, 1))

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Determine threshold_rms and hop_length based on cluster centers
    threshold_rms = cluster_centers.min() * 1.5  # Adjust the multiplier as needed
    hop_length = int(np.floor(len(audio_data) / (5 * sr)))  # Adjust the divisor as needed

    return threshold_rms, hop_length

def load_and_print_audio(audio_file_path):
    audio_data, sr = load_audio(audio_file_path)
    
    if audio_data is None:
        print("Failed to load audio.")
    else:
        print(f"Audio loaded successfully. Shape: {audio_data.shape}, Sampling rate: {sr} Hz")
    
    return audio_data, sr

def analyze_audio(audio_data, threshold_rms, hop_length):
    speech_intervals, rms_energy = amplitude_analysis(audio_data, threshold_rms, hop_length)
    return speech_intervals, rms_energy

def plot_waveform(audio_data, sr):
    plt.figure(figsize=(15, 4))
    librosa.display.waveshow(audio_data, sr=sr, color="navy")
    plt.title('Audio Waveform')
    plt.show()

def plot_rms_energy(audio_data, sr, threshold_rms, hop_length):
    times = librosa.times_like(audio_data, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(10, 4))
    plt.plot(times, audio_data, label='RMS Energy', color="navy")
    plt.axhline(threshold_rms, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Energy')
    plt.legend()
    plt.title('RMS Energy of the Audio Signal')
    plt.show()

def plot_speech_segments(audio_data, sr, speech_segments):
    plt.figure(figsize=(15, 4))
    times = librosa.times_like(audio_data, sr=sr)
    
    # Plot the audio waveform
    librosa.display.waveshow(audio_data, sr=sr, color="navy", alpha=0.5)

    # Highlight speech segments
    for segment in speech_segments:
        start_time, end_time = segment
        plt.axvspan(start_time, end_time, color='r', alpha=0.5)

    plt.title('Speech Segments')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def detect_and_segment_pauses(rms_energy, threshold_rms, audio_data, sr):
    pause_intervals = detect_pauses(rms_energy, threshold_rms)

    if not pause_intervals:
        print("No pauses detected.")
        return None

    print(f"Total Pause intervals: {len(pause_intervals)}")

    audio_duration = librosa.get_duration(y=audio_data, sr=sr)
    speech_segments = speech_segmentation(pause_intervals, audio_duration)

    if not speech_segments:
        print("No speech segments found.")
        return None

    print(f"Total Speech segments: {len(speech_segments)}")
    return speech_segments

def create_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def save_speech_segment(audio_data, segment, sr, output_path):
    start_frame, end_frame = map(int, (segment[0] * sr, segment[1] * sr))
    segment_audio = audio_data[start_frame:end_frame]

    try:
        sf.write(output_path, segment_audio, sr)
    except Exception as e:
        print(f"Error saving speech segment: {e}")

def save_segments(audio_data, speech_segments, sr, output_dir='./files/output_segments'):
    create_output_directory(output_dir)

    for i, segment in enumerate(speech_segments):
        output_path = os.path.join(output_dir, f'segment_{i + 1}.wav')
        save_speech_segment(audio_data, segment, sr, output_path)
        print(f"Speech segment {i + 1} saved to {output_path}")

def main():
    audio_file_path = './files/audio.wav'
    audio_data, sr = load_and_print_audio(audio_file_path)

    if audio_data is None:
        return
    
    # # Automatically adjust parameters using unsupervised learning
    # threshold_rms, hop_length = auto_adjust_parameters(audio_data, sr)
    
    # Set parameters
    threshold_rms = 0.002
    hop_length = 512
    
    print(f"Determined threshold_rms: {threshold_rms}")
    print(f"Determined hop_length: {hop_length}")

    # Plot audio waveform
    plot_waveform(audio_data, sr)

    # Perform amplitude analysis
    speech_intervals, rms_energy = analyze_audio(audio_data, threshold_rms, hop_length)

    if not speech_intervals:
        print("No speech intervals found. Adjust the threshold_rms.")
        return

    print(f"Total Speech intervals: {len(speech_intervals)}")

    # Plot RMS energy
    plot_rms_energy(rms_energy, sr, threshold_rms, hop_length)

    # Detect pauses and segment the audio
    speech_segments = detect_and_segment_pauses(rms_energy, threshold_rms, audio_data, sr)

    if speech_segments is not None:
        # Save speech segments to .wav files
        save_segments(audio_data, speech_segments, sr)
        
        # Plot speech segments on the audio waveform
        plot_speech_segments(audio_data, sr, speech_segments)

if __name__ == "__main__":
    main()

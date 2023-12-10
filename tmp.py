from pydub import AudioSegment
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt
import numpy as np

def load_audio(audio_file):
    return AudioSegment.from_wav(audio_file)

def detect_pauses(audio, silence_thresh=-40):
    # Split audio on silence to detect pauses
    pauses = split_on_silence(audio, silence_thresh=silence_thresh)
    return pauses

def plot_audio_waveform(audio, title="Original Audio Waveform"):
    # Plot the audio waveform
    plt.figure(figsize=(10, 4))
    plt.plot(audio.get_array_of_samples())

    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()

def plot_pauses(audio, pauses, title="Detected Pauses"):
    # Get the audio samples as a numpy array
    samples = np.array(audio.get_array_of_samples())

    # Create a time axis for the samples
    time = np.arange(0, len(samples)) / audio.frame_rate

    # Plot the audio waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, samples, label="Audio Waveform")

    # Highlight the detected pauses
    cumulative_duration = 0
    for pause in pauses:
        pause_start = cumulative_duration
        cumulative_duration += pause.duration_seconds
        pause_end = cumulative_duration
        plt.axvspan(pause_start, pause_end, color='red', alpha=0.3)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Set x-axis limits slightly beyond the maximum audio length
    plt.xlim(0, audio.duration_seconds + 1)

    # Create a custom legend outside of the loop
    plt.legend(['Audio Waveform', 'Pause'], loc='upper right')

    plt.show()

def segment_audio_by_pauses(audio, pauses):
    # Segment the audio based on detected pauses
    segments = []
    start_time = 0

    # Iterate through pauses and add segments between pauses
    for pause in pauses:
        # Pause is an AudioSegment object
        pause_start = start_time
        pause_end = start_time + len(pause)
        
        # Add the segment before the pause
        segments.append(audio[pause_start:pause_end])

        # Update the start time for the next segment
        start_time = pause_end

    # Add the last segment from the end of the last pause to the end of the audio
    segments.append(audio[start_time:])

    return segments

def save_segments(segments, output_folder="./files/output_segments"):
    # Save each segment to a new file
    for i, segment in enumerate(segments):
        segment.export(f"{output_folder}/segment_{i + 1}.wav", format="wav")

if __name__ == "__main__":
    audio_file_path = "./files/audio_en.wav"
    
    # Step 2: Load the audio
    audio = load_audio(audio_file_path)
    
    # Step 3: Plot the audio waveform
    plot_audio_waveform(audio, title="Original Audio Waveform")
    
    # Step 4: Detect pauses
    pauses = detect_pauses(audio)

    
    # Step 5: Plot the audio waveform with detected pauses
    plot_pauses(audio, pauses, title="Detected Pauses")
    
    # Step 6: Segment audio based on pauses
    segments = segment_audio_by_pauses(audio, pauses)
    
    # Step 7: Save segments
    save_segments(segments)

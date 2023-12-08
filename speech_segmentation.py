def speech_segmentation(pause_intervals, audio_duration):
    try:
        # Segment the audio based on pause intervals
        speech_segments = []
        if len(pause_intervals) > 0:
            # Add the first speech segment (from the start of the audio to the first pause)
            speech_segments.append((0.0, pause_intervals[0][0]))

            # Add speech segments between pauses
            for i in range(len(pause_intervals) - 1):
                start = pause_intervals[i][1]
                end = pause_intervals[i + 1][0]
                speech_segments.append((start, end))

            # Add the last speech segment (from the last pause to the end of the audio)
            speech_segments.append((pause_intervals[-1][1], audio_duration))

        return speech_segments
    except Exception as e:
        print(f"Error in speech segmentation: {e}")
        return None

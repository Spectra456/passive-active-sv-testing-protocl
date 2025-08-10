from pyannote.audio import Model, Inference
from pathlib import Path
from tqdm import tqdm


# Load the pretrained model (replace the token with your actual access token)
model = Model.from_pretrained("pyannote/brouhaha")

inference = Inference(model)

def analyze_audio(audio_path: str, vad_threshold: float = 0.35) -> dict:
    """
    Analyze an audio file to compute speech duration (using a VAD threshold),
    and average SNR and C50 values over the speech frames.

    Parameters:
        audio_path (str): Path to the audio file.
        vad_threshold (float): Minimum VAD value to consider a frame as speech (default is 0.35).

    Returns:
        dict: A dictionary with keys:
              "speech_duration" (in seconds),
              "avg_snr" (average SNR over speech frames),
              "avg_c50" (average C50 over speech frames).
    """

    # Run inference on the audio file.
    # The output is an iterator over frames, where each element is a tuple:
    # (frame, (vad, snr, c50))
    output = list(inference(audio_path))
    
    # If there are no frames, return zeros
    if not output:
        return {"speech_duration": 0.0, "avg_snr": 0.0, "avg_c50": 0.0}
    
    # Estimate the frame duration using the middle timestamps of the first and last frames.
    if len(output) > 1:
        start = output[0][0].middle
        end = output[-1][0].middle
        frame_duration = (end - start) / (len(output) - 1)
    else:
        frame_duration = 0.017  # Fallback value if only one frame is present.
    
    # Filter out the frames that are considered speech (vad >= threshold)
    speech_frames = [(frame, vad, snr, c50) for frame, (vad, snr, c50) in output if vad >= vad_threshold]
    n_speech = len(speech_frames)
    
    # Calculate total speech duration (approximate)
    speech_duration = n_speech * frame_duration
    
    # Compute average SNR and C50 over speech frames if available.
    if n_speech > 0:
        avg_snr = sum(snr for _, _, snr, _ in speech_frames) / n_speech
        avg_c50 = sum(c50 for _, _, _, c50 in speech_frames) / n_speech
    else:
        avg_snr = 0.0
        avg_c50 = 0.0
    
    return {"speech_duration": speech_duration, "avg_snr": avg_snr, "avg_c50": avg_c50}


# Example usage:
if __name__ == "__main__":
    audio_file = "/mnt/datasets/voxconverse_dev/audio/abjxc.wav"
    results = analyze_audio(audio_file)
    print(f"Speech duration: {results['speech_duration']:.3f} s")
    print(f"Average SNR: {results['avg_snr']:.0f}")
    print(f"Average C50: {results['avg_c50']:.0f}")


# Paths
audio_root = Path('./vox1o')
trials_file = './veri_test.txt'
scores_output_file = 'scores_snr_vad_c50.txt'

# Step 1: Load trials
pairs = []
labels = []
with open(trials_file, 'r') as f:
    for line in f:
        label, audio1, audio2 = line.strip().split()
        labels.append(int(label))
        pairs.append((audio1, audio2))


# Step 3: Calculate similarity scores
with open(scores_output_file, 'w') as sf:
    for audio1, audio2 in tqdm(pairs, desc="Calculating similarity scores"):
        file1 = audio_root / audio1
        file2 = audio_root / audio2
        results1 = analyze_audio(file1)
        results2 = analyze_audio(file2)

        sf.write(f"{audio1} {audio2} {results1['speech_duration']:.3f} {results1['avg_snr']:.0f} {results1['avg_c50']:.0f} {results2['speech_duration']:.3f} {results2['avg_snr']:.0f} {results2['avg_c50']:.0f}\n")

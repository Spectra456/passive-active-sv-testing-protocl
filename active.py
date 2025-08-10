import os
import subprocess
import random
import argparse
from tqdm import tqdm
import pandas as pd

# Speaker verification dependencies
import torch
import torchaudio
import onnxruntime as ort
import torchaudio.compliance.kaldi as kaldi
from torchaudio.transforms import FrequencyMasking
from sklearn.metrics.pairwise import cosine_similarity

# Custom EER and minDCF calculations
from compute_EER import calculate_eer
from compute_min_dcf import ComputeErrorRates, ComputeMinDcf

# Supported conditions
DOWNSAMPLE_RATES = [16000, 8000]
CODECS = {
    'mp3':    {'format': 'mp3',                'bitrate': '128k',   'ext': 'mp3'},
    'aac':    {'format': 'aac',                'bitrate': '128k',   'ext': 'aac'},
    'opus':   {'format': 'libopus',            'bitrate': '16k',    'ext': 'opus'},
    'gsm':    {'format': 'libgsm',             'bitrate': None,     'ext': 'gsm'}
}
NOISE_LEVELS = [5, 10, 15, 20, 25]       # SNR in dB for Gaussian noise
VOLUME_LEVELS = [-30, -20, -10, 10, 20, 30]  # dB adjustments for volume

# ### NEW: Speed‐change factors (FFmpeg atempo supports 0.5–2.0 directly)
SPEED_FACTORS = [0.5, 1.5, 2.0]

# ### NEW: Time‐mask durations (in seconds)
TIME_MASK_DURS = [0.1, 0.25, 0.5]

# ### NEW: Frequency‐mask widths (in number of mel‐bins)
#    These are “freq_mask_param” values for torchaudio.transforms.FrequencyMasking.
#    In practice, 15–30 is common. Here, we choose one “useful for speech” example (e.g. 15).
FREQ_MASK_PARAMS = [15]


class Verification:
    """
    ONNX‐based speaker verification class
    """
    def __init__(self, model_path='eng_rus.onnx'):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 10
        so.intra_op_num_threads = 10
        self.session = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

    def compute_fbank(self, wav_path,
                      num_mel_bins=80,
                      frame_length=25,
                      frame_shift=10,
                      dither=1.0):
        waveform, sample_rate = torchaudio.load(wav_path)
        # Scale to 16‐bit int range for Kaldi
        waveform = waveform * (1 << 15)
        feat = kaldi.fbank(
            waveform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            sample_frequency=sample_rate,
            window_type='hamming',
            use_energy=False
        )
        feat = feat - torch.mean(feat, dim=0)
        return feat  # shape: [T, num_mel_bins]

    def verify_speakers(self, file1, file2):
        feats1 = self.compute_fbank(file1).unsqueeze(0).numpy()
        feats2 = self.compute_fbank(file2).unsqueeze(0).numpy()
        emb1 = self.session.run(['embs'], {'feats': feats1})[0]
        emb2 = self.session.run(['embs'], {'feats': feats2})[0]
        score = cosine_similarity(emb1, emb2)[0][0]
        return score

    # ### NEW: Verify with frequency masking applied to filterbank features.
    def verify_speakers_with_freq_mask(self, file1, file2, mask_param):
        """
        Load each file, compute fbank, apply FrequencyMasking (SpecAugment style),
        then run through ONNX model and cosine‐compare.
        """
        # 1) Compute raw fbank (torch.Tensor [T, F])
        feat1 = self.compute_fbank(file1)  # [T, F]
        feat2 = self.compute_fbank(file2)  # [T, F]

        # 2) torchaudio.transforms.FrequencyMasking expects input shape [batch, freq, time].
        #    Our feat is [time, freq], so we transpose, unsqueeze, mask, and transpose back.
        fm = FrequencyMasking(freq_mask_param=mask_param)

        # Mask for feat1
        f1 = feat1.transpose(0, 1).unsqueeze(0)  # [1, freq, time]
        f1_masked = fm(f1).squeeze(0).transpose(0, 1)  # [time, freq]

        # Mask for feat2
        f2 = feat2.transpose(0, 1).unsqueeze(0)
        f2_masked = fm(f2).squeeze(0).transpose(0, 1)

        # 3) Convert to numpy + add batch dim
        feats1 = f1_masked.unsqueeze(0).numpy()
        feats2 = f2_masked.unsqueeze(0).numpy()

        emb1 = self.session.run(['embs'], {'feats': feats1})[0]
        emb2 = self.session.run(['embs'], {'feats': feats2})[0]
        score = cosine_similarity(emb1, emb2)[0][0]
        return score


def add_gaussian_noise(input_path, output_path, snr_db):
    """
    Add Gaussian noise to an audio file to achieve the specified SNR (in dB).
    - Loads waveform with torchaudio (shape: [channels, num_samples], values in [-1,1])
    - Computes signal power (mean squared), then noise power for desired SNR.
    - Generates white Gaussian noise, adds it, and prevents clipping by normalization if needed.
    - Saves the noisy waveform back to disk as a WAV (same sample rate).
    """
    waveform, sr = torchaudio.load(input_path)  # [C, L], sample values in [-1,1]
    # Compute signal power (mean squared over all channels & samples)
    power_signal = waveform.pow(2).mean().item()
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    # Compute required noise power: P_noise = P_signal / SNR_linear
    power_noise = power_signal / snr_linear
    # Generate Gaussian noise: zero‐mean, variance = power_noise
    noise = torch.randn_like(waveform) * torch.sqrt(torch.tensor(power_noise))
    noisy = waveform + noise

    # Prevent clipping: if any sample exceeds [-1,1], scale down
    max_val = noisy.abs().max().item()
    if max_val > 1.0:
        noisy = noisy / max_val

    # Save the noisy waveform as a WAV (floating‐point) at the same sample rate
    torchaudio.save(output_path, noisy, sr)


def modify_volume(input_path, output_path, db_change):
    """
    Modify an audio file's volume by a specified number of dB.
    - Loads waveform with torchaudio (shape: [channels, num_samples], values in [-1,1])
    - Computes linear gain = 10^(db_change/20), multiplies waveform by gain
    - Clips or normalizes if needed to prevent exceeding [-1,1]
    - Saves the adjusted waveform back to disk as a WAV at the same sample rate
    """
    waveform, sr = torchaudio.load(input_path)  # [C, L], sample values in [-1,1]
    gain = 10 ** (db_change / 20.0)  # linear gain factor
    adjusted = waveform * gain

    # If any sample goes beyond [-1,1], scale entire waveform
    max_val = adjusted.abs().max().item()
    if max_val > 1.0:
        adjusted = adjusted / max_val

    torchaudio.save(output_path, adjusted, sr)


def ffmpeg_convert(input_path, output_path, rate=None, codec=None):
    cmd = ['ffmpeg', '-y', '-i', input_path]

    # Apply explicit down-sampling if requested
    if rate:
        cmd += ['-ar', str(rate)]

    if codec:
        fmt = codec['format']

        # Special-case for GSM: enforce 8 kHz mono
        if fmt == 'libgsm':
            cmd += ['-ar', '8000', '-ac', '1']
        # AMR-NB requires 8 kHz mono
        elif fmt == 'libopencore_amrnb':
            cmd += ['-ar', '8000', '-ac', '1']
        # AMR-WB requires 16 kHz mono
        elif fmt == 'libvo_amrwbenc':
            cmd += ['-ar', '16000', '-ac', '1']

        cmd += ['-acodec', fmt]

        # Optional bitrate
        if codec.get('bitrate'):
            cmd += ['-b:a', codec['bitrate']]

    cmd.append(output_path)
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


# ### NEW: Speed‐change using FFmpeg’s atempo filter
def change_speed(input_path, output_path, speed_factor):
    """
    Change the playback speed by `speed_factor` (0.5, 1.5, 2.0, etc.) using FFmpeg’s atempo.
    atempo supports values in [0.5, 2.0] directly. If you need <0.5 or >2.0 in one shot,
    you would chain multiple atempo filters, but here our factors are within [0.5, 2.0].
    """
    # Example: ffmpeg -y -i in.wav -filter:a "atempo=1.5" out.wav
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-filter:a', f"atempo={speed_factor}",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


# ### NEW: Time masking (zero out a random chunk of length `mask_dur` seconds)
def add_time_mask(input_path, output_path, mask_dur):
    """
    Zero out a random contiguous segment of length mask_dur (in seconds) in the waveform.
    - Loads waveform with torchaudio.
    - Selects a random start so that the mask fits entirely.
    - Sets that segment to zero, then re‐normalizes if needed.
    - Saves the result to `output_path` (same sample rate).
    """
    waveform, sr = torchaudio.load(input_path)  # [C, L], values in [-1,1]
    num_samples = waveform.size(1)
    mask_length = int(sr * mask_dur)

    if mask_length >= num_samples:
        # If mask is as long as the entire audio, just zero out everything.
        masked = torch.zeros_like(waveform)
    else:
        start = random.randint(0, num_samples - mask_length)
        end = start + mask_length
        masked = waveform.clone()
        masked[:, start:end] = 0.0

        # If zeroing created very low‐energy segments, we can re‐normalize entire waveform:
        max_val = masked.abs().max().item()
        if max_val > 1.0:
            masked = masked / max_val

    torchaudio.save(output_path, masked, sr)


# load_trials and main() are mostly unchanged, except for adding new conditions and branches.

def load_trials(trials_file, num_pairs=100, seed=42):
    trials = []
    with open(trials_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            label = bool(int(parts[0]))
            trials.append((parts[1], parts[2], label))
    random.Random(seed).shuffle(trials)
    #return trials[:num_pairs]
    return trials



def main(args):
    verifier = Verification(model_path=args.model)
    trials = load_trials(args.trials, num_pairs=args.num_pairs)
    results = []

    tmp_dir = os.path.join(args.output, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # Build combined list of conditions:
    #  - 'orig'
    #  - downsampling (e.g., 'ds16000', 'ds8000')
    #  - codecs ('mp3', 'aac', 'opus', 'gsm')
    #  - Gaussian noise ('noise5', 'noise10', ...)
    #  - volume modifications ('vol-30', 'vol-20', ...)
    #  - speed changes ('speed0.5', 'speed1.5', 'speed2.0')
    #  - time masking ('timemask0.1', 'timemask0.25', 'timemask0.5')
    #  - frequency masking ('freqmask15')  # mask_param=15 as example
    conditions = (
        ['orig'] +
        [f"ds{r}" for r in DOWNSAMPLE_RATES] +
        list(CODECS.keys()) +
        [f"noise{n}" for n in NOISE_LEVELS] +
        [f"vol{v}" for v in VOLUME_LEVELS] +
        [f"speed{sf}" for sf in SPEED_FACTORS] +                            ### NEW
        [f"timemask{int(d*1000)}" for d in TIME_MASK_DURS] +                 ### NEW
        [f"freqmask{p}" for p in FREQ_MASK_PARAMS]                            ### NEW
    )

    for cond in conditions:
        scores, labels = [], []
        print(f"Evaluating condition: {cond}")
        for p1, p2, lbl in tqdm(trials):
            p1_path = os.path.join(args.vox1_dir, p1)
            p2_path = os.path.join(args.vox1_dir, p2)

            # By default, use the original WAVs
            if cond == 'orig':
                f1, f2 = p1_path, p2_path

            # Downsampling, e.g. "ds16000", "ds8000"
            elif cond.startswith('ds'):
                rate = int(cond[2:])
                f1 = os.path.join(tmp_dir, f"{os.path.basename(p1)}_{cond}.wav")
                f2 = os.path.join(tmp_dir, f"{os.path.basename(p2)}_{cond}.wav")
                ffmpeg_convert(p1_path, f1, rate=rate)
                ffmpeg_convert(p2_path, f2, rate=rate)

            # Codec‐based condition: encode → decode back to 16 kHz WAV
            elif cond in CODECS:
                codec = CODECS[cond]
                ext = codec['ext']

                # Encode step
                o1 = os.path.join(tmp_dir, f"{os.path.basename(p1)}_{cond}.{ext}")
                o2 = os.path.join(tmp_dir, f"{os.path.basename(p2)}_{cond}.{ext}")
                ffmpeg_convert(p1_path, o1, codec=codec)
                ffmpeg_convert(p2_path, o2, codec=codec)

                # Decode back to WAV at 16 kHz
                f1 = o1.replace(f".{ext}", f"_{cond}.wav")
                f2 = o2.replace(f".{ext}", f"_{cond}.wav")
                ffmpeg_convert(o1, f1, rate=16000)
                ffmpeg_convert(o2, f2, rate=16000)

            # Gaussian noise condition, e.g. "noise10", "noise15", ...
            elif cond.startswith('noise'):
                snr_db = int(cond.replace('noise', ''))
                f1 = os.path.join(tmp_dir, f"{os.path.basename(p1)}_{cond}.wav")
                f2 = os.path.join(tmp_dir, f"{os.path.basename(p2)}_{cond}.wav")
                add_gaussian_noise(p1_path, f1, snr_db)
                add_gaussian_noise(p2_path, f2, snr_db)

            # Volume modification, e.g. "vol-15", "vol-10", "vol-5", "vol5", "vol10", "vol15"
            elif cond.startswith('vol'):
                db_change = int(cond.replace('vol', ''))
                f1 = os.path.join(tmp_dir, f"{os.path.basename(p1)}_{cond}.wav")
                f2 = os.path.join(tmp_dir, f"{os.path.basename(p2)}_{cond}.wav")
                modify_volume(p1_path, f1, db_change)
                modify_volume(p2_path, f2, db_change)

            # ### NEW: Speed‐change (using ffmpeg atempo)
            elif cond.startswith('speed'):
                sf = float(cond.replace('speed', ''))
                f1 = os.path.join(tmp_dir, f"{os.path.basename(p1)}_{cond}.wav")
                f2 = os.path.join(tmp_dir, f"{os.path.basename(p2)}_{cond}.wav")
                change_speed(p1_path, f1, sf)
                change_speed(p2_path, f2, sf)

            # ### NEW: Time masking, e.g. "timemask100" means 0.1 s (100 ms)
            elif cond.startswith('timemask'):
                # extract integer like 'timemask100' → 100 → divide by 1000 → 0.1 s
                ms = int(cond.replace('timemask', ''))
                dur = ms / 1000.0
                f1 = os.path.join(tmp_dir, f"{os.path.basename(p1)}_{cond}.wav")
                f2 = os.path.join(tmp_dir, f"{os.path.basename(p2)}_{cond}.wav")
                add_time_mask(p1_path, f1, dur)
                add_time_mask(p2_path, f2, dur)

            # ### NEW: Frequency masking (no intermediate WAV creation)
            elif cond.startswith('freqmask'):
                # We will skip creating any WAV for freq‐mask. Instead, we verify directly.
                mask_param = int(cond.replace('freqmask', ''))
                # Call the new method that masks features instead of raw audio:
                score = verifier.verify_speakers_with_freq_mask(p1_path, p2_path, mask_param)
                scores.append(score)
                labels.append(lbl)
                continue  # skip the “compute via f1/f2” below

            else:
                raise ValueError(f"Unknown condition: {cond}")

            # For all other conditions (orig, ds, codec, noise, vol, speed, timemask),
            # we now have temporary WAVs f1,f2. Compute score normally:
            score = verifier.verify_speakers(f1, f2)
            scores.append(score)
            labels.append(lbl)

        # After processing all trials for this condition, compute EER and minDCF
        eer, threshold_eer = calculate_eer(labels, scores, pos=1)
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        min_dcf, threshold_dcf = ComputeMinDcf(
            fnrs, fprs, thresholds,
            p_target=0.05, c_miss=1, c_fa=1
        )
        results.append({'condition': cond, 'EER': eer, 'minDCF': min_dcf})

    # Save the final results as CSV
    df = pd.DataFrame(results)
    os.makedirs(args.output, exist_ok=True)
    out_csv = os.path.join(args.output, 'condition_evaluation.csv')
    df.to_csv(out_csv, index=False)
    print(f"Done. Results at {out_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate codecs/downsampling/gaussian-noise/volume-modifications "
                    "plus speed, time‐masking, and freq‐masking with ONNX speaker verifier"
    )
    parser.add_argument('--vox1-dir', type=str, default='./vox1o')
    parser.add_argument('--trials', type=str, default='./veri_test.txt')
    parser.add_argument('--output', type=str, default='./results')
    parser.add_argument('--model', type=str, default='eng_rus.onnx')
    parser.add_argument('--num-pairs', type=int, default=10)
    args = parser.parse_args()
    main(args)

import os
import subprocess
import random
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np

# Speaker verification dependencies
import torch
import torchaudio
import onnxruntime as ort
import torchaudio.compliance.kaldi as kaldi
from sklearn.metrics.pairwise import cosine_similarity

# Verification metrics

# Supported conditions
dOWNSAMPLE_RATES = [16000, 8000]
CODECS = {
    'mp3': {'format': 'mp3', 'bitrate': '128k'},
    'aac': {'format': 'aac', 'bitrate': '128k'},
    'opus': {'format': 'opus', 'bitrate': '16k'},
    'gsm': {'format': 'gsm', 'bitrate': None},
    'amr_nb': {'format': 'amr', 'bitrate': '12.2k'},
    'amr_wb': {'format': 'amr', 'bitrate': '23.85k'}
}

class Verification:
    """
    ONNX-based speaker verification class
    """
    def __init__(self, model_path='eng_rus.onnx', threshold=0.45):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 10
        so.intra_op_num_threads = 10
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.threshold = threshold

    def compute_fbank(self, wav_path,
                    num_mel_bins=80,
                    frame_length=25,
                    frame_shift=10,
                    dither=1.0):
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform * (1 << 15)
        mat = kaldi.fbank(waveform,
                        num_mel_bins=num_mel_bins,
                        frame_length=frame_length,
                        frame_shift=frame_shift,
                        dither=dither,
                        sample_frequency=sample_rate,
                        window_type='hamming',
                        use_energy=False)
        mat = mat - torch.mean(mat, dim=0)
        return mat

    def verify_speakers(self, file1, file2):
        feats1 = self.compute_fbank(file1).unsqueeze(0).numpy()
        feats2 = self.compute_fbank(file2).unsqueeze(0).numpy()

        emb1 = self.session.run(['embs'], {'feats': feats1})[0]
        emb2 = self.session.run(['embs'], {'feats': feats2})[0]

        score = cosine_similarity(emb1, emb2)[0][0]
        return score


def ffmpeg_convert(input_path, output_path, rate=None, codec=None):
    cmd = ['ffmpeg', '-y', '-i', input_path]
    if rate:
        cmd += ['-ar', str(rate)]
    if codec:
        fmt = codec['format']
        cmd += ['-acodec', fmt]
        if codec.get('bitrate'):
            cmd += ['-b:a', codec['bitrate']]
    cmd.append(output_path)
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def load_trials(trials_file, num_pairs=100, seed=42):
    trials = []
    with open(trials_file, 'r') as f:
        for line in f:
            spk1, spk2, path1, path2 = line.strip().split()
            label = (spk1 == spk2)
            trials.append((path1, path2, label))
    random.Random(seed).shuffle(trials)
    return trials[:num_pairs]


def main(args):
    verifier = Verification(model_path=args.model, threshold=args.threshold)
    trials = load_trials(args.trials, num_pairs=args.num_pairs)
    results = []

    tmp_dir = os.path.join(args.output, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    for cond in ['orig'] + [f"ds{r}" for r in dOWNSAMPLE_RATES] + list(CODECS.keys()):
        scores, labels = [], []
        print(f"Evaluating condition: {cond}")
        for p1, p2, lbl in tqdm(trials):
            if cond == 'orig':
                f1, f2 = p1, p2
            elif cond.startswith('ds'):
                rate = int(cond[2:])
                f1 = os.path.join(tmp_dir, f"{os.path.basename(p1)}_{cond}.wav")
                f2 = os.path.join(tmp_dir, f"{os.path.basename(p2)}_{cond}.wav")
                ffmpeg_convert(p1, f1, rate=rate)
                ffmpeg_convert(p2, f2, rate=rate)
            else:
                codec = CODECS[cond]
                ext = codec['format'] if codec['format'] != 'amr' else 'amr'
                o1 = os.path.join(tmp_dir, f"{os.path.basename(p1)}_{cond}.{ext}")
                o2 = os.path.join(tmp_dir, f"{os.path.basename(p2)}_{cond}.{ext}")
                ffmpeg_convert(p1, o1, codec=codec)
                ffmpeg_convert(p2, o2, codec=codec)
                f1 = o1.replace(f".{ext}", f"_{cond}.wav")
                f2 = o2.replace(f".{ext}", f"_{cond}.wav")
                ffmpeg_convert(o1, f1, rate=16000)
                ffmpeg_convert(o2, f2, rate=16000)

            score = verifier.verify_speakers(f1, f2)
            scores.append(score)
            labels.append(lbl)

        eer = compute_eer(scores, labels)
        mindcf = compute_min_dcf(scores, labels)
        results.append({'condition': cond, 'EER': eer, 'minDCF': mindcf})

    df = pd.DataFrame(results)
    os.makedirs(args.output, exist_ok=True)
    df.to_csv(os.path.join(args.output, 'condition_evaluation.csv'), index=False)
    print("Done. Results at condition_evaluation.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate codecs/downsampling with ONNX speaker verifier")
    parser.add_argument('--vox1-dir', type=str, required=True)
    parser.add_argument('--trials', type=str, required=True)
    parser.add_argument('--output', type=str, default='./results')
    parser.add_argument('--model', type=str, default='eng_rus.onnx')
    parser.add_argument('--threshold', type=float, default=0.45)
    parser.add_argument('--num-pairs', type=int, default=100)
    args = parser.parse_args()
    main(args)

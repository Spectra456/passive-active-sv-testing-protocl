import os
from pathlib import Path
from tqdm import tqdm
from verification import Verification  # Assuming your provided class is in verification.py
import whisper
import torchaudio
# Imported from user's provided scripts
from compute_EER import calculate_eer
from compute_min_dcf import ComputeErrorRates, ComputeMinDcf


#Modules
whisper_model = whisper.load_model("base")
def get_language(audio_path):
    # The whisper model is expected to be pre-loaded and passed in.
    result = whisper_model.transcribe(audio_path)
    language = result.get("language", "unknown")
    return language

def get_codec(audio_path):
    import ffmpeg
    probe = ffmpeg.probe(audio_path)
    audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    if audio_stream is None:
        return "Unknown"
    return audio_stream.get('codec_name', 'Unknown')

##Emotions:
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np

model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(model_id)

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label

def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)

    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs

def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]

    return predicted_label
###
###AGE
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender

def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict age and gender or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model1(y)
        if embeddings:
            y = y[0]
        else:
            y = torch.hstack([y[1], y[2]])

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y
import soundfile as sf

def proc_age_genderf(a_path):
    data, sr = torchaudio.load(a_path)
    res = process_func(data, sr)[0]
    return res[0],  np.argmax(res[1:4])
device = 'cpu'
model_name1 = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
processor = Wav2Vec2Processor.from_pretrained(model_name1)
model1 = AgeGenderModel.from_pretrained(model_name1)
###
#vad/snr

###
# Paths
audio_root = Path('/mnt/vox1o')
trials_file = '/mnt/veri_test.txt'
scores_output_file = 'scores.txt'

# Step 1: Load trials
pairs = []
labels = []
with open(trials_file, 'r') as f:
    for line in f:
        label, audio1, audio2 = line.strip().split()
        labels.append(int(label))
        pairs.append((audio1, audio2))

# Step 2: Initialize Verification class
verifier = Verification()

# Step 3: Calculate similarity scores
scores = []
files = []
with open(scores_output_file, 'w') as sf:
    for audio1, audio2 in tqdm(pairs, desc="Calculating similarity scores"):
        file1 = audio_root / audio1
        file2 = audio_root / audio2
        files.append(file1)
        files.append(file2)
files = set(files)


# with open(scores_output_file, 'w') as sf:
#     for file1 in tqdm(files):
#         lg1 = get_language(str(file1))
#         cd1 = get_codec(str(file1))
#         emo1 = predict_emotion(str(file1), model, feature_extractor, id2label)
#         age1, gender1 = proc_age_genderf(str(file1))
        
#         sf.write(f"{str(file1)} {lg1} {cd1} {emo1}  {age1} {gender1}\n")


with open("scores_vse.txt", 'w') as sf:
    for audio1, audio2 in tqdm(pairs, desc="Calculating similarity scores"):
        file1 = audio_root / audio1
        file2 = audio_root / audio2
        cos_score, _ = verifier.verify_speakers(str(file1), str(file2))
        lg1 = get_language(str(file1))
        lg2 =  get_language(str(file2))
        cd1 = get_codec(str(file1))
        cd2 = get_codec(str(file2))
        emo1 = predict_emotion(str(file1), model, feature_extractor, id2label)
        emo2 = predict_emotion(str(file2), model, feature_extractor, id2label)
        age1, gender1 = proc_age_genderf(str(file1))
        age2, gender2 = proc_age_genderf(str(file2))
        
        #sf.write(f"{cos_score[0]} {audio1} {audio2}\n")

        #scores.append(cos_score)
        sf.write(f"{cos_score[0]} {audio1} {audio2} {lg1} {lg2} {cd1} {cd2} {emo1} {emo2} {age1} {age2}  {gender1} {gender2}\n")

# # Step 4: Calculate EER
# eer, threshold_eer = calculate_eer(labels, [s[0] for s in scores], pos=1)
# print(f"Equal Error Rate (EER): {eer*100:.2f}% at threshold {threshold_eer:.4f}")

# # Step 5: Calculate minDCF
# fnrs, fprs, thresholds = ComputeErrorRates([s[0] for s in scores], labels)
# min_dcf, threshold_dcf = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1)
# print(f"Minimum Detection Cost Function (minDCF): {min_dcf:.4f} at threshold {threshold_dcf:.4f}")

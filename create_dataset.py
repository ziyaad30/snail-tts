import os
import time
import anyio
import random
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write as write_wav
import soundfile
import librosa
import tqdm
import argparse
import json
import librosa
import re
import torch
import torchaudio
from text.numbers import normalize_numbers
from num2words import num2words

print("Loading Tortoise...")
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# load phonemizer
from phonemizer.backend import EspeakBackend

if os.name == 'nt':
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    _ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'    # For Windows
    EspeakWrapper.set_library(_ESPEAK_LIBRARY)

def phoneme_text(text, lang="en-us"):
    backend = EspeakBackend(language=lang, preserve_punctuation=True, with_stress=False, punctuation_marks=';:,.!?¡¿—…"«»“”()', language_switch='remove-flags')
    text = backend.phonemize([text], strip=True)[0]
    return text.strip()
    
seed = 1234

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def create_edit_train_file(wav_path, speaker_id, text):
    
    text = text.strip()
    
    '''
    with open(f"./dataset/train.txt", 'a', encoding='utf-8') as f:
        f.write(wav_path + '|' + speaker_id + '|' + text + '\n')
        
    with open(f"./dataset/val.txt", 'a', encoding='utf-8') as f:
        f.write(wav_path + '|' + speaker_id + '|' + text + '\n')
    '''
    wav_path = Path(wav_path).stem
    
    with open(f"./dataset/hifigan_training.txt", 'a', encoding='utf-8') as f:
        f.write(wav_path + '|' + text + '\n')
        
    with open(f"./dataset/hifigan_validation.txt", 'a', encoding='utf-8') as f:
        f.write(wav_path + '|' + text + '\n')
    
def save_dataset(audio_array, text, speaker):
    
    SAMPLE_RATE = 22050
    
    # Create speaker folder
    # if not os.path.exists(f"dataset/{speaker}"):
    #    os.makedirs(f"dataset/{speaker}")
        
    if not os.path.exists(f"dataset/wavs"):
        os.makedirs(f"dataset/wavs")
    
    if speaker == 'xavier':
        speaker_id = 0
    else:
        speaker_id = 1
    
    # _, _, files = next(os.walk(f"dataset/{speaker}/"))
    _, _, files = next(os.walk(f"dataset/wavs/"))
    j = len(files)
    filename = f'audio_{j}.wav'

    # wav_path = f'./dataset/{speaker}/' + filename
    # processed_path = f'./dataset/{speaker}/' + f'processed_{speaker}_' + filename
    wav_path = f'./dataset/wavs/' + filename
    processed_path = f'./dataset/wavs/' + f'processed_{speaker}_' + filename
    write_wav(wav_path, 24000, audio_array)
    
    wav, sr = torchaudio.load(wav_path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
    wav = wav.mean(dim=0).unsqueeze(0)
    
    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(wav)
    torchaudio.save(processed_path, wav, SAMPLE_RATE, channels_first=True)
    
    # Try to trim silence
    audio, sr = librosa.load(processed_path, sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio)
    soundfile.write(processed_path, audio, SAMPLE_RATE, format='WAV')

    os.remove(wav_path)
    
    create_edit_train_file(processed_path, str(speaker_id), text)

tts = TextToSpeech()
if not os.path.exists(f"dataset"):
    os.makedirs(f"dataset")

print("Tortoise loaded.")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
    ('inc', 'incorporated'),
]]

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def lowercase(text):
    return text.lower()
    
def expand_numbers(text):
    return normalize_numbers(text)

def ordinal_dates(text):
    texts = re.findall("(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.)?(\w*)?(\.)?(\s*\d{0,2}\s*),(\s*\d{4})", text)
    for result in texts:
        rs = result[0] + result[2] + result[4]
        word_date = num2words(result[4], to='ordinal')
        rss = result[0] + result[2] + " " + word_date
        text = text.replace(rs, rss)
    return text

'''
def name_replacer(text):
    text = lowercase(text)
    text = expand_abbreviations(text)
    text = text.replace("ruggiero", "rujero")
    text = text.replace("bufalino", "buffalino")
    text = text.replace("genovese", "gɛnoviz")
    text = text.replace("apalachin", "apalaykin")
    text = text.replace("stefano", "stɛfɛnoʊ")
    text = text.replace("capo", "kapo")
    text = text.replace("decicco", "de chicco")
    text = text.replace("giuseppe", "jewseppe")
    text = text.replace("luciano", "lewcheeyano")
    text = text.replace("ANC", "A N C")
    text = text.replace("riina", "rina")
    text = text.replace("leggio", "lejio")
    text = text.replace("meyer", "maiyer")
    text = text.replace("salvatore", "salvator")
    text = text.replace("buscetta", "Bushetta")
    text = text.replace("stand-in", "standin")
    text = text.replace("u.s.", "u s")
    return text
'''

def generate_voice(text, speaker, auto_regressive_samples=2, diffusion_iterations=50, cond_free=True, temperature=2, top_p=0.0, length_penalty=8, repetition_penalty=4, breathing_room=20):
    set_all_seeds(seed)
    
    print(f'Speaker: {speaker}, Text: {text}')
    speaker = speaker.lower()
    
    voice_samples, conditioning_latents = load_voice(speaker)
    
    text = ordinal_dates(text)
    text = expand_numbers(text)

    talk_text = text
    
    # talk_text = name_replacer(text)
    print(f'Speaker: {speaker}, Text: {talk_text}')
    
    gen, dbg_state = tts.tts_with_preset(talk_text, 
                                    voice_samples=voice_samples, 
                                    conditioning_latents=conditioning_latents, 
                                    num_autoregressive_samples=auto_regressive_samples, 
                                    diffusion_iterations=diffusion_iterations, 
                                    cond_free=cond_free,
                                    temperature=float(temperature),
                                    top_p=float(top_p), length_penalty=length_penalty, repetition_penalty=float(repetition_penalty), breathing_room=breathing_room, use_deterministic_seed=seed, return_deterministic_state=True)
    gen = gen.squeeze(0).cpu()
    audio_array = np.array(gen.squeeze(0).cpu())
    save_dataset(audio_array, text, speaker)


def main(speakers_list):
    # speakers_list = [folder for folder in os.listdir("tortoise/voices")]
    speaker2id = {}
    
    for i, speaker in enumerate(speakers_list):
        print(f'[{i}] {speaker}')
        speaker2id[speaker] = i
        new_annos = []

        with open("./dataset_set.txt", 'r', encoding='utf-8') as f:
            text = f.readlines()
            new_annos += text
            
        for i, line in enumerate(new_annos):
            if i < 0:
                continue
            # path, txt = line.split("|")
            txt = line
            print(f'Retrieving line: {i}')
            generate_voice(txt, speaker)
            if i == 50:
                break
    '''
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
        
    # modify n_speakers
    hps['data']["n_speakers"] = len(speakers_list)
    # overwrite speaker names
    hps['speakers'] = speaker2id
    # modify training params
    hps['train']['log_interval'] = 10
    hps['train']['eval_interval'] = 50
    hps['train']['batch_size'] = 8
    hps['data']['training_files'] = "./dataset/test_train.txt"
    hps['data']['validation_files'] = "./dataset/test_val.txt"
    # save modified config
    with open("./configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
        json.dump(hps, f, indent=2)
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Voice Dataset with Tortoise')
    parser.add_argument("--voicelists", nargs="+", default=["Xavier"])
    args = parser.parse_args()
    main(args.voicelists)

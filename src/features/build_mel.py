import numpy as np
import librosa
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm


DATA_PATH = '../../data/processed/train_curated.pkl'
SAVE_PATH = '../../data/processed/mel/train_curated_mel.npy'

# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
# also N of FFT is defined with this value

print("INIT\n")

def wav2mel(wav_file, n_mels, hop_length, max_pad_len, normalize, scaling, padding_mode):

    wave, sr = librosa.load(wav_file, mono=True)
    
    if 0 < len(wave): # workaround: 0 length causes error
        wave, _ = librosa.effects.trim(wave) # trim, top_db=default(60)

    if normalize:
        wave = librosa.util.normalize(wave) # normalizing data before mfcc

    # making melspect from signal
    S = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=n_mels*20, hop_length=hop_length, n_mels=n_mels, fmax=sr//2)

    # scaling
    if scaling:
        S = preprocessing.scale(S, axis=1)

    if max_pad_len:
        if S.shape[1] > max_pad_len:
            S = S[:,:max_pad_len]
        else:
            pad_width = max_pad_len - S.shape[1]
            S = np.pad(S, pad_width=((0, 0), (0, pad_width)), mode=padding_mode)
    
    S = librosa.power_to_db(S)
    S = S.astype(np.float32)
   
    return S


def gen_mel(input_path=DATA_PATH, output_path=SAVE_PATH, n_mels=128, hop_length=512, max_pad_len=None, normalize=True, scaling=False, padding_mode='constant'):
    wavfiles = pd.read_pickle(input_path)['path']


    # Init Mel vectors
    mel_vectors = []

    for wavfile in tqdm(wavfiles):
        S = wav2mel('../' + wavfile, n_mels=n_mels, hop_length=hop_length, max_pad_len=max_pad_len, normalize=normalize, scaling=scaling, padding_mode=padding_mode)
        mel_vectors.append(S)


    np.save(output_path, mel_vectors)



print("runing....\n")

gen_mel(DATA_PATH, SAVE_PATH, n_mels=128, hop_length=512, max_pad_len=200)

print("END\n")

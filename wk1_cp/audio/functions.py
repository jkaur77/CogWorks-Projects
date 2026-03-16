# run this cell (maybe twice to get %matplotlib notebook to work)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from microphone import record_audio
from typing import Tuple
from typing import Union, Tuple, Sequence
from pathlib import Path
import os
import matplotlib.mlab as mlab
import librosa

#Reads in files from audio/data/
def getFiles() -> Sequence[str]:
    os.chdir("..")
    path = Path.cwd() / "audio"/ "data"
    music = os.listdir(path)
    array = [os.path.join(path,item) for item in music if item[-4:]==".mp3"]
    return array


#takes in an array of file paths and outputs librosa audio info
def toAudio(array: Sequence[str]) -> Sequence[Tuple[np.ndarray, int]]:
    audios = []
    for item in array:
        recorded, rate = librosa.load(item, sr =44100, mono = True)
        audios.append((recorded, rate))
    return audios
# `audios` is a numpy array of N audio samples

#audios = toAudio(getFiles())


#takes in an audio tuple(audio, sampling rate)
#outputs a tuple of log10spectrogram, frequencies, and times
def toSpectrogram(audio: Tuple[np.ndarray, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    recorded_audio = audio[0]
    sampling_rate = audio[1]
    S, freqs, times = mlab.specgram(
        recorded_audio,
        NFFT=4096,
        Fs=sampling_rate,
        window=mlab.window_hanning,
        noverlap=int(4096 / 2)
    )
    S = np.clip(S, 1e-20, None)
    S = np.log10(S)
    return ((S,freqs,times))


#Probably used as a training sample thing
#Takes in a list of audios and returns a list of spectrograms
def getAllSamples(audios: Sequence[Tuple[np.ndarray, int]]) ->Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ret = []
    for audio in audios:
        ret.append(toSpectrogram(audio))
    return ret
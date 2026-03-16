import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from typing import Tuple
import os 
import pickle
import librosa
from pathlib import Path 

from song_recog import peaks, fanout, database, match

filepaths = ["audio/data/champagne.mp3", "audio/data/closure.mp3", "audio/data/coney.mp3", "audio/data/cowboy.mp3",
             "audio/data/crime.mp3", "audio/data/damn.mp3", "audio/data/dorothea.mp3", "audio/data/evermore.mp3",
            "audio/data/gold.mp3", "audio/data/happiness.mp3", "audio/data/ivy.mp3", "audio/data/longstory.mp3",
            "audio/data/marjorie.mp3", "audio/data/tolerate.mp3", "audio/data/willow.mp3"]

database_fps = []

for path in filepaths :
    database_audio, sampling_rate = librosa.load(path, sr=44100, mono=True)

    spec, freqs, times = mlab.specgram(
        database_audio,
        NFFT=4096,
        Fs=sampling_rate,
        window=mlab.window_hanning,
        noverlap=int(4096 / 2)
    )

    fan = fanout(peaks(spec), 15)
    
    database_fps.append(fan)


np.save(database_fps, "")
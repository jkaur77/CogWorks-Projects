from audio.functions import getFiles, toAudio, toSpectrogram
from fp_module.fingerprints import fanout, peaks
from database.functions import add_to_database
import numpy as np

audios, names = toAudio(*getFiles())

matcher = {}
songs = {}
ct = 0
for audio, name in zip(audios, names):
    print(ct)
    sample = toSpectrogram(audio)
    pks = peaks(sample)
    fingerprints, times = fanout(pks, 15)
    matcher, songs = add_to_database(fingerprints, times, name, matcher, songs)

with open ("matcher.npy", "wb") as f:
    np.save(f, matcher)
with open ("songs.npy", "wb") as f:
    np.save(f, songs)




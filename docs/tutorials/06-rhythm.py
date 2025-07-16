# coding: utf-8
"""
===============
Rhythm analysis
===============

This section covers methods for estimating tempo and beat positions from audio signals.
"""

# %% 
# The previous section introduced methods for identifying the positions of note onsets.
# 

import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Audio, HTML

# Load an example audio file
y, sr = librosa.load(librosa.ex('choice'))
HTML(librosa.util.example_info('choice', html=True))

# %%
# 

Audio(data=y, rate=sr)

# %% 
# The starting point for rhythm analysis is often the onset strength envelope.
# We often use the envelope rather than detected onset positions to avoid
# error propagation from incorrect detections.
#
# First, let's visualize the audio signal, its spectrogram, and the onset
# strength envelope.

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
times = librosa.times_like(onset_env, sr=sr)
S = librosa.stft(y)

fig, ax = plt.subplots(nrows=3, sharex=True, height_ratios=[1, 1, 2])
ax[0].plot(times, onset_env, label='Onset strength', color='C1')
librosa.display.waveshow(y, sr=sr, ax=ax[1], label='Waveform')
img = librosa.display.specshow(S, vscale='dBFS', x_axis='time', y_axis='log', ax=ax[2])
librosa.display.colorbar_db(img, label='dBFS')
ax[0].legend()
ax[1].legend()
ax[0].label_outer()
ax[1].label_outer()


# %% 
# In the onset strength envelope, we can see 

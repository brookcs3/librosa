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
import matplotlib.animation as animation
from IPython.display import display, Audio, HTML

# Load an example audio file with a beat
y, sr = librosa.load(librosa.ex('sweetwaltz'), duration=20.0)
HTML(librosa.util.example_info('sweetwaltz', html=True))

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
# The onset strength envelope displayed above contains many peaks, which correspond to
# musical onsets.  In this example, the peaks are also regularly spaced in time.
# When we listen to this example this regular spacing is perceived as a pulse or beat.
# After computing the onset strength, the next step in rhythm analysis is often to 
# estimate the *tempo*, that is, the time spacing of the pulse.


# %%
# Estimating the tempo
# --------------------
# A typical method for estimating tempo from the onset strength envelope is to compute
# the *autocorrelation* of the envelope, which measures how similar the envelope is to
# a delayed copy of itself.
# The animation below illustrates this process, limited to the first 5 seconds of the
# audio signal.

fig, ax = plt.subplots(nrows=2)
times = librosa.times_like(onset_env, sr=sr)
ax[0].plot(times, onset_env, label='Onset strength', color='C1')
onset_delayed = ax[0].plot(times, onset_env, label='Delayed onset strength', color='C2', linestyle='--')[0]

ax[0].set(xlim=(0, 5), xlabel='Time (s)')
ax[0].legend()

xcorr = np.correlate(onset_env, onset_env, mode='same')
corrplot = ax[1].plot(times, xcorr, label='Autocorrelation', color='C3')[0]
ax[1].legend()
ax[1].set(xlim=(0, 5), xlabel='Lag (s)')

def _update(num):
    """Update the plot for each frame."""
    # Update the delayed onset strength
    # Show the onset strength envelope delayed by `num` frames
    onset_delayed.set_xdata(times + num * times[1])
    # Show the autocorrelation up to the current amount of lag
    corrplot.set_xdata(times[:num])
    corrplot.set_ydata(xcorr[:num])
    return (onset_delayed, corrplot)

ani = animation.FuncAnimation(fig,
                              func=_update,
                              frames=np.ceil(5 / times[1]).astype(int),
                              interval=50,
                              blit=True)

# %%
# The tempo is determined by the position of the first prominent peak in the autocorrelation,
# not including the peak at lag of zero.
# In this example, the first peak occurs at around 0.4 seconds.
# Treated as a period, this amount of lag corresponds to a frequency of 2.5 Hz.
# More often, tempo is expressed in units of beats per minute (BPM), rather than cycles per second (Hz),
# and we can convert between the two by multiplying by 60.
# This calculation results in a tempo of 150 BPM.
#
# Of course, librosa provides a convenient function to compute the tempo directly,
# either from the signal `y` or from a pre-computed onset strength envelope.

tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
print(f"Estimated tempo: {tempo[0]:.2f} BPM")

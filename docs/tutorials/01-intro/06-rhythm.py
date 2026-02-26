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
from IPython.display import Audio, HTML

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

fig, ax = plt.subplots(nrows=3, sharex=True, height_ratios=(1, 1, 2))
ax[0].plot(times, onset_env, label='Onset strength', color='C1')
librosa.display.waveshow(y, sr=sr, ax=ax[1], label='Waveform')
img = librosa.display.specshow(S, sr=sr, vscale='dBFS', x_axis='time', y_axis='log', ax=ax[2])
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

# %%
# Time-varying tempo
# ------------------
# The analysis above is useful for estimating a single tempo that describes the
# entire recording.  However, in music we often have changes in tempo throughout
# the piece.  For example, the following piece changes tempo dramatically several
# times in a relatively short time-span.

y, sr = librosa.load(librosa.ex('brahms'))
HTML(librosa.util.example_info('brahms', html=True))

# %%
# 
Audio(data=y, rate=sr)

# %%
# Instead of estimating a single global tempo, we can instead estimate a time-varying
# tempo by disabling aggregation.  

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
times = librosa.times_like(onset_env, sr=sr)
tempi = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
print(f"Estimated tempi: {tempi}")

# %%
# Behind the scenes, the tempo estimator will compute auto-correlation on
# short fragments of the onset envelope, and estimate an independent tempo
# from each fragment.
# If we collect these autocorrelation results together, we can visualize
# the data as a *tempogram*, just like we did previously for *spectrograms*,
# and plot the estimated tempo over top.
#
tgram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
fig, ax = plt.subplots()
librosa.display.specshow(tgram, x_axis='time', y_axis='tempo', sr=sr, ax=ax)
ax.plot(times, tempi, label='Estimated tempo', color='lime', linewidth=4)
ax.legend(loc='upper right')

# %%
# From tempo to beats
# -------------------
# So far, we've seen how to estimate tempo from the onset strength envelope.
# This tells us roughly the speed at which beats (typically quarter-notes, `♩`)
# occur, but it does not identify *where* they occur: that is the job of a *beat tracker*.
#
# The main beat tracking algorithm implemented by librosa is based on the method of 
# 
#
# It essentially works as follows:
# 1. Estimate the tempo of the recording.  This can be either static or dynamic, as described
# above.
# 2. Identify peaks in the onset envelope which are approximately spaced by the tempo.
# 3. Globally optimize the selection of onset envelope peaks subject to tempo constraints.
#
# If a tempo is not provided to the tracker, it will be estimated from the signal directly.
# Either way, the tracker returns both the tempo estimate and the identified beat positions.
# Like the onset detector, we can select the units applied to the beat tracker's estimates:
# frame indices (default), sample indices, or time (seconds).

tempo_global, beats_global = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr,
                                                     units='time')
print(f"Estimated tempo: {tempo_global}")
print(f"Estimated beats: {beats_global}")

# %%
# We can play back the estimated beats with a *click track*:

beats_global_click = librosa.clicks(times=beats_global, sr=sr, length=len(y))
Audio(data=librosa.to_stereo(left=y, right=beats_global_click), rate=sr)

# %%
# And we can also visualize the results by plotting over the onset envelope.
# This can be done directly with matplotlib, or using the display helpers
# included in the `mir_eval` package.
# For this example, we'll use `mir_eval`, and zoom in on the middle ten seconds of the
# recording.

import mir_eval.display

fig, ax = plt.subplots()
ax.plot(times, onset_env, label='Onset envelope')
mir_eval.display.events(beats_global, ax=ax, label='Beats', color='C1')
ax.legend(loc='upper right')
ax.set(xlim=[10, 20])

# %%
# As we can see from the plot, and more directly, by listening to the click track,
# the beat tracker is not doing well in this region of the recording.
# This is due to the dramatic change in tempo that occurs around time=14s.
# Since the tracker assumes a static tempo by default, it will fail to identify onset peaks
# with appropriate time spacing when the tempo changes significantly.
# 
# However, we can also use a dynamic tempo estimate to give it a better chance in recordings
# like this.  For this, we will use the time-varying tempo estimate `tempi` as computed above.

tempo_dynamic, beats_dynamic = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr,
                                                       bpm=tempi,
                                                       units='time')
print(f"Estimated beats: {beats_dynamic}")
beats_dynamic_click = librosa.clicks(times=beats_dynamic, sr=sr, length=len(y))
Audio(data=librosa.to_stereo(left=y, right=beats_dynamic_click), rate=sr)

# %%
#

fig, ax = plt.subplots()
ax.plot(times, onset_env, label='Onset envelope')
mir_eval.display.events(beats_global, ax=ax, label='Beats (global)', color='C1')
mir_eval.display.events(beats_dynamic, ax=ax, label='Beats (dynamic)',
                        color='C2', linestyle='--')
ax.legend(loc='upper right')
ax.set(xlim=[10, 20])


# coding: utf-8
"""
================
Tempo and rhythm
================

This section introduces tools for analyzing the temporal aspects of music signals,
including note onset detection, tempo estimation, and beat tracking.
"""

# %%
# Onsets
# ------
# As a starting point, we'll first look at the problem of identifying when each 
# musical event (e.g., a note sounding) occurs in the input signal.
# We will focus on identifying the beginning of the note events, known as *onsets*,
# as this is often the first stage of processing for more sophisticated analyses
# such as tempo estimation or beat tracking, which we will see below.
# 
# First, we will load in a simple monophonic example recording, and display its waveform
# and spectrogram.

import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio

# Load the audio
y, sr = librosa.load(librosa.ex('trumpet'))

# Compute the STFT
S = librosa.stft(y)

# Generate a plot of waveform and spectrogram
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw=dict(height_ratios=(1, 4)))
librosa.display.waveshow(y=y, sr=sr, ax=ax[0], label='Waveform')
librosa.display.specshow(S, vscale='dBFS', x_axis='time', y_axis='log', sr=sr)
ax[0].label_outer()
ax[0].legend()

# And an audio playback object so we can listen to it
Audio(data=y, rate=sr)

# %%
# Try listening to the example above while following the visual display.
# With some practice, you should be able to draw a correspondence between
# when each note is sounding, visually detectable changes in the spectrogram
# and waveform.
#
# The spans of time where the acoustic content is stable, e.g., the final 
# sustained F note beginning at around 2.5 seconds, there are no abrupt
# changes in the spectrogram from one vertical slice to the next.
# Put another way, there are sustained horizontal patterns which correspond
# to sustained tones.
#
# When we are looking for new note onsets, we are interested in exactly the
# cases where the spectrogram is *not* constant in time.
# There are many ways to formalize this idea, but a core principle is the idea
# of detecting change or *novelty* by comparing each time step of a spectrogram
# to the preceding time, i.e., `S[:, t]  - S[:, t-1]`.
# 
# Typically we are not so much interested in the complex spectrogram values
# as the magnitudes, which are proportional to energy (as displayed above).
# Additionally, working in log magnitudes (or decibels) provides some robustness
# to overall changes in loudness.
#
# Finally, we are generally more interested only in places where energy is 
# *increasing*, since decreasing energy does not generally indicate the onset
# of a new event.
# 
# We can put these ideas together as follows.

# Map the magnitude abs(S) to decibels
logS = librosa.amplitude_to_db(np.abs(S), ref=np.max)

# Compute the first-order difference logS[:, t] - logS[:, t-1]
# along the time direction.
# We'll pad the differencing operation with the first column of
# logS to prevent a spike in the first step
diffS = np.diff(logS, axis=-1, prepend=logS[:, :1])

# We'll threshold out any negative values as these correspond to
# falling energy
diffS_thresh = np.maximum(diffS, 0)

# Visualize the results
fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
i1 = librosa.display.specshow(S, vscale='dBFS', x_axis='time', y_axis='log', ax=ax[0], sr=sr)
i2 = librosa.display.specshow(diffS, x_axis='time', y_axis='log', ax=ax[1], sr=sr)
i3 = librosa.display.specshow(diffS_thresh, x_axis='time', y_axis='log', ax=ax[2], sr=sr, cmap='Reds')

fig.colorbar(i1, ax=ax[0])
fig.colorbar(i2, ax=ax[1])
fig.colorbar(i3, ax=ax[2])
ax[0].label_outer()
ax[1].label_outer()
ax[0].set(ylabel='STFT')
ax[1].set(ylabel='Difference')
ax[2].set(ylabel='Thresholded diff')


# %%
# In the middle plot above, we can see that constant regions map to a neutral
# color (gray), while regions where the difference is positive (increasing energy)
# are encoded in red, while negative differences (decreasing energy) are encoded
# in blue.
# 
# The bottom plot discards the negative regions, retaining only time-frequency
# positions where energy is increasing.
#
# Finally, we are usually not interested in changes at each individual frequency,
# but rather the aggregated change across all frequencies at each time.
# A simple way to aggregate is by summing across frequencies, resulting in 
# what is usually called an *onset strength envelope* or a *novelty curve*.
#

# Sum across the frequency dimension
onset_env = np.sum(diffS_thresh, axis=0)

# Plot the waveform, spectrogram, and onset envelope together

fig, ax = plt.subplots(nrows=3, sharex=True, gridspec_kw=dict(height_ratios=(1,1,4)))
librosa.display.waveshow(y=y, sr=sr, ax=ax[0], label='Waveform')
librosa.display.specshow(S, vscale='dBFS', x_axis='time', y_axis='log', ax=ax[2], sr=sr)
times = librosa.times_like(onset_env, sr=sr)
ax[1].plot(times, onset_env, label='Onset envelope', color='r')
ax[1].legend()
ax[0].legend()
ax[0].label_outer()
ax[1].label_outer()

# %%
# So far, we've computed the onset strength envelope (middle plot above, red) manually.
# This is so common of an operation, however, that librosa provides a function that implements
# this, along with several other variations on the core idea.

# Equivalent to the above
onset_env = librosa.onset.onset_strength(S=logS)

# %%
# It's worth emphasizing here that the onset strength envelope does not make decisions about
# whether or not a new event occurs at each time.
# Rather, it should be taken as a soft representation that something new *might* be happening.
# Still, onset strength envelopes can be useful objects on their own, and often form an intermediate
# signal representation that is passed into a subsequent stage of processing.

# %%
# Onset detection
# ---------------
# To actually detect onsets, we need to make a decision about
# whether each frame in the onset strength envelope contains a new event or not.
#
# A simple heuristic is to simply take the peak positions of the onset strength envelope,m
# i.e., local maxima where `o[t] > o[t-1]` and `o[t] > o[t+1]`.
# This codifies the intuition that peaks of the onset envelope correspond to
# the steepest increase of energy, and should therefore align with the perception
# of a new event.
# This can be implemented simply using the `librosa.util.localmax` utility function:

onset_peaks = librosa.util.localmax(onset_env)

fig, ax = plt.subplots(nrows=2, sharex=True)

librosa.display.waveshow(y=y, sr=sr, ax=ax[0], label='Waveform')
ax[0].legend()
ax[1].plot(times, onset_env, label='Onset envelope', color='r')
ax[1].scatter(times[onset_peaks], onset_env[onset_peaks], marker='x', color='k', label='Peaks')
ax[1].legend()

# %%
# As illustrated above, simply identifying local maxima leads to a highly sensitive
# detector that produces far more events than actually occur in the signal.
# This can be attributed to two principal causes: 1) the onset strength envelope is 
# somewhat noisy, and 2) the magnitude of the envelope is not considered at all.
# Additionally, direct peak picking does not account for proximity effects,
# e.g., that it is unusual (or imperceptible) for two onsets to occur within
# a very short amount of time (e.g. within 30ms).
#
# Librosa therefore implements a heuristic peak-picking algorithm which seeks
# to select peaks which are sufficiently separated and sufficiently high in
# value relative to their surroundings.

onset_detect = librosa.onset.onset_detect(onset_envelope=onset_env)

fig, ax = plt.subplots(nrows=2, sharex=True)

librosa.display.waveshow(y=y, sr=sr, ax=ax[0], label='Waveform')
ax[0].legend()
ax[1].plot(times, onset_env, label='Onset envelope', color='r')
ax[1].scatter(times[onset_peaks], onset_env[onset_peaks], marker='x', color='k', label='Localmax Peaks')
ax[1].scatter(times[onset_detect], onset_env[onset_detect], marker='o', color='b', label='onset_detect')
ax[1].legend()

# %%
# We can also sonify these detected events to hear how they align with the onset of new notes.

# Generate a click track from the detected frames 
# and match the length to the original input signal
clicks = librosa.clicks(frames=onset_detect, length=len(y), sr=sr)

# Sonify the result
Audio(data=y + clicks, rate=sr)

#    - click track
# Tempograms and tempo estimation
# Beat tracking
#

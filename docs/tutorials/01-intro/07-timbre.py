# coding: utf-8
"""
===========================
Timbre and spectral content
===========================

This section introduces common tools for representing the timbral or textural
content of audio signals.
"""

# %%
# What is timbre?
# ---------------
#
# Earlier sections introduced techniques for representing pitch, harmony, and timing elements
# of music.
# The major remaining aspect of music that we haven't seen yet is *timbre*. 
# Unlike the previous sections, timbre doesn't have a precise definition on its own, and is
# typically defined in contrast to pitch and rhythm: what makes two different instruments 
# playing the same notes sound different?
#
# To demonstrate timbral analysis, we'll need recordings that have differing content in timbre
# while pitch and timing are held constant.
# This is luckily provided to us by the `Chorale Bricks
# <https://audiolabs-erlangen.de/resources/MIR/2025-ChoraleBricks>`_ dataset, from which we
# have selected an excerpt including Alto Saxophone, Clarinet, and Trumpet (all playing in
# unison).
# We augmented this example with two more instruments: a midi Piano, and a synthsized string
# instrument.
# Each instrument's isolated recording is encoded in a separate channel of the same audio file.
#
# For context, we can listen to each instrument in isolation:



import numpy as np
import matplotlib.pyplot as plt
import librosa
from IPython.display import Audio, HTML

y, sr = librosa.load("drese+midi.ogg", mono=False)

instruments = ["Alto Sax", "Clarinet", "Trumpet", "Piano", "Synth string"]
n_instruments = len(instruments)

html_content = ""
for i, instrument in enumerate(instruments):
    audio_widget = Audio(y[i], rate=sr)._repr_html_()
    html_content += f"<div><strong>{instrument}:</strong><br>{audio_widget}</div><br>"

# Must be the final statement evaluated in this script block
HTML(html_content)

# %%
# which we can visualize as follows:

fig, ax = plt.subplots(nrows=n_instruments, sharex=True, layout="constrained",
                       figsize=(8, 3))
librosa.display.multiplot("waveshow", y, sr=sr, invert=True, labels=instruments, axes=ax)
fig.legend(loc="outside right center")

# %%
# Some differences in dynamics might be apparent already, but probably not much else.
# To get a better sense of how the instruments differ from each other, we can look at their
# spectrograms.

stft = librosa.stft(y)
fig, ax = plt.subplots(nrows=n_instruments, sharex=True, sharey=True,
                       figsize=(10, 6), gridspec_kw=dict(hspace=0.05, wspace=0.05))
imgs = librosa.display.multiplot("specshow", stft, vscale='dBFS', sr=sr,
                                 x_axis="time", y_axis="log", axes=ax)
for i, inst in enumerate(instruments):
    ax[i].set(ylabel=inst)
librosa.display.colorbar_db(imgs[0], ax=ax, pad=0.01)

# %%
# We can now start to observe some specific differences between the instruments,
# especially if we focus on how the energy is distributed as different harmonics
# of the fundamental frequency at any given time.
# 
# Let's zoom in on a particular note. The he spectra within the slice we're selecting are mostly
# stationary, so we can look at the average over time within this slice to more clearly see
# how the energy is distributed across the harmonics of the fundamental frequency.

fig, ax = plt.subplots(nrows=5, ncols=2, sharex='col', sharey=True,
                       figsize=(10, 8), gridspec_kw=dict(hspace=0.05, wspace=0.05,
                                                        width_ratios=[3, 1]))
imgs = librosa.display.multiplot("specshow", stft, vscale='dBFS', sr=sr,
                                 x_axis="time", y_axis="log", axes=ax[:, 0])
for i, inst in enumerate(instruments):
    ax[i, 0].set(ylabel=inst)

time_slice = (7.9, 10.2)  # seconds
ax[0, 0].set(xlim=time_slice)  # Zoom in to the time slice

# Convert time in seconds to frame indices
frames = librosa.time_to_frames(time_slice, sr=sr)
# Aggregate over time
avg_spec = np.mean(np.abs(stft[..., frames[0]:frames[1]]), axis=-1)
# Get the frequency range for our plot
freqs = librosa.fft_frequencies(sr=sr)
for i in range(n_instruments):
    # Scale to dB and plot on the same axes as the spectrogram
    # We'll clip to 60dB below peak to focus on the most prominent parts of the spectrum
    ax[i, 1].plot(librosa.amplitude_to_db(avg_spec[i], ref=np.max, top_db=60),
                  freqs, color=f"C{i}")

ax[0, 0].set(title="Spectrogram")
ax[0, 1].set(title="Average spectrum")
ax[-1, 1].set(xlabel="dB")

# %%
# All of our instruments have energy peaks at the same set of frequencies, but the relative
# amount of energy differs among them.  This contributes to the different timbral
# characteristics of each instrument, and is what we will be exploiting to distinguish between
# them.


# %%
# Similarity
# ----------
# To understand how well our representations of timbre work to separate different instruments,
# we'll use the `UMAP <https://umap-learn.readthedocs.io/en/latest/>`_ dimensionality reduction
# method to map the data down to two dimensions for visualization.
# 
# Because UMAP uses a sample of data to estimate the dimensionality reduction, it will be
# helpful to have some held-out data to illustrate how well the method generalizes to
# previously unseen data.  We'll accomplish this by splitting the signal in time, using
# the first 10 seconds to estimate the UMAP projection, and another 10 second slice to
# evaluate how well the projection generalizes to new data.

import umap

y_fit, sr = librosa.load("drese+midi.ogg", mono=False, duration=10)
y_test, _ = librosa.load("drese+midi.ogg", mono=False, offset=-10)

# %%
# Now let's first see how well the basic short-time Fourier transform works, using dB scaling
# on the amplitudes.
#
# We'll be reusing this plotting code below, so we'll put it in a function to avoid repetition.
#

def minmax_normalize(x, axis=-1):
    return (x - np.min(x, axis=axis, keepdims=True)) / (np.max(x, axis=axis, keepdims=True) - np.min(x, axis=axis, keepdims=True))

def plot_umap(data_fit, data_test, alpha_fit, alpha_test, ax):
    # Fixing the random state and number of jobs to ensure reproducibility
    reducer = umap.UMAP(random_state=5, n_jobs=1)
    # We need to reshape the data so that each time frame is a sample, and the frequency bins are
    # features.
    embed_fit = reducer.fit_transform(data_fit.transpose(0, 2, 1).reshape(-1, data_fit.shape[1]))
    embed_test = reducer.transform(data_test.transpose(0, 2, 1).reshape(-1, data_test.shape[1]))

    n_fit = data_fit.shape[-1]
    n_test = data_test.shape[-1]

    # min-max normalize the alpha values for fit and test
    # Then reshape so we can use them in the scatter plot
    alpha_fit = minmax_normalize(alpha_fit).transpose(0, 2, 1).reshape(-1)
    alpha_test = minmax_normalize(alpha_test).transpose(0, 2, 1).reshape(-1)

    # We'll use a subtle stroke effect to help the individual data points stand out
    hl = librosa.display.highlight(ax=ax, alpha=0.5, linewidth=.5)

    # Now plot the results, coloring by instrument label:
    for i in range(n_instruments):
        idx_fit = slice(i * n_fit, (i+1)*n_fit)
        idx_test = slice(i * n_test, (i+1)*n_test)
        ax.scatter(embed_fit[idx_fit, 0], embed_fit[idx_fit, 1],
                   label=f"{instruments[i]} (fit)", color=f"C{i}", marker=".", s=15,
                   alpha=alpha_fit[idx_fit]**2,
                   path_effects=hl, zorder=10)
        ax.scatter(embed_test[idx_test, 0], embed_test[idx_test, 1],
                   label=f"{instruments[i]} (test)", color=f"C{i}", marker="o", s=25,
                   alpha=alpha_test[idx_test]**2,
                   path_effects=hl, zorder=5)
    ax.set(xticks=[], yticks=[]) # X and Y axes are arbitrary units, so we can hide the ticks
    # Fix the alpha channels in the legend for legibility
    fig = ax.get_figure()
    leg = fig.legend(loc='outside right center')
    for lh in leg.legend_handles:
        lh.set_alpha(np.ones_like(lh.get_alpha()))

# %%
# Now we can compute the STFT magnitudes and plot them 
stft_fit = librosa.amplitude_to_db(np.abs(librosa.stft(y_fit)), ref=np.max)
stft_test = librosa.amplitude_to_db(np.abs(librosa.stft(y_test)), ref=np.max)

# %%
# Let's first visualize the fit and test data:

fig, ax = plt.subplots(nrows=n_instruments, ncols=2, sharex='col', sharey=True,
                       figsize=(10, 6), gridspec_kw=dict(hspace=0.05, wspace=0.05))
imgs = librosa.display.multiplot("specshow", stft_fit,  sr=sr,
                                 x_axis="time", y_axis="log", axes=ax[:, 0])
librosa.display.multiplot("specshow", stft_test, sr=sr,
                          x_axis="time", y_axis="log", axes=ax[:, 1])
ax[0, 0].set(title="STFT (fit)")
ax[0, 1].set(title="STFT (test)")
for i, inst in enumerate(instruments):
    ax[i, 0].set(ylabel=inst)
librosa.display.colorbar_db(imgs[0], ax=ax, pad=0.01)


# Now plot the results, coloring by instrument label.
# We'll use transparency to encode amplitude, so that quiet parts of the signal
# do not contribute visual clutter
alpha_fit = np.mean(stft_fit, axis=1, keepdims=True)
alpha_test = np.mean(stft_test, axis=1, keepdims=True)

fig, ax = plt.subplots(layout='constrained')
plot_umap(stft_fit, stft_test, alpha_fit, alpha_test, ax)
ax.set(title='STFT magnitude UMAP projection')

# %%
# In the plot above, each data point corresponds to a single frame of a recording.  If two
# frames are close in the plot, they have similar spectral magnitudes.
# If our representation is working well to separate instruments, then we should see distinct
# clusters of points corresponding to each instrument (coded by color).
# Indeed, some but not all of our instruments are well separated in the STFT magnitude space.
#
# There is however quite a bit of crowding in some parts of the space: this is because the STFT
# similarity tends to be dominated by agreement in pitch content (e.g., fundamental frequency)
# rather than the overall shape of the spectrum independent of f0.
# That is, two distinct instruments playing the same note will likely be close to each other in
# this representation, making it a poor choice for representing timbre independent of pitch.

# %%
# Mel spectra
# -----------
# A first step toward a more timbre-focused representation is to transform the frequency range
# into something that better aligns with human perception of frequency.
# For this, we'll use the Mel scale, which you can think of as an approximately logarithmic
# transformation of the frequency axis, coupled with a dimensionality reduction to group nearby
# frequencies together.
#
# Mel spectrograms are computed in librosa by ``librosa.feature.melspectrogram``:

mel_fit = librosa.feature.melspectrogram(y=y_fit, sr=sr)
mel_test = librosa.feature.melspectrogram(y=y_test, sr=sr)

# Mel spectra, by default, are computed in power units instead of amplitude, so we'll use 
# the power_to_db converter instead here.

mel_fit_db = librosa.power_to_db(mel_fit, ref=np.max)
mel_test_db = librosa.power_to_db(mel_test, ref=np.max)

# %%
# Mel spectrograms look pretty much like STFT's, but with vastly fewer frequency bins:

print(f"STFT shape (instruments, frequencies, frames): {stft_fit.shape}")
print(f"Mel spectrogram shape: {mel_fit_db.shape}")

fig, ax = plt.subplots(nrows=n_instruments, ncols=2, sharex='col', sharey=True,
                       figsize=(10, 6), gridspec_kw=dict(hspace=0.05, wspace=0.05))
imgs = librosa.display.multiplot("specshow", mel_fit, vscale='dBFS[power]', sr=sr,
                                 x_axis="time", y_axis="mel", axes=ax[:, 0])
librosa.display.multiplot("specshow", mel_test, vscale='dBFS[power]', sr=sr,
                          x_axis="time", y_axis="mel", axes=ax[:, 1])
ax[0, 0].set(title="Mel spectrogram (fit)")
ax[0, 1].set(title="Mel spectrogram (test)")
for i, inst in enumerate(instruments):
    ax[i, 0].set(ylabel=inst)
librosa.display.colorbar_db(imgs[0], ax=ax, pad=0.01)

# %%
# And plot the UMAP embeddings from Mel spectra:

fig, ax = plt.subplots(layout='constrained')
plot_umap(mel_fit_db, mel_test_db, alpha_fit, alpha_test, ax)
ax.set(title='Mel spectrogram UMAP projection')

# %%
# We now have a little more separation between instruments here, but still quite a bit of
# overlap.  This is because the mel spectrogram is still sensitive to pitch; it's just less
# sensitive than the STFT because the mel filter bank groups together nearby frequencies.



# %% 
# Mel frequency cepstral coefficients (MFCCs)
# -------------------------------------------
# To further invariance to the exact fundamental frequency of the signal,
# *mel frequency cepstral coefficients* (MFCCs) are commonly used.
# Essentially, these work by projecting each dB-scaled mel spectrum (frame) onto a set of basis
# functions which capture the overall shape of the spectrum.
# These are computed by ``librosa.feature.mfcc``:

mfcc_fit = librosa.feature.mfcc(y=y_fit, sr=sr)
mfcc_test = librosa.feature.mfcc(y=y_test, sr=sr)

# %%
# And we can visualize them in a similar fashion to spectrograms, though the interpretation of
# values and vertical axes are quite different here:

fig, ax = plt.subplots(nrows=n_instruments, ncols=2, sharex='col', sharey=True,
                          figsize=(10, 6), gridspec_kw=dict(hspace=0.05, wspace=0.05))
imgs = librosa.display.multiplot("specshow", mfcc_fit, sr=sr,
                                 x_axis="time", axes=ax[:, 0])
librosa.display.multiplot("specshow", mfcc_test, sr=sr,
                          x_axis="time",  axes=ax[:, 1])
ax[0, 0].set(title="MFCC (fit)")
ax[0, 1].set(title="MFCC (test)")
for i, inst in enumerate(instruments):
    ax[i, 0].set(ylabel=inst)
librosa.display.colorbar_db(imgs[0], ax=ax, pad=0.01)

# %%
# A few things to note here:
#
#   1. The vertical axis does not correspond to frequency, but rather the response of each basis function applied to the spectrum.
#   2. The values here can be positive or negative, and the sign matters.
#   3. The scale of values tends to be much larger for lower coefficients, and smaller for
#      higher coefficients.
#
# Without going into the details of the calculation, the first coefficient (bottom row of
# each subplot above) captures the overall amplitude over time.
# The second coefficient captures the energy balance between low and high frequencies;
# the third coefficient captures the balance between the middle frequencies and the extremes
# (high and low together); and so on.
#
# The basic idea here is to describe the general shape of the spectrum, not the fine details.
# For this reason, MFCCs are typically restricted to a small number of coefficients (e.g., 13
# or 20), which is much smaller than the number of mel bins or STFT bins.
print(f"MFCC shape: {mfcc_fit.shape}")
print(f"Mel spectrogram shape: {mel_fit_db.shape}")
print(f"STFT shape: {stft_fit.shape}")

# %%
# Let's see how well the MFCCs do at separating our instruments:
#

# sphinx_gallery_thumbnail_number = 9
fig, ax = plt.subplots(layout='constrained')
plot_umap(mfcc_fit, mfcc_test, alpha_fit, alpha_test, ax)
ax.set(title='MFCC UMAP projection')


# %%
# While the MFCC representation is definitely not perfect, it is doing a better job of
# reducing overlap between instrument clusters.

# %%
# Quantitative evaluation
# -----------------------
# To get a more quantitative sense of how well these different representations separate our
# instruments, we can use a simple k-nearest neighbor classifier to predict the instrument
# label of each frame in the test set based on the closest frame in the fit set, and then
# compute a classification report based on the true and predicted labels.
# Note that this classification is performed in the original feature space (e.g., STFT, Mel, or
# MFCC), not the UMAP space, so it is not directly related to the UMAP visualizations above.
# Rather, it is meant as a quantitative check to ensure that the visualizations are not
# misleading us.

import sklearn.neighbors
import sklearn.metrics
import pandas as pd

# Limit dataframe precision to 2 decimal places for better readability
pd.set_option('display.precision', 2)

def knn_eval(data_fit, data_test):
    n_fit = data_fit.shape[-1]
    n_test = data_test.shape[-1]

    # Create labels for each instrument's frames
    labels_fit = np.repeat(np.arange(n_instruments), n_fit)
    labels_test = np.repeat(np.arange(n_instruments), n_test)

    # Reshape the data so that each time frame is a sample, and the frequency bins are features.
    X_fit = data_fit.transpose(0, 2, 1).reshape(-1, data_fit.shape[1])
    X_test = data_test.transpose(0, 2, 1).reshape(-1, data_test.shape[1])

    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_fit, labels_fit)
    pred_labels = knn.predict(X_test)

    report =  sklearn.metrics.classification_report(labels_test, pred_labels,
                                                    target_names=instruments,
                                                    output_dict=True)
    df = pd.DataFrame(report).transpose()
    df['support'] = df['support'].astype(int)
    return df

# %%
# For each of our representations, we can compute a classification report to summarize
# for each instrument:
#
#  - Precision: the proportion of frames predicted to be a given instrument that are actually that instrument.
#  - Recall: the proportion of frames of a given instrument that are correctly predicted as that instrument.
#  - F1-score: the harmonic mean of precision and recall, which provides a single metric that balances both precision and recall.
#
# as well as the overall accuracy across all instruments.  The higher these values are, the
# better the representation is at separating the different instruments.
#
# **STFT evaluation**
knn_eval(stft_fit, stft_test)

# %%
# **Mel evaluation**
knn_eval(mel_fit_db, mel_test_db)

# %%
# **MFCC evaluation**
knn_eval(mfcc_fit, mfcc_test)

# %%
# Summary
# -------
# It is worth noting that although MFCCs have historically been used for timbre analysis and
# more general, high-level classification of audio signals, they are by now very far from
# "state of the art".
# That said, it is still instructive to see how these different representations work to
# eliminate unwanted sensitivity.
#
# It is also worth noting that there are many extensions that could be implemented on top of
# this basic MFCC pipeline: for example, we could eliminate the 0th coefficient to remove
# sensitivity to overall amplitude, concatenate first- and second-order time differences to
# capture temporal dynamics (see ``feature.delta``), use more powerful classification methods,
# and so on.

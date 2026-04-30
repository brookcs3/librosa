# coding: utf-8
"""
==============
Music notation
==============

This section introduces tools for working with various music notational systems, and
converting between them.
"""


# %%
# Pitches, frequency, and MIDI numbers
# ------------------------------------
# In the earlier section on :ref:`tutorial-f0` we saw an example of 
# fundamental frequency (f₀) estimation, which produces estimates in units of
# Hz (cycles per second).
# We then showed how to use unit conversion to translate these
# numerical estimates into human-readable pitch classes.
# It's worth pausing at this point to better understand what
# these different representations do, and how conversion between
# them is implemented.
#
# It helps to first clarify some definitions:
#
# - **Frequency** is a physical property, measured in Hertz (cycles per
#   second, abbreviated Hz).  This is something that we can observe directly in a
#   recorded signal.
#
# - **Pitch** is a perceptual concept relating frequency to what we understand as
#   musical tones.  In librosa, we adopt the `Scientific Pitch Notation (SPN)
#   <https://en.wikipedia.org/wiki/Scientific_pitch_notation>`_ standard for representing pitch in
#   Western notation (*C*, *C♯*, *D♭*, etc.).  SPN represents pitches as a combination of note name (*C*,
#   *D*, *E*, etc.), accidentals (*♯*, *♭*, *♮*, etc.), and an octave number.  Librosa extends this
#   slightly to additionally encode cent deviations from the underlying equal temperament grid.
#
#   .. admonition:: Example
#
#       15¢ above middle C can be represented as *C4+15*, which is equivalent to 263.902 Hz.  
#       The same frequency could be represented equivalently as either *C♯4-85* or *D♭4-85*.
#
#       SPN assumes 12-tone equal temperament (12TET) and A440 tuning.
#
# - **MIDI** (Musical Instrument Digital Interface) standard assigns integer values 0-127 to pitches
#   following the conventions described above (12TET, A440).  
#   A MIDI note number *n* can be converted to a frequency via the equation 
#   :math:`f = 440 · 2^{(n-69)/12}`.  Librosa also supports fractional MIDI note
#   numbers, which allow for conversion of frequencies between those on the 12TET grid.
#
# The table below illustrates the relationships between the three systems described above.
#
# +------+-------------+----------------+
# | MIDI | Pitch (SPN) | Frequency (Hz) |
# +======+=============+================+
# | 0    | C-1         | 8.1758         | 
# +------+-------------+----------------+
# | 1    | C♯-1        | 8.6619         |
# +------+-------------+----------------+
# | 2    | D-1         | 9.1770         |
# +------+-------------+----------------+
# | ...  | ...         | ...            |
# +------+-------------+----------------+
# | 60   | C4          | 261.626        |
# +------+-------------+----------------+
# | 61   | C♯4         | 277.183        |
# +------+-------------+----------------+
# | 62   | D4          | 293.665        |
# +------+-------------+----------------+
# | ...  | ...         | ...            |
# +------+-------------+----------------+
# | 127  | G9          | 12543.854      |
# +------+-------------+----------------+
#
# Librosa provides functions to convert between any pair of these representations, as illustrated the 
# example code below.
import numpy as np
import librosa

# Generate one octave of MIDI notes, starting at middle C (MIDI 60):
midi = np.arange(60, 72)
print(midi)

# %%
# We can then convert these MIDI note numbers to frequencies in Hz:
frequencies = librosa.midi_to_hz(midi)
print(frequencies)

# %%
# ... or to pitches in SPN:
pitches = librosa.midi_to_note(midi)
print(pitches)

# %%
# The conversion can be done in the other direction as well:
midi_from_pitch = librosa.note_to_midi(pitches)
print(midi_from_pitch)

# %%
# And we can short-cut directly between pitch and frequency:
freq_from_pitch = librosa.note_to_hz(pitches)
print(freq_from_pitch)

# %%
#

# %%
# Keys and degrees
# ----------------

# %%
# Unicode
# -------

# %%
# Hindustani and Carnatic notation
# --------------------------------

# %% 
# Non-equal temperament
# ---------------------

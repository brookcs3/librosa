# coding: utf-8
"""
===========================
Timbre and spectral content
===========================

This section introduces common tools for representing the timbral or textural
content of audio signals.
"""

# to include:
# 
#   - rms energy
#   - spectral centroid
#   - spectral bandwidth
#   - spectral rolloff
#   - spectral contrast
#   
# what we need is a recording that shows three different instruments
# playing three different notes, arpeggiated and then as a chord
#   the arpeggiation should vary dynamics as well as pitch
#   probably we can assemble this from tinysol or something
#   we can then mix in some noise to show how these features change

# introduce mel spectrogram first
# then, mfccs
#  - we can show how they are derived from the mel spectrogram
#  - plot the DCT basis and explain spectral envelope extraction
#  - show how pitch variation is captured in the high order coefficients


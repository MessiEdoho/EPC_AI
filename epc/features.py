"""Feature extraction, matching the MATLAB code the model's training features were made with.

Reference scripts (MICE DATA_1/EEG_ML_VS_TL): eeg_50Hz.m, eeg_win2.m, waveSub2.m and the
feature spreadsheets Time_Domain_Features.xlsx / Time_Frequency_Domain_Features_Repair.xlsx.
tests/test_features.py checks this module against those spreadsheets.

Per channel, in volts:
  1. remove the mean
  2. 50 Hz notch, Q = 35, applied once forward (MATLAB `filter`, not zero-phase)
  3. split into 5 s windows of 2500 samples; a shorter final window is kept (eeg_win2.m)
  4. nonlinear energy of each window of the filtered signal
  5. db4 wavelet decomposition to level 7 of the whole filtered signal; the level 1 detail
     ("fast ripple", 125-250 Hz at 500 Hz sampling) is reconstructed as MATLAB
     wrcoef('d', c, l, 'db4', 1) does, windowed as in 3, and the energy of each window taken

The four features and their order follow the published model (Edoho et al., Fig. 6):
nonlinear energy of channels 1 and 2, then fast ripple energy of channels 1 and 2.
The spreadsheet columns are nonlin_energy_C1/C2 and energy_FR_C1/C2. The paper labels the
fast ripple band 250-500 Hz; at 500 Hz sampling the computed band is 125-250 Hz.
"""
import numpy as np
import pandas as pd
import pywt
from scipy import signal

FS = 500                    # Hz; the model was trained on 500 Hz recordings only
WIN = 2500                  # samples per window (5 s)
NOTCH_HZ = 50.0
NOTCH_Q = 35.0
WAVELET = "db4"
WAVELET_LEVEL = 7
FAST_RIPPLE_LEVEL = 1       # wavelet detail level 1: 125-250 Hz

# Column order is the order the model was trained with. Do not change it.
FEATURE_NAMES = ["Nl_ene_c1", "Nl_ene_c2", "Energy_fr_c1", "Energy_fr_c2"]

_NOTCH_B, _NOTCH_A = signal.iirnotch(NOTCH_HZ, NOTCH_Q, FS)


def preprocess(x):
    """Remove the mean, then notch out 50 Hz with a single forward pass."""
    x = np.asarray(x, dtype=np.float64)
    return signal.lfilter(_NOTCH_B, _NOTCH_A, x - x.mean())


def windows(x, win=WIN):
    """Split as eeg_win2.m does: windows of `win` samples, plus a shorter final window."""
    return [x[i:i + win] for i in range(0, len(x) - 1, win)]


def detail_band(x, level):
    """Wavelet detail at `level` reconstructed at full length (MATLAB wrcoef 'd')."""
    coeffs = pywt.wavedec(x, WAVELET, level=WAVELET_LEVEL)
    keep = [np.zeros_like(c) for c in coeffs]
    keep[-level] = coeffs[-level]
    return pywt.waverec(keep, WAVELET)[:len(x)]


def energy(w):
    return float(np.sum(w ** 2))


def nonlinear_energy(w):
    """Sum of the Teager-Kaiser operator x[t]^2 - x[t-1] x[t+1]."""
    return float(np.sum(w[1:-1] ** 2 - w[:-2] * w[2:]))


def channel_features(x):
    """Nonlinear energy and fast ripple energy for each window of one channel (in volts)."""
    clean = preprocess(x)
    nl = [nonlinear_energy(w) for w in windows(clean)]
    fr = [energy(w) for w in windows(detail_band(clean, FAST_RIPPLE_LEVEL))]
    return nl, fr


def extract_features(eeg1, eeg2):
    """Feature table (one row per window) for a two-channel recording in volts."""
    nl1, fr1 = channel_features(eeg1)
    nl2, fr2 = channel_features(eeg2)
    return pd.DataFrame(np.column_stack([nl1, nl2, fr1, fr2]), columns=FEATURE_NAMES)

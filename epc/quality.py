"""Data-quality warnings shown with a result. They never block an analysis.

  Q01  signal amplitude far outside the EEG range in volts (probably a wrong unit)
  Q02  recording much shorter than the recordings the model was developed on
  Q03  flat or disconnected stretches
  Q04  the two channels are nearly identical
"""
import numpy as np

from epc import features as feat

TYPICAL_STD_V = (1e-6, 1e-2)       # training EEG: window SD around 30 µV
SHORT_MINUTES = 10                 # training recordings averaged 34-43 minutes (paper, Table I)
FLAT_FRACTION = 0.02               # window SD below 2% of the recording's median window SD
FLAT_WARN_SHARE = 0.01             # warn when more than 1% of windows are flat
IDENTICAL_CORR = 0.98


def _window_sd(x):
    return np.array([np.std(w) for w in feat.windows(x) if w.size > 1])


def check(eeg1, eeg2):
    """Warnings for a two-channel recording in volts, as dicts with code, title and message."""
    warnings = []
    clean = [feat.preprocess(x) for x in (eeg1, eeg2)]
    sds = [_window_sd(x) for x in clean]

    typical = [float(np.median(sd)) for sd in sds]
    for name, value in zip(("EEG1", "EEG2"), typical):
        if value > TYPICAL_STD_V[1]:
            warnings.append(dict(code="Q01", title="Signal unit may be wrong", message=(
                f"{name} varies by about {value:.3g} V per window, far more than EEG in volts "
                "(tens of µV). If the data are in µV or mV, choose that unit and run the analysis again; "
                "the result below is not reliable.")))
        elif 0 < value < TYPICAL_STD_V[0]:
            warnings.append(dict(code="Q01", title="Signal unit may be wrong", message=(
                f"{name} varies by only about {value * 1e6:.3g} µV per window, much less than typical EEG. "
                "If the data are in volts, choose V and run the analysis again; the result below is "
                "not reliable.")))

    minutes = len(eeg1) / feat.FS / 60
    if minutes < SHORT_MINUTES:
        warnings.append(dict(code="Q02", title="Short recording", message=(
            f"The recording is {minutes:.1f} minutes long. The model was developed on status epilepticus "
            "recordings averaging 34-43 minutes, so a short recording gives a less certain result.")))

    flat = np.zeros(min(len(s) for s in sds), dtype=bool)
    for sd, med in zip(sds, typical):
        flat |= sd[:flat.size] < FLAT_FRACTION * med
    if flat.size and flat.mean() > FLAT_WARN_SHARE:
        warnings.append(dict(code="Q03", title="Flat stretches", message=(
            f"{int(flat.sum())} of {flat.size} windows ({flat.mean():.0%}) are flat on at least one "
            "channel, which can mean a disconnected electrode or missing data. These windows still "
            "count in the vote.")))

    if np.std(clean[0]) > 0 and np.std(clean[1]) > 0:
        r = float(np.corrcoef(clean[0], clean[1])[0, 1])
        if r > IDENTICAL_CORR:
            warnings.append(dict(code="Q04", title="Channels look identical", message=(
                f"EEG1 and EEG2 are almost the same signal (correlation {r:.3f}). Check that two "
                "different electrodes were uploaded or selected.")))
    return warnings

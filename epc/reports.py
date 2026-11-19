"""The downloadable result: feature tables and amplitude/spectrogram figures in one zip."""
import io
import zipfile

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from epc.features import FS  # noqa: E402

ZIP_NAME = "Feature_analysis.zip"


def eeg_figure(x_volts, title):
    """Amplitude and spectrogram of one channel, as JPEG bytes."""
    x = np.asarray(x_volts) * 1e6                           # plot in µV
    t = np.arange(x.size) / FS
    fig = Figure(figsize=(12, 10))
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(t, x, linewidth=0.5)
    ax1.set_ylabel("µV")
    ax1.set_title(f"{title}: amplitude")
    peak = np.max(np.abs(x))
    vmin = 20 * np.log10(peak) - 58.1 if peak > 0 else None   # hide anything far below the peak
    with np.errstate(divide="ignore"):                      # flat stretches have zero power
        ax2.specgram(x, Fs=FS, cmap="jet", vmin=vmin)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Freq (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"{title}: spectrogram")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg")
    return buf.getvalue()


def result_zip(path, features, window_class, eeg1, eeg2, channel_labels):
    """Write the result zip for one analysis."""
    summary = pd.concat([features.mean(), features.std()], axis=1)
    summary.columns = ["Mean", "STD"]
    per_window = features.copy()
    per_window.insert(0, "start_s", np.arange(len(features)) * 5)
    per_window["outlier"] = window_class
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Feature_analysis.csv", summary.to_csv())
        z.writestr("Window_features.csv", per_window.to_csv(index=False))
        z.writestr("EEG1.jpg", eeg_figure(eeg1, f"EEG1 ({channel_labels[0]})"))
        z.writestr("EEG2.jpg", eeg_figure(eeg2, f"EEG2 ({channel_labels[1]})"))

"""Result files and figures: the downloadable zip, the EEG figures and the window timeline."""
import io
import zipfile
from html import escape

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from epc.features import FS, WIN  # noqa: E402

ZIP_NAME = "Feature_analysis.zip"
FIGURES = ("EEG1.jpg", "EEG2.jpg")


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


def write_results(folder, features, window_class, window_prob, eeg1, eeg2, channel_labels):
    """Write the EEG figures and the result zip for one analysis into `folder`."""
    summary = pd.concat([features.mean(), features.std()], axis=1)
    summary.columns = ["Mean", "STD"]
    per_window = features.copy()
    per_window.insert(0, "start_s", np.arange(len(features)) * WIN // FS)
    per_window["p_outlier"] = np.round(window_prob, 4)
    per_window["outlier"] = window_class
    figures = {
        FIGURES[0]: eeg_figure(eeg1, f"EEG1 ({channel_labels[0]})"),
        FIGURES[1]: eeg_figure(eeg2, f"EEG2 ({channel_labels[1]})"),
    }
    for name, data in figures.items():
        (folder / name).write_bytes(data)
    with zipfile.ZipFile(folder / ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Feature_analysis.csv", summary.to_csv())
        z.writestr("Window_features.csv", per_window.to_csv(index=False))
        for name, data in figures.items():
            z.writestr(name, data)


def timeline_svg(window_class, window_prob, n_samples):
    """One coloured block per 5 s window, the outlier probability above, minutes below."""
    width, strip_top, strip_h, prob_h = 1000.0, 46, 22, 36
    plot_w = width - 34                                     # right margin for the probability scale
    total_s = n_samples / FS
    scale = plot_w / total_s
    parts = [f'<svg class="timeline" viewBox="0 0 {width:g} 92" role="img" '
             'aria-label="Classification of each five-second window over the recording">']
    for i, cls in enumerate(window_class):
        x0 = i * WIN / FS * scale
        w = min(WIN / FS, total_s - i * WIN / FS) * scale
        kind = "outlier" if cls else "normal"
        parts.append(f'<rect class="tl-{kind}" x="{x0:.2f}" y="{strip_top}" width="{max(w, 0.5):.2f}" '
                     f'height="{strip_h}"><title>{i * 5 // 60}:{i * 5 % 60:02d} '
                     f'{kind}, p = {window_prob[i]:.2f}</title></rect>')
    centres = [(i * WIN / FS + min(WIN / FS, total_s - i * WIN / FS) / 2) * scale for i in range(len(window_prob))]
    points = " ".join(f"{x:.1f},{4 + (1 - p) * (prob_h - 4):.1f}" for x, p in zip(centres, window_prob))
    half = 4 + 0.5 * (prob_h - 4)
    parts.append(f'<line class="tl-half" x1="0" x2="{plot_w:g}" y1="{half:.1f}" y2="{half:.1f}"/>')
    parts.append(f'<polyline class="tl-prob" points="{points}"/>')
    for p_label, y in (("p=1", 4), ("0.5", half), ("0", prob_h)):
        parts.append(f'<text class="tl-axis" x="{width - 2:g}" y="{y + 3:.1f}" text-anchor="end">{p_label}</text>')
    minutes = total_s / 60
    step = 1 if minutes <= 10 else 5 if minutes <= 40 else 10
    for m in range(0, int(minutes) + 1, step):
        x = m * 60 * scale
        anchor = "start" if m == 0 else "end" if x > plot_w - 20 else "middle"
        parts.append(f'<line class="tl-tick" x1="{x:.1f}" x2="{x:.1f}" y1="{strip_top + strip_h}" '
                     f'y2="{strip_top + strip_h + 4}"/>')
        parts.append(f'<text x="{x:.1f}" y="90" text-anchor="{anchor}">{escape(str(m))} min</text>')
    parts.append("</svg>")
    return "".join(parts)

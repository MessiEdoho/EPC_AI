"""Make the images in static/.

    python tools/make_images.py                  # banner and mouse illustration only
    python tools/make_images.py <EEG_ML_VS_TL>   # also the feature and example-EEG figures

EEG_amp.png and Mice_img.png are drawn from synthetic signals. EEG_feature.png and
EEG_signal.png are computed from the readable EDF recordings in the given data folder
(TRAINING/GOOD = normal, TRAINING/NONGOOD = outlier; "New folder" is ignored).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epc import features, validation  # noqa: E402

STATIC = ROOT / "static"
ORANGE, GREY, BLUE = "#ff7d33", "#5b5b5b", "#3a6ea5"


def synthetic_eeg(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 500
    x = np.cumsum(rng.standard_normal(n)) * 0.08
    x -= np.convolve(x, np.ones(200) / 200, mode="same")
    x += 0.6 * np.sin(2 * np.pi * 6 * t) * (1 + 0.6 * np.sin(2 * np.pi * 0.3 * t))
    for c in rng.choice(n - 60, 6, replace=False):               # spikes
        x[c:c + 60] += 3 * np.exp(-np.linspace(0, 6, 60)) * np.sin(np.linspace(0, 3 * np.pi, 60))
    return x[200:-200]                                           # drop the smoothing edges


def banner():
    fig = Figure(figsize=(6, 2), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    for k, seed in enumerate((1, 2)):
        ax.plot(synthetic_eeg(2000, seed) + 4 * k, color=(ORANGE, BLUE)[k], linewidth=1)
    ax.axis("off")
    fig.savefig(STATIC / "EEG_amp.png", transparent=True)


def mouse():
    fig = Figure(figsize=(8, 5), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fur, ear, line = "#c9c4bd", "#e8b4b0", "#4a4a4a"
    t = np.linspace(0, 1, 100)                                    # tail
    ax.plot(2.2 - 2.0 * t, 2.6 + 1.2 * np.sin(np.pi * t * 1.5), color=ear, linewidth=5, solid_capstyle="round")
    ax.add_patch(Ellipse((5.0, 3.4), 6.0, 3.4, color=fur, ec=line, lw=1.5))       # body
    ax.add_patch(Ellipse((8.3, 4.0), 3.2, 2.4, angle=-15, color=fur, ec=line, lw=1.5))  # head
    ax.add_patch(Circle((7.6, 5.3), 0.75, color=ear, ec=line, lw=1.5))           # ear
    ax.add_patch(Circle((9.0, 4.2), 0.16, color="black"))                        # eye
    ax.add_patch(Circle((9.85, 3.55), 0.14, color=ear, ec=line, lw=1))           # nose
    for dy in (-0.15, 0.1):                                                      # whiskers
        ax.plot([9.7, 10.6], [3.6 + dy, 3.6 + dy * 3], color=line, linewidth=0.8)
    for x0 in (3.6, 6.0):                                                        # feet
        ax.add_patch(Ellipse((x0, 1.75), 0.9, 0.35, color=ear, ec=line, lw=1))
    ax.add_patch(FancyBboxPatch((7.75, 4.9), 0.9, 0.45, boxstyle="round,pad=0.05",  # headmount
                                color="#8c8c8c", ec=line, lw=1.2))
    for k, (colour, seed) in enumerate(((ORANGE, 3), (BLUE, 4))):                 # leads and traces
        y = 8.6 - 1.6 * k
        ax.plot([8.2, 8.2 + 0.4 * k, 10.6], [5.35, y, y], color=colour, linewidth=1.5)
        trace = synthetic_eeg(900, seed)
        ax.plot(np.linspace(10.8, 15.6, trace.size), y + 0.28 * trace, color=colour, linewidth=0.9)
        ax.text(15.6, y + 0.75, f"EEG{k + 1}", color=colour, ha="right", fontsize=12)
    fig.savefig(STATIC / "Mice_img.png", transparent=True)


def readable_recordings(data_dir, split):
    for cls, label in (("GOOD", "Normal"), ("NONGOOD", "Outlier")):
        for path in sorted((data_dir / split / cls).glob("*.edf")):   # skips "New folder"
            try:
                usable = [c["index"] for c in validation.edf_channels(path) if c["usable"]]
                eeg1, eeg2, _ = validation.load_edf(path, usable[:2])
            except validation.ValidationError:
                continue
            yield path, label, eeg1, eeg2


def feature_figure(data_dir):
    tables = {"Normal": [], "Outlier": []}
    mice = {"Normal": 0, "Outlier": 0}
    for _, label, eeg1, eeg2 in readable_recordings(data_dir, "TRAINING"):
        tables[label].append(features.extract_features(eeg1, eeg2))
        mice[label] += 1
    titles = ["Nonlinear energy, EEG1", "Nonlinear energy, EEG2", "Fast ripple energy, EEG1",
              "Fast ripple energy, EEG2"]
    fig = Figure(figsize=(11, 5), dpi=120)
    axes = fig.subplots(1, 4)
    for ax, name, title in zip(axes, features.FEATURE_NAMES, titles):
        data = [np.concatenate([t[name].to_numpy() for t in tables[c]]) for c in ("Normal", "Outlier")]
        data = [d[d > 0] for d in data]                                # log scale
        box = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.6)
        for patch, colour in zip(box["boxes"], (BLUE, ORANGE)):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)
        for med in box["medians"]:
            med.set_color("black")
        ax.set_yscale("log")
        ax.set_xticks([1, 2], ["Normal", "Outlier"])
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("Feature value per 5 s window (V²)")
    fig.suptitle(f"Model features by phenotype (training recordings: {mice['Normal']} normal, "
                 f"{mice['Outlier']} outlier mice; whiskers 5th-95th percentile)", fontsize=12)
    fig.tight_layout()
    fig.savefig(STATIC / "EEG_feature.png")


def example_figure(data_dir, seconds=20):
    path, _, eeg1, eeg2 = next(r for r in readable_recordings(data_dir, "TESTING") if r[1] == "Outlier")
    n = seconds * features.FS
    start = (len(eeg1) // 2 // n) * n                                  # a stretch from mid-recording
    t = np.arange(n) / features.FS
    fig = Figure(figsize=(11, 6), dpi=120)
    axes = fig.subplots(3, 1, sharex=True)
    for ax, x, name, colour in ((axes[0], eeg1, "EEG1", ORANGE), (axes[1], eeg2, "EEG2", BLUE)):
        ax.plot(t, features.preprocess(x)[start:start + n] * 1e6, color=colour, linewidth=0.6)
        ax.set_ylabel(f"{name} (µV)")
    axes[2].specgram(eeg1[start:start + n] * 1e6, Fs=features.FS, cmap="jet")
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("Freq (Hz)")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title(f"Example outlier recording: {seconds} s of status epilepticus EEG with EEG1 spectrogram")
    fig.tight_layout()
    fig.savefig(STATIC / "EEG_signal.png")
    return path.name


if __name__ == "__main__":
    STATIC.mkdir(exist_ok=True)
    banner()
    mouse()
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
        feature_figure(data_dir)
        print("example recording:", example_figure(data_dir))

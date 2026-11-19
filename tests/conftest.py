import os
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from epc import model as mdl
from epc.config import DEFAULT_MODEL_PATH, Settings

MODEL_PATH = Path(os.environ.get("EPC_MODEL_PATH", DEFAULT_MODEL_PATH))
DATA_DIR = Path(os.environ.get(
    "EPC_DATA_DIR", r"C:\Users\messi\OneDrive\Documents\UCD PhD\MICE DATA_1\EEG_ML_VS_TL"))

needs_model = pytest.mark.skipif(not MODEL_PATH.is_file(), reason="model file not available")
needs_data = pytest.mark.skipif(not DATA_DIR.is_dir(), reason="training data folder not available")


def eeg(seconds, fs=500, seed=0, scale=50e-6):
    """Noise plus an 8 Hz rhythm, in volts."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    return scale * (rng.standard_normal(t.size) + np.sin(2 * np.pi * 8 * t))


def write_edf(path, signals, fs=500, unit="V"):
    """Write {label: samples} to an EDF file. Values must fit within ±5 mV (in `unit`)."""
    limit = {"V": 0.005, "mV": 5.0, "uV": 5000.0}[unit]
    rates = fs if isinstance(fs, dict) else {label: fs for label in signals}
    w = pyedflib.EdfWriter(str(path), len(signals), file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        w.setSignalHeaders([
            {"label": label, "dimension": unit, "sample_frequency": rates[label],
             "physical_max": limit, "physical_min": -limit, "digital_max": 32767, "digital_min": -32768}
            for label in signals
        ])
        w.writeSamples([np.asarray(x, dtype=np.float64) for x in signals.values()])
    finally:
        w.close()
    return path


@pytest.fixture(scope="session")
def model():
    if not MODEL_PATH.is_file():
        pytest.skip("model file not available")
    return mdl.load_model(MODEL_PATH)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        model_path=MODEL_PATH,
        upload_dir=tmp_path / "uploads",
        results_dir=tmp_path / "results",
        samples_dir=tmp_path / "samples",
        max_upload_mb=5,
        keep_hours=3,
    )


@pytest.fixture
def client(settings, model):
    from epc.server import create_app
    app = create_app(settings, model)
    return app.test_client()

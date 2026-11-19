"""Data-quality warnings Q01-Q04."""
import numpy as np

from conftest import eeg
from epc import quality


def codes(eeg1, eeg2):
    return {w["code"] for w in quality.check(eeg1, eeg2)}


def test_clean_recording_has_no_warnings():
    assert codes(eeg(15 * 60), eeg(15 * 60, seed=1)) == set()


def test_q01_microvolts_uploaded_as_volts():
    assert "Q01" in codes(eeg(15 * 60) * 1e6, eeg(15 * 60, seed=1) * 1e6)


def test_q01_volts_uploaded_as_microvolts():
    assert "Q01" in codes(eeg(15 * 60) * 1e-6, eeg(15 * 60, seed=1) * 1e-6)


def test_q02_short_recording():
    assert codes(eeg(5 * 60), eeg(5 * 60, seed=1)) == {"Q02"}


def test_q03_flat_stretch():
    a, b = eeg(15 * 60), eeg(15 * 60, seed=1)
    b[:60 * 500] = 0.0                                     # one minute disconnected
    assert codes(a, b) == {"Q03"}


def test_q04_identical_channels():
    a = eeg(15 * 60)
    assert codes(a, a + 1e-9 * np.random.default_rng(5).standard_normal(a.size)) == {"Q04"}

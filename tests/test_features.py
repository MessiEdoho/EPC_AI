"""Feature extraction: windowing rules, per-channel processing, and parity with training features."""
import numpy as np
import pandas as pd
import pytest

from conftest import DATA_DIR, eeg, needs_data
from epc import features, validation


def test_windows_follow_eeg_win2():
    x = np.arange(6000)
    w = features.windows(x)
    assert [len(v) for v in w] == [2500, 2500, 1000]          # full 5 s windows plus the remainder
    assert w[1][0] == 2500 and w[2][-1] == 5999
    assert [len(v) for v in features.windows(np.arange(5000))] == [2500, 2500]


def test_length_not_multiple_of_window():
    table = features.extract_features(eeg(12.3), eeg(12.3, seed=1))
    assert len(table) == 3 and list(table.columns) == features.FEATURE_NAMES


def test_each_channel_uses_its_own_wavelet_decomposition():
    a, b = eeg(20), eeg(20, seed=1, scale=500e-6)
    t = features.extract_features(a, b)
    assert not np.allclose(t.Energy_rip_c1, t.Energy_rip_c2)
    swapped = features.extract_features(b, a)
    assert np.allclose(t.Energy_rip_c2, swapped.Energy_rip_c1)
    assert np.allclose(t.Nl_ene_c2, swapped.Nl_ene_c1)


@needs_data
@pytest.mark.parametrize("mouse,edf", [
    ("S2_M30", r"TESTING\GOOD\OMS2-M30-1-111121_reduced.edf"),
    ("S2_M16", r"TESTING\NONGOOD\OMS2 M16.1_reduced.edf"),
])
def test_matches_training_feature_spreadsheets(mouse, edf):
    """The four model features equal the MATLAB training features for the same recording.

    The first window differs by up to 2e-5 (filter start-up), so it has a looser tolerance.
    Not used here: S10_M01, S10_M13 and S10_M20, whose spreadsheet C2 columns are a copy of C1.
    """
    td = pd.read_excel(DATA_DIR / "Time_Domain_Features.xlsx", sheet_name="TEST_SET")
    tf = pd.read_excel(DATA_DIR / "Time_Frequency_Domain_Features_Repair.xlsx", sheet_name="Sheet2")
    ref_td, ref_tf = td[td.mouseid == mouse], tf[tf.mouseid == mouse]
    eeg1, eeg2, _ = validation.load_edf(DATA_DIR / edf)
    t = features.extract_features(eeg1, eeg2)
    assert len(t) == len(ref_td) == len(ref_tf)
    pairs = [("Nl_ene_c1", ref_td.nonlin_energy_C1), ("Nl_ene_c2", ref_td.nonlin_energy_C2),
             ("Energy_rip_c1", ref_tf.energy_ripple_C1), ("Energy_rip_c2", ref_tf.energy_ripple_C2)]
    for name, ref in pairs:
        ours, ref = t[name].to_numpy(), ref.to_numpy()
        np.testing.assert_allclose(ours[1:], ref[1:], rtol=1e-6, err_msg=name)
        np.testing.assert_allclose(ours[0], ref[0], rtol=1e-4, err_msg=f"{name}, first window")

"""Upload checks V01-V08 on small synthetic files."""
import numpy as np
import pytest

from conftest import eeg, write_edf
from epc import validation
from epc.validation import ValidationError


def code_of(call):
    with pytest.raises(ValidationError) as exc:
        call()
    return exc.value.code


def write_csv(path, rows, header=None):
    lines = ([",".join(header)] if header else []) + [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("name", ["", "data.pickle", "data.txt", "noext"])
def test_v01_file_kind(name):
    assert code_of(lambda: validation.file_kind(name)) == "V01"


def test_file_kind_accepts_any_case():
    assert validation.file_kind("A.EDF") == "edf" and validation.file_kind("b.Csv") == "csv"


def test_two_channel_edf_in_volts(tmp_path):
    a, b = eeg(20), eeg(20, seed=1)
    p = write_edf(tmp_path / "ok.edf", {"EEG 1": a, "EEG 2": b})
    x1, x2, labels = validation.load_edf(p)
    assert labels == ["EEG 1", "EEG 2"]
    np.testing.assert_allclose(x1, a, atol=1e-6)               # within EDF 16-bit resolution


def test_edf_microvolts_converted_to_volts(tmp_path):
    a = eeg(20)
    p = write_edf(tmp_path / "uv.edf", {"A": a * 1e6, "B": a * 1e6}, unit="uV")
    x1, _, _ = validation.load_edf(p)
    np.testing.assert_allclose(x1, a, atol=1e-6)


def test_v02_damaged(tmp_path):
    p = tmp_path / "bad.edf"
    p.write_bytes(b"not an edf file" * 50)
    assert code_of(lambda: validation.edf_channels(p)) == "V02"


def test_more_than_two_channels_needs_choice(tmp_path):
    p = write_edf(tmp_path / "four.edf", {k: eeg(20, seed=i) for i, k in enumerate("ABCD")})
    assert validation.needs_channel_choice(validation.edf_channels(p))
    _, _, labels = validation.load_edf(p, (3, 1))
    assert labels == ["D", "B"]


def test_v04_one_channel(tmp_path):
    p = write_edf(tmp_path / "one.edf", {"A": eeg(20)})
    assert code_of(lambda: validation.needs_channel_choice(validation.edf_channels(p))) == "V04"


@pytest.mark.parametrize("picks", [(0, 0), (0, 9), ("x", 1), (None, 1)])
def test_v04_bad_channel_choice(tmp_path, picks):
    p = write_edf(tmp_path / "three.edf", {k: eeg(20, seed=i) for i, k in enumerate("ABC")})
    assert code_of(lambda: validation.load_edf(p, picks)) == "V04"


def test_v05_edf_not_500_hz(tmp_path):
    p = write_edf(tmp_path / "fast.edf", {"A": eeg(20, fs=1000), "B": eeg(20, fs=1000)}, fs=1000)
    assert code_of(lambda: validation.load_edf(p)) == "V05"


def test_v08_too_short(tmp_path):
    p = write_edf(tmp_path / "short.edf", {"A": eeg(4), "B": eeg(4, seed=1)})
    assert code_of(lambda: validation.load_edf(p)) == "V08"


@pytest.mark.parametrize("layout", ["two", "index", "header", "index+header"])
def test_csv_layouts(tmp_path, layout):
    a, b = eeg(10), eeg(10, seed=1)
    rows = [(i, x, y) if "index" in layout else (x, y) for i, (x, y) in enumerate(zip(a, b))]
    header = (["", "EEG1", "EEG2"] if "index" in layout else ["EEG1", "EEG2"]) if "header" in layout else None
    p = write_csv(tmp_path / "ok.csv", rows, header)
    x1, x2, _ = validation.load_csv(p, "500", "V")
    np.testing.assert_allclose(x1, a)
    np.testing.assert_allclose(x2, b)


def test_csv_units(tmp_path):
    a = eeg(10)
    p = write_csv(tmp_path / "uv.csv", [(x * 1e6, x * 1e6) for x in a])
    x1, _, _ = validation.load_csv(p, "500", "µV")
    np.testing.assert_allclose(x1, a)


@pytest.mark.parametrize("fs", ["250", "", "abc", None])
def test_v05_csv_sampling_frequency(tmp_path, fs):
    p = write_csv(tmp_path / "ok.csv", [(1, 2)] * 3000)
    assert code_of(lambda: validation.load_csv(p, fs, "V")) == "V05"


def test_v06_csv_unit_required(tmp_path):
    p = write_csv(tmp_path / "ok.csv", [(1, 2)] * 3000)
    assert code_of(lambda: validation.load_csv(p, "500", None)) == "V06"


def test_v04_csv_wrong_column_count(tmp_path):
    p = write_csv(tmp_path / "four.csv", [(0, 1, 2, 3)] * 3000)
    assert code_of(lambda: validation.load_csv(p, "500", "V")) == "V04"


def test_v03_csv_not_numeric(tmp_path):
    p = write_csv(tmp_path / "text.csv", [(1, 2)] * 3000 + [("a", "b")])
    assert code_of(lambda: validation.load_csv(p, "500", "V")) == "V03"

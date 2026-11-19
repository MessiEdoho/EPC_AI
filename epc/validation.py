"""Upload checks. Each check raises ValidationError with a code and a plain message.

  V01  no file, or not an EDF or CSV file
  V02  the file is damaged or could not be read
  V03  the signal data are missing or not numeric
  V04  not exactly two EEG channels, or an invalid channel choice
  V05  sampling frequency is not 500 Hz
  V06  signal unit not recognised
  V07  file too large (checked by the server before saving)
  V08  recording shorter than one 5 s window

Signals are returned in volts, the unit the model was trained with.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib

from epc.features import FS, WIN

ALLOWED_EXTENSIONS = {".edf", ".csv"}
UNIT_TO_VOLT = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6, "μV": 1e-6}   # µ: micro sign and Greek mu
CSV_UNITS = ["V", "mV", "µV"]


class ValidationError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def file_kind(filename):
    """'edf' or 'csv' from the file name; anything else is refused."""
    if not filename:
        raise ValidationError("V01", "No file was selected. Please choose an EDF or CSV file.")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError("V01", "This file is not an EDF or CSV file.")
    return ext[1:]


def _to_volts(unit):
    factor = UNIT_TO_VOLT.get(str(unit).strip())
    if factor is None:
        raise ValidationError("V06", f"Signal unit '{unit}' is not recognised. Use V, mV or µV.")
    return factor


def _check_signals(eeg1, eeg2):
    for x in (eeg1, eeg2):
        if x.size == 0:
            raise ValidationError("V03", "The file contains no signal data.")
        if not np.all(np.isfinite(x)):
            raise ValidationError("V03", "The signal contains missing or non-numeric values.")
    if eeg1.size != eeg2.size:
        raise ValidationError("V04", "The two EEG channels have different lengths.")
    if eeg1.size < WIN:
        raise ValidationError("V08", f"The recording is shorter than 5 s ({WIN} samples at {FS} Hz).")


# --------------------------------------------------------------------------- EDF

def edf_channels(path):
    """Header information for each signal in an EDF file."""
    try:
        reader = pyedflib.EdfReader(str(path))
    except Exception:
        raise ValidationError("V02", "The file appears to be damaged or is not a valid EDF file.")
    try:
        return [
            {
                "index": i,
                "label": reader.getLabel(i).strip(),
                "fs": float(reader.getSampleFrequency(i)),
                "unit": reader.getPhysicalDimension(i).strip(),
                "usable": float(reader.getSampleFrequency(i)) == FS,
            }
            for i in range(reader.signals_in_file)
        ]
    finally:
        reader.close()


def needs_channel_choice(channels):
    if len(channels) < 2:
        raise ValidationError("V04", "The EDF file must contain two EEG channels; this one has "
                                     f"{len(channels)}.")
    return len(channels) > 2


def load_edf(path, channel_indices=(0, 1)):
    """The two chosen EDF channels, checked and converted to volts."""
    channels = edf_channels(path)
    try:
        picks = [int(i) for i in channel_indices]
    except (TypeError, ValueError):
        raise ValidationError("V04", "Please choose two EEG channels.")
    if len(picks) != 2 or picks[0] == picks[1]:
        raise ValidationError("V04", "Please choose two different EEG channels.")
    if any(i < 0 or i >= len(channels) for i in picks):
        raise ValidationError("V04", "The selected channel was not found in this file.")
    for i in picks:
        ch = channels[i]
        if ch["fs"] != FS:
            raise ValidationError("V05", f"Channel '{ch['label']}' is sampled at {ch['fs']:g} Hz. "
                                         f"EPC-AI accepts {FS} Hz recordings only.")
    factors = [_to_volts(channels[i]["unit"]) for i in picks]

    try:
        reader = pyedflib.EdfReader(str(path))
    except Exception:
        raise ValidationError("V02", "The file appears to be damaged or is not a valid EDF file.")
    try:
        eeg1, eeg2 = (reader.readSignal(i) * f for i, f in zip(picks, factors))
    except Exception:
        raise ValidationError("V02", "The signal data could not be read from this EDF file.")
    finally:
        reader.close()
    _check_signals(eeg1, eeg2)
    return eeg1, eeg2, [channels[i]["label"] for i in picks]


# --------------------------------------------------------------------------- CSV

def _is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def load_csv(path, sampling_frequency, unit):
    """Two EEG columns from a CSV file, checked and converted to volts.

    Accepted layouts: two signal columns, or an index/time column followed by two signal
    columns. A header row is optional.
    """
    try:
        fs = float(sampling_frequency)
    except (TypeError, ValueError):
        raise ValidationError("V05", f"Please enter the sampling frequency in Hz ({FS}).")
    if fs != FS:
        raise ValidationError("V05", f"Sampling frequency is {fs:g} Hz. EPC-AI accepts {FS} Hz "
                                     "recordings only.")
    if unit not in CSV_UNITS:
        raise ValidationError("V06", "Please choose the signal unit (V, mV or µV).")

    try:
        df = pd.read_csv(path, header=None, dtype=str, skip_blank_lines=True)
    except Exception:
        raise ValidationError("V02", "The file appears to be damaged and could not be read as CSV.")
    if df.empty:
        raise ValidationError("V03", "The file contains no signal data.")
    if not all(_is_number(v) for v in df.iloc[0]):
        df = df.iloc[1:]                                    # header row
    if df.shape[1] == 3:
        df = df.iloc[:, 1:]                                 # leading index or time column
    elif df.shape[1] != 2:
        raise ValidationError("V04", "The CSV file must have two EEG columns (optionally after an "
                                     f"index column); this one has {df.shape[1]} columns.")
    try:
        data = df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    except Exception:
        raise ValidationError("V03", "The EEG columns must contain numbers only.")
    data = data * UNIT_TO_VOLT[unit]
    eeg1, eeg2 = data[:, 0], data[:, 1]
    _check_signals(eeg1, eeg2)
    return eeg1, eeg2, ["EEG1", "EEG2"]

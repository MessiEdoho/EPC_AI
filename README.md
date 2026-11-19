# EPC-AI

Web application that predicts the emergent epilepsy phenotype (normal or outlier) of a mouse from
two-channel EEG recorded during status epilepticus in the intra-amygdala kainic acid model. It runs the
trained XGBoost classifier `Mice.model` on features computed exactly as in the MATLAB code used to
build its training data.

EPC-AI is a research aid. It does not diagnose and is not a clinical device.

## Layout

| Path | Contents |
| --- | --- |
| `epc/features.py` | Feature extraction, matching the MATLAB training code |
| `epc/model.py` | Model loading (checksum-guarded, read-only) and classification |
| `epc/validation.py` | Upload checks V01 to V08 |
| `epc/reports.py` | The downloadable result zip |
| `epc/server.py` | Web routes |
| `epc/config.py` | Settings from environment variables |
| `templates/`, `static/` | Pages, stylesheet and images |
| `tools/make_images.py` | Regenerates the images in `static/` |
| `tests/` | Tests; `test_features.py` includes the parity check against the training features |
| `EPC_backend.py` | Starts the server |

## How a prediction is made

For each of the two channels, in volts:

1. remove the mean;
2. 50 Hz notch filter, Q = 35, applied once forward (MATLAB `filter`);
3. split into 5 s windows of 2500 samples, keeping a shorter final window (`eeg_win2.m`);
4. nonlinear energy of each window;
5. db4 wavelet decomposition to level 7 of the filtered channel; the level 2 detail (62.5–125 Hz,
   "ripple") is reconstructed as MATLAB `wrcoef` does, and its energy taken per window.

The model classifies each window from the four features `Nl_ene_c1, Energy_rip_c1, Nl_ene_c2,
Energy_rip_c2`; the mouse is "Outlier" when 50% or more of its windows are.

## Input rules

- EDF or CSV, at most 100 MB.
- Sampling frequency must be 500 Hz. Other rates are refused.
- Exactly two EEG channels. An EDF with more channels asks the user to choose two.
- EDF units are read from the file (V, mV or µV). CSV uploads state the unit.
- CSV: two signal columns, optionally after an index or time column; header row optional.
- At least 5 s of recording.

## Setup

```
pip install -r requirements.txt
```

The model is not in the repository. By default it is read from
`C:\Users\messi\EPC_AI_Model\Mice.model`. The server refuses to start if the file does not match
SHA-256 `7ecbb9fdac0d1d54d088b3de22759a19ccb485ddcee999eee2db9d3e64a213d1`. The app never writes to
it. `Mice.json` in the same folder is an equivalent copy in XGBoost's current format (identical
predictions), kept for future XGBoost versions.

## Run

```
python EPC_backend.py
```

Then open http://localhost:8000/EP-Classification/.

Settings (environment variables):

| Variable | Default | Meaning |
| --- | --- | --- |
| `EPC_MODEL_PATH` | `C:\Users\messi\EPC_AI_Model\Mice.model` | Model file |
| `EPC_UPLOAD_DIR` | `<temp>/epc_uploads` | Uploads, deleted after each analysis |
| `EPC_RESULTS_DIR` | `<temp>/epc_results` | One folder per analysis with its result zip |
| `EPC_SAMPLES_DIR` | `samples/` | Put `Mice_example.zip` here to offer sample files |
| `EPC_MAX_UPLOAD_MB` | `100` | Largest accepted upload |
| `EPC_KEEP_HOURS` | `3` | Results and unfinished uploads older than this are deleted |
| `EPC_PORT` | `8000` | Server port (IctalX also defaults to 8000; use e.g. 8010 to run both) |

## Test

```
python -m pytest
```

The parity test needs the training data folder (`EPC_DATA_DIR`, default
`C:\Users\messi\OneDrive\Documents\UCD PhD\MICE DATA_1\EEG_ML_VS_TL`) and is skipped without it.
Tests that need the model are skipped when it is not available.

## Images

`static/EEG_amp.png` and `static/Mice_img.png` are drawn from synthetic signals.
`static/EEG_feature.png` and `static/EEG_signal.png` are computed from the readable training and test
recordings:

```
python tools/make_images.py "C:\Users\messi\OneDrive\Documents\UCD PhD\MICE DATA_1\EEG_ML_VS_TL"
```

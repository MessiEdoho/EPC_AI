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
5. db4 wavelet decomposition to level 7 of the filtered channel; the level 1 detail ("fast ripple",
   125–250 Hz) is reconstructed as MATLAB `wrcoef` does, and its energy taken per window.

The model classifies each window from four features, in this order (the published model, Fig. 6):
`Nl_ene_c1, Nl_ene_c2, Energy_fr_c1, Energy_fr_c2` (spreadsheet columns `nonlin_energy_C1/C2`,
`energy_FR_C1/C2`). The mouse is "Outlier" when 50% or more of its windows are. On the spreadsheet
test-set features this reproduces the published 11 of 16 mice (69%).

Channel 1 (EEG1) is the ipsilateral electrode (injection side), channel 2 (EEG2) the contralateral.

Note: the paper labels the fast ripple band 250–500 Hz; at 500 Hz sampling the highest possible
frequency is 250 Hz, and the band actually computed is 125–250 Hz.

## Input rules

- EDF or CSV, at most 100 MB.
- Sampling frequency must be 500 Hz. Other rates are refused.
- Exactly two EEG channels. An EDF with more channels asks the user to choose two.
- EDF units are read from the file (V, mV or µV). CSV uploads state the unit.
- CSV: two signal columns, optionally after an index or time column; header row optional.
- At least 5 s of recording.

## Setup

EPC-AI has its own conda environment, `epc-ai` (Python 3.12, packages pinned in
`environment.yml`; `requirements.txt` lists the same packages for pip):

```
conda env create -f environment.yml
conda activate epc-ai
```

After changing `environment.yml`: `conda env update -n epc-ai -f environment.yml --prune`.

The model is not in the repository. By default it is read from
`C:\Users\messi\EPC_AI_Model\Mice.model`; elsewhere set `EPC_MODEL_PATH`. The server refuses to
start if the file does not match SHA-256
`7ecbb9fdac0d1d54d088b3de22759a19ccb485ddcee999eee2db9d3e64a213d1`. The app never writes to it.
`Mice.json` in the same folder is an equivalent copy in XGBoost's current format (identical
predictions), kept for future XGBoost versions.

## Run locally

```
conda activate epc-ai
python EPC_backend.py
```

Then open http://localhost:8000/EP-Classification/. If port 8000 is taken (IctalX also uses it),
set another port first, e.g. in PowerShell `$env:EPC_PORT = "8010"`.

Settings (environment variables):

| Variable | Default | Meaning |
| --- | --- | --- |
| `EPC_MODEL_PATH` | `C:\Users\messi\EPC_AI_Model\Mice.model` | Model file |
| `EPC_UPLOAD_DIR` | `<temp>/epc_uploads` | Uploads, deleted after each analysis |
| `EPC_RESULTS_DIR` | `<temp>/epc_results` | One folder per analysis with its result zip |
| `EPC_SAMPLES_DIR` | `samples/` | Put `Mice_example.zip` here to offer sample files |
| `EPC_MAX_UPLOAD_MB` | `100` | Largest accepted upload |
| `EPC_KEEP_HOURS` | `3` | Results and unfinished uploads older than this are deleted |
| `EPC_URL_PREFIX` | `/EP-Classification` | Path every page and file is served under; empty if a proxy strips it |
| `EPC_PORT` | `8000` | Port for `python EPC_backend.py` (not used by gunicorn) |

## Deploy on lisda

The public address is `https://lisda.ucd.ie/EP-Classification/`. EPC-AI runs under gunicorn behind
nginx, like IctalX. Port 8001 and the paths below are examples; agree them with the lisda admin.

1. Copy the repository and the model to the server, e.g. `/opt/epc-ai/` and
   `/opt/epc-ai-model/Mice.model`. Check the model: `sha256sum Mice.model` must print the value
   above.
2. Create the environment: `conda env create -f environment.yml`.
3. Start the server from the repository folder:

   ```
   export EPC_MODEL_PATH=/opt/epc-ai-model/Mice.model
   export EPC_UPLOAD_DIR=/var/tmp/epc_uploads
   export EPC_RESULTS_DIR=/var/tmp/epc_results
   conda run -n epc-ai gunicorn -w 2 --threads 4 --timeout 120 -b 127.0.0.1:8001 EPC_backend:app
   ```

   Each worker loads the model once. Uploads and results are files on disk, so any worker can
   serve any request. Run it as a service (systemd, or whatever lisda uses for IctalX) so it
   restarts with the server.
4. nginx, keeping the path (the default `EPC_URL_PREFIX` matches it):

   ```
   location = /EP-Classification { return 301 /EP-Classification/; }
   location /EP-Classification/ {
       proxy_pass http://127.0.0.1:8001;
       proxy_set_header Host $host;
       proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       proxy_set_header X-Forwarded-Proto $scheme;
       client_max_body_size 100m;
       proxy_read_timeout 120s;
   }
   ```

   If nginx strips the path instead (as it does for IctalX), set `EPC_URL_PREFIX=` (empty) and add
   `proxy_set_header X-Forwarded-Prefix /EP-Classification;` so links keep the public address.
5. Check that `https://lisda.ucd.ie/EP-Classification/` opens and that an upload returns a result.

## Test

```
conda activate epc-ai
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

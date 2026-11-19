"""Web flows through the Flask test client, with the real model."""
import io
import re
import zipfile

import pytest

from conftest import MODEL_PATH, eeg, needs_model, write_edf
from epc import model as mdl

pytestmark = needs_model


def post_file(client, path, name=None, **form):
    data = {"eeg_file": (open(path, "rb"), name or path.name), **form}
    return client.post("/EPC_result", data=data, content_type="multipart/form-data")


def phenotype(response):
    m = re.search(rb"phenotype of this mouse is: <b>(\w+)</b>", response.data)
    return m.group(1).decode() if m else None


def result_id(response):
    return re.search(rb"/psrdownload/([0-9a-f]{32})", response.data).group(1).decode()


@pytest.mark.parametrize("url", ["/EP-Classification/", "/EP-Classification/EDF", "/EP-Classification/CSV",
                                 "/EP-Classification/EPC_help.html", "/static/EEG_amp.png",
                                 "/static/Mice_img.png", "/static/EEG_feature.png", "/static/EEG_signal.png"])
def test_pages(client, url):
    assert client.get(url).status_code == 200


def test_old_addresses_redirect(client):
    assert client.get("/").status_code == 302
    assert client.get("/EP-Classification/CSV-PICKLE").status_code == 302
    assert client.get("/EPC_result").status_code == 302


def test_two_channel_edf(client, settings, tmp_path):
    p = write_edf(tmp_path / "rec.edf", {"EEG 1": eeg(60), "EEG 2": eeg(60, seed=1)})
    r = post_file(client, p, format="edf")
    assert r.status_code == 200 and phenotype(r) in ("Normal", "Outlier")
    assert not any(settings.upload_dir.iterdir())                    # upload deleted
    z = client.get(f"/psrdownload/{result_id(r)}")
    assert z.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(z.data)).namelist()
    assert sorted(names) == ["EEG1.jpg", "EEG2.jpg", "Feature_analysis.csv", "Window_features.csv"]


def test_four_channel_edf_asks_for_channels(client, settings, tmp_path):
    p = write_edf(tmp_path / "rec.edf", {"Temp": eeg(60, fs=10), "A": eeg(60), "B": eeg(60, seed=1),
                                         "C": eeg(60, seed=2)}, fs={"Temp": 10, "A": 500, "B": 500, "C": 500})
    r = post_file(client, p, format="edf")
    assert r.status_code == 200 and b"Choose two EEG channels" in r.data
    assert b"not usable" in r.data                                   # the 10 Hz channel
    token = re.search(rb'name="token" value="([0-9a-f]{32})"', r.data).group(1).decode()

    same = client.post("/EPC_result/channels", data={"token": token, "channel1": "1", "channel2": "1"})
    assert same.status_code == 400 and b"two different" in same.data   # asked again, upload kept

    r = client.post("/EPC_result/channels", data={"token": token, "channel1": "3", "channel2": "1",
                                                  "filename": "rec.edf"})
    assert r.status_code == 200 and phenotype(r)
    assert b"C (EEG1), A (EEG2)" in r.data
    assert not any(settings.upload_dir.iterdir())


def test_channel_choice_rejects_wrong_rate(client, tmp_path):
    p = write_edf(tmp_path / "rec.edf", {"Temp": eeg(60, fs=10), "A": eeg(60), "B": eeg(60, seed=1)},
                  fs={"Temp": 10, "A": 500, "B": 500})
    r = post_file(client, p, format="edf")
    token = re.search(rb'name="token" value="([0-9a-f]{32})"', r.data).group(1).decode()
    r = client.post("/EPC_result/channels", data={"token": token, "channel1": "0", "channel2": "1"})
    assert r.status_code == 400 and b"500 Hz recordings only" in r.data


@pytest.mark.parametrize("token", ["", "../../etc", "0" * 32])
def test_expired_or_invalid_token(client, token):
    r = client.post("/EPC_result/channels", data={"token": token, "channel1": "0", "channel2": "1"})
    assert r.status_code == 400 and b"expired" in r.data


def test_csv(client, tmp_path):
    a, b = eeg(60), eeg(60, seed=1)
    p = tmp_path / "rec.csv"
    p.write_text("".join(f"{i},{x},{y}\n" for i, (x, y) in enumerate(zip(a, b))))
    r = post_file(client, p, format="csv", sampling_frequency="500", unit="V")
    assert r.status_code == 200 and phenotype(r)


def test_csv_wrong_sampling_frequency(client, tmp_path):
    p = tmp_path / "rec.csv"
    p.write_text("1,2\n" * 3000)
    r = post_file(client, p, format="csv", sampling_frequency="1000", unit="V")
    assert r.status_code == 400 and b"500 Hz recordings only" in r.data


def test_no_file(client):
    r = client.post("/EPC_result", data={"format": "edf"}, content_type="multipart/form-data")
    assert r.status_code == 400 and b"No file was selected" in r.data


def test_pickle_refused(client, tmp_path):
    p = tmp_path / "rec.pickle"
    p.write_bytes(b"\x80\x04")
    r = post_file(client, p, format="csv")
    assert r.status_code == 400 and b"not an EDF or CSV" in r.data


def test_too_large(client, tmp_path):
    p = tmp_path / "big.csv"
    p.write_bytes(b"0" * (6 * 1024 * 1024))
    r = post_file(client, p, format="csv")
    assert r.status_code == 413


def test_each_analysis_has_its_own_result(client, tmp_path):
    p = write_edf(tmp_path / "rec.edf", {"A": eeg(30), "B": eeg(30, seed=1)})
    ids = {result_id(post_file(client, p, format="edf")) for _ in range(2)}
    assert len(ids) == 2


def test_missing_downloads(client):
    assert client.get("/psrexample/").status_code == 404
    assert client.get("/psrdownload/" + "0" * 32).status_code == 404
    assert client.get("/psrdownload/not-a-token").status_code == 404


def test_samples_served_when_present(client, settings):
    settings.samples_dir.mkdir()
    (settings.samples_dir / "Mice_example.zip").write_bytes(b"PK")
    assert client.get("/psrexample/").status_code == 200


def test_model_never_written(client, tmp_path):
    p = write_edf(tmp_path / "rec.edf", {"A": eeg(30), "B": eeg(30, seed=1)})
    post_file(client, p, format="edf")
    assert mdl.sha256_file(MODEL_PATH) == mdl.MODEL_SHA256

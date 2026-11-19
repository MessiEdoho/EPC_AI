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
    return client.post("/EP-Classification/EPC_result", data=data, content_type="multipart/form-data")


def phenotype(response):
    m = re.search(rb'<p class="verdict-label">(\w+)</p>', response.data)
    return m.group(1).decode() if m else None


def result_id(response):
    return re.search(rb"/EP-Classification/psrdownload/([0-9a-f]{32})", response.data).group(1).decode()


@pytest.mark.parametrize("url", ["/EP-Classification/", "/EP-Classification/EDF", "/EP-Classification/CSV",
                                 "/EP-Classification/EPC_help.html", "/EP-Classification/static/EEG_amp.png",
                                 "/EP-Classification/static/Mice_img.png", "/EP-Classification/static/EEG_feature.png", "/EP-Classification/static/EEG_signal.png"])
def test_pages(client, url):
    assert client.get(url).status_code == 200


def test_old_addresses_redirect(client):
    r = client.get("/")
    assert r.status_code == 302 and r.headers["Location"].endswith("/EP-Classification/")
    assert client.get("/EP-Classification/CSV-PICKLE").status_code == 302
    assert client.get("/EP-Classification/EPC_result").status_code == 302


def test_two_channel_edf(client, settings, tmp_path):
    p = write_edf(tmp_path / "rec.edf", {"EEG 1": eeg(60), "EEG 2": eeg(60, seed=1)})
    r = post_file(client, p, format="edf")
    assert r.status_code == 200 and phenotype(r) in ("Normal", "Outlier")
    assert not any(settings.upload_dir.iterdir())                    # upload deleted
    z = client.get(f"/EP-Classification/psrdownload/{result_id(r)}")
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

    same = client.post("/EP-Classification/EPC_result/channels", data={"token": token, "channel1": "1", "channel2": "1"})
    assert same.status_code == 400 and b"two different" in same.data   # asked again, upload kept

    r = client.post("/EP-Classification/EPC_result/channels", data={"token": token, "channel1": "3", "channel2": "1",
                                                  "filename": "rec.edf"})
    assert r.status_code == 200 and phenotype(r)
    assert re.search(rb"EEG1 \(ipsilateral\)</th><td>C</td>", r.data)
    assert re.search(rb"EEG2 \(contralateral\)</th><td>A</td>", r.data)
    assert not any(settings.upload_dir.iterdir())


def test_channel_choice_rejects_wrong_rate(client, tmp_path):
    p = write_edf(tmp_path / "rec.edf", {"Temp": eeg(60, fs=10), "A": eeg(60), "B": eeg(60, seed=1)},
                  fs={"Temp": 10, "A": 500, "B": 500})
    r = post_file(client, p, format="edf")
    token = re.search(rb'name="token" value="([0-9a-f]{32})"', r.data).group(1).decode()
    r = client.post("/EP-Classification/EPC_result/channels", data={"token": token, "channel1": "0", "channel2": "1"})
    assert r.status_code == 400 and b"500 Hz recordings only" in r.data


@pytest.mark.parametrize("token", ["", "../../etc", "0" * 32])
def test_expired_or_invalid_token(client, token):
    r = client.post("/EP-Classification/EPC_result/channels", data={"token": token, "channel1": "0", "channel2": "1"})
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
    r = client.post("/EP-Classification/EPC_result", data={"format": "edf"}, content_type="multipart/form-data")
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
    assert client.get("/EP-Classification/psrexample/").status_code == 404
    assert client.get("/EP-Classification/psrdownload/" + "0" * 32).status_code == 404
    assert client.get("/EP-Classification/psrdownload/not-a-token").status_code == 404


def test_samples_served_when_present(client, settings):
    settings.samples_dir.mkdir()
    (settings.samples_dir / "Mice_example.zip").write_bytes(b"PK")
    assert client.get("/EP-Classification/psrexample/").status_code == 200


def test_model_never_written(client, tmp_path):
    p = write_edf(tmp_path / "rec.edf", {"A": eeg(30), "B": eeg(30, seed=1)})
    post_file(client, p, format="edf")
    assert mdl.sha256_file(MODEL_PATH) == mdl.MODEL_SHA256


def test_everything_is_under_the_prefix(client):
    page = client.get("/EP-Classification/EDF").data
    for link in re.findall(rb'(?:href|src|action)="([^"#]+)"', page):
        if link.startswith((b"https://fonts.", b"data:")):              # web fonts and the inline icon
            continue
        assert link.startswith(b"/EP-Classification/"), link
    assert client.get("/static/epc.css").status_code == 404
    assert client.get("/EPC_result").status_code == 404


def test_prefix_stripped_by_proxy(settings, model):
    """EPC_URL_PREFIX empty: the proxy strips the path and sends X-Forwarded-Prefix."""
    from dataclasses import replace
    from epc.server import create_app
    client = create_app(replace(settings, url_prefix=""), model).test_client()
    r = client.get("/EDF", headers={"X-Forwarded-Prefix": "/EP-Classification"})
    assert r.status_code == 200
    assert b'action="/EP-Classification/EPC_result"' in r.data


def test_result_page_content(client, tmp_path):
    p = write_edf(tmp_path / "rec.edf", {"EEG 1": eeg(60), "EEG 2": eeg(60, seed=1)})
    r = post_file(client, p, format="edf")
    html = r.data
    assert b'class="vote-fill"' in html and b"five-second windows" in html
    assert html.count(b'<rect class="tl-') == 12                       # one block per 5 s window
    assert b"Short recording" in html                                  # 1 minute: Q02 warning
    assert b"Research aid" in html and mdl.MODEL_SHA256[:8].encode() in html
    rid = result_id(r)
    for name in ("EEG1.jpg", "EEG2.jpg"):
        img = client.get(f"/EP-Classification/psrfigure/{rid}/{name}")
        assert img.status_code == 200 and img.mimetype == "image/jpeg"
    assert client.get(f"/EP-Classification/psrfigure/{rid}/Feature_analysis.zip").status_code == 404
    assert client.get(f"/EP-Classification/psrfigure/{rid}/..%2F..%2Fsecret").status_code == 404


def test_unit_warning_shown(client, tmp_path):
    a, b = eeg(60) * 1e6, eeg(60, seed=1) * 1e6                        # µV values, declared as V
    p = tmp_path / "rec.csv"
    p.write_text("".join(f"{x},{y}\n" for x, y in zip(a, b)))
    r = post_file(client, p, format="csv", sampling_frequency="500", unit="V")
    assert r.status_code == 200 and b"Signal unit may be wrong" in r.data

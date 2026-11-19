"""Model loading: checksum guard, read-only use and the JSON copy."""
import numpy as np
import pytest

from conftest import MODEL_PATH, needs_model
from epc import model as mdl


def test_refuses_model_with_wrong_checksum(tmp_path):
    fake = tmp_path / "Mice.model"
    fake.write_bytes(b"not the model")
    with pytest.raises(mdl.ModelChecksumError):
        mdl.load_model(fake)


def test_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        mdl.load_model(tmp_path / "absent.model")


@needs_model
def test_model_file_unchanged():
    assert mdl.sha256_file(MODEL_PATH) == mdl.MODEL_SHA256


@needs_model
def test_json_copy_predicts_identically(model):
    json_path = MODEL_PATH.with_suffix(".json")
    if not json_path.is_file():
        pytest.skip("Mice.json not available")
    from xgboost import XGBClassifier
    copy = XGBClassifier()
    copy.load_model(str(json_path))
    X = np.random.default_rng(0).lognormal(-16, 2, (5000, 4))
    np.testing.assert_array_equal(model.predict_proba(X), copy.predict_proba(X))


@needs_model
def test_classify_majority_rule(model):
    X = np.random.default_rng(1).lognormal(-16, 2, (101, 4))
    label, share, window_class = mdl.classify(model, X)
    assert share == window_class.mean()
    assert label == ("Outlier" if share >= 0.5 else "Normal")

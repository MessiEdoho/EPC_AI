"""Settings, read from environment variables so the same code runs locally and on a server."""
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_PATH = r"C:\Users\messi\EPC_AI_Model\Mice.model"


@dataclass(frozen=True)
class Settings:
    model_path: Path       # XGBoost model, opened read-only and checked against its SHA-256
    upload_dir: Path       # temporary uploads, deleted after each analysis
    results_dir: Path      # one folder per analysis, holding its result zip
    samples_dir: Path      # sample recordings offered for download (may be empty)
    max_upload_mb: int     # larger uploads are refused
    keep_hours: float      # results and unfinished uploads older than this are deleted


def load_settings():
    root = Path(__file__).resolve().parents[1]
    tmp = Path(tempfile.gettempdir())
    env = os.environ.get
    return Settings(
        model_path=Path(env("EPC_MODEL_PATH", DEFAULT_MODEL_PATH)),
        upload_dir=Path(env("EPC_UPLOAD_DIR", tmp / "epc_uploads")),
        results_dir=Path(env("EPC_RESULTS_DIR", tmp / "epc_results")),
        samples_dir=Path(env("EPC_SAMPLES_DIR", root / "samples")),
        max_upload_mb=int(env("EPC_MAX_UPLOAD_MB", "100")),
        keep_hours=float(env("EPC_KEEP_HOURS", "3")),
    )

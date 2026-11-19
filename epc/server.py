"""EPC-AI web server.

Run locally:  python epc_app.py   (then open http://localhost:8000/EP-Classification/)

Every page and file is served under settings.url_prefix (default /EP-Classification), so the
app works the same directly and behind a reverse proxy that forwards that path unchanged. If
the proxy strips the path instead, set EPC_URL_PREFIX to an empty value and have the proxy send
X-Forwarded-Prefix; links then still point to the public address.
"""
import re
import shutil
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, redirect, render_template, request, send_file, url_for
from markupsafe import Markup
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import redirect as wz_redirect
from werkzeug.wrappers import Response

from epc import __version__, quality, reports, validation
from epc import features as feat
from epc import model as mdl
from epc.config import load_settings
from epc.validation import ValidationError

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"^[0-9a-f]{32}$")
SAMPLES_ZIP = "Mice_example.zip"
BORDERLINE = (0.40, 0.60)          # share of outlier windows shown as a borderline result


def create_app(settings=None, model=None):
    """Build the app. The model is loaded once here; a checksum mismatch stops start-up."""
    settings = settings or load_settings()
    model = model if model is not None else mdl.load_model(settings.model_path)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024
    app.config["EPC_SETTINGS"] = settings

    def sweep():
        """Delete unfinished uploads and results older than the keep time.

        Age-based, so it is safe with several server processes sharing the folders.
        """
        cutoff = time.time() - settings.keep_hours * 3600
        for folder in (settings.upload_dir, settings.results_dir):
            for p in folder.iterdir():
                try:
                    old = p.stat().st_mtime < cutoff
                except OSError:                              # removed by another process
                    continue
                if old:
                    _remove(p)

    sweep()                                                  # leftovers from a previous run

    def error_page(message, back="psr_index", status=400):
        return render_template("EPC_error.html", message=message, back=back), status

    def analyse(eeg1, eeg2, channel_labels, filename, kind):
        table = feat.extract_features(eeg1, eeg2)
        label, share, window_class, window_prob = mdl.classify(model, table)
        result_id = uuid.uuid4().hex
        folder = settings.results_dir / result_id
        folder.mkdir()
        reports.write_results(folder, table, window_class, window_prob, eeg1, eeg2, channel_labels)
        now = datetime.now().astimezone()
        return render_template(
            "EPC_result.html",
            prediction_text=label,
            share=share,
            borderline=BORDERLINE[0] <= share <= BORDERLINE[1],
            n_windows=len(window_class),
            n_outlier=int(window_class.sum()),
            timeline=Markup(reports.timeline_svg(window_class, window_prob, len(eeg1))),
            warnings=quality.check(eeg1, eeg2),
            result_id=result_id,
            filename=filename,
            file_format=kind.upper(),
            channels=channel_labels,
            minutes=len(eeg1) / feat.FS / 60,
            fs=feat.FS,
            analysed=_stamp(now, "%Y-%m-%d %H:%M"),
            expires=_stamp(now + timedelta(hours=settings.keep_hours), "%H:%M"),
            model_id=mdl.MODEL_SHA256[:8],
            version=__version__,
            again="psr_indexcsv" if kind == "csv" else "psr_indexedf",
        )

    def save_upload(file, kind):
        path = settings.upload_dir / f"{uuid.uuid4().hex}.{kind}"
        file.save(path)
        return path

    # ------------------------------------------------------------------ pages

    @app.route("/")
    def psr_index():
        return render_template("EPC_home.html")

    @app.route("/EDF")
    def psr_indexedf():
        return render_template("EPC_edf.html", max_mb=settings.max_upload_mb)

    @app.route("/CSV")
    def psr_indexcsv():
        return render_template("EPC_csv.html", max_mb=settings.max_upload_mb, fs=feat.FS,
                               units=validation.CSV_UNITS)

    @app.route("/CSV-PICKLE")
    def psr_indexcsv_old():                                  # old address, kept for bookmarks
        return redirect(url_for("psr_indexcsv"))

    @app.route("/EPC_help.html")
    def psr_help():
        return render_template("EPC_help.html")

    # ------------------------------------------------------------------ analysis

    @app.route("/EPC_result", methods=["GET", "POST"])
    def psr_transform_view():
        if request.method == "GET":
            return redirect(url_for("psr_index"))
        sweep()
        back = "psr_indexcsv" if request.form.get("format") == "csv" else "psr_indexedf"
        file = request.files.get("eeg_file")
        try:
            kind = validation.file_kind(file.filename if file else "")
        except ValidationError as e:
            return error_page(e.message, back)
        back = "psr_indexcsv" if kind == "csv" else "psr_indexedf"
        path = save_upload(file, kind)
        keep_upload = False
        try:
            if kind == "csv":
                eeg1, eeg2, labels = validation.load_csv(
                    path, request.form.get("sampling_frequency"), request.form.get("unit"))
            else:
                channels = validation.edf_channels(path)
                if validation.needs_channel_choice(channels):
                    keep_upload = True
                    return render_template("EPC_channels.html", token=path.stem,
                                           channels=channels, filename=file.filename, fs=feat.FS)
                eeg1, eeg2, labels = validation.load_edf(path)
            return analyse(eeg1, eeg2, labels, file.filename, kind)
        except ValidationError as e:
            return error_page(e.message, back)
        finally:
            if not keep_upload:
                _remove(path)

    @app.route("/EPC_result/channels", methods=["POST"])
    def psr_channels():
        token = request.form.get("token", "")
        path = settings.upload_dir / f"{token}.edf"
        if not TOKEN.match(token) or not path.is_file():
            return error_page("This upload has expired. Please upload the EDF file again.", "psr_indexedf")
        filename = request.form.get("filename", "")
        keep_upload = False
        try:
            eeg1, eeg2, labels = validation.load_edf(
                path, (request.form.get("channel1"), request.form.get("channel2")))
            return analyse(eeg1, eeg2, labels, filename, "edf")
        except ValidationError as e:
            if e.code == "V04":                              # wrong choice: let the user choose again
                keep_upload = True
                return render_template("EPC_channels.html", token=token, error=e.message,
                                       channels=validation.edf_channels(path),
                                       filename=filename, fs=feat.FS), 400
            return error_page(e.message, "psr_indexedf")
        finally:
            if not keep_upload:
                _remove(path)

    # ------------------------------------------------------------------ downloads

    @app.route("/psrdownload/<result_id>")
    def psr_downloadFigure(result_id):
        path = settings.results_dir / result_id / reports.ZIP_NAME
        if not TOKEN.match(result_id) or not path.is_file():
            return error_page(f"This result is no longer available. Results are kept for "
                              f"{settings.keep_hours:g} hours; please run the analysis again.", status=404)
        return send_file(path, as_attachment=True, download_name=reports.ZIP_NAME)

    @app.route("/psrfigure/<result_id>/<name>")
    def psr_figure(result_id, name):
        path = settings.results_dir / result_id / name
        if not TOKEN.match(result_id) or name not in reports.FIGURES or not path.is_file():
            return error_page("This figure is no longer available.", status=404)
        return send_file(path, mimetype="image/jpeg")

    @app.route("/psrexample/")
    def psr_downloadexample():
        path = settings.samples_dir / SAMPLES_ZIP
        if not path.is_file():
            return error_page("Sample EEG files are not available on this server yet. Please "
                              "contact us for example recordings.", status=404)
        return send_file(path, as_attachment=True, download_name=SAMPLES_ZIP)

    @app.errorhandler(413)
    def too_large(_):
        return error_page(f"The file is larger than {settings.max_upload_mb} MB. Please contact us "
                          "to analyse larger recordings.", status=413)

    @app.errorhandler(404)
    def not_found(_):
        return error_page("Page not found.", status=404)

    return _mount(app, settings.url_prefix)


def _mount(app, prefix):
    """Serve the app under `prefix`; the bare server root redirects there."""
    if prefix:
        def outside(environ, start_response):
            if environ.get("PATH_INFO", "") in ("", "/"):
                target = environ.get("SCRIPT_NAME", "") + prefix + "/"
                return wz_redirect(target)(environ, start_response)
            return Response("Not found", status=404)(environ, start_response)
        app.wsgi_app = DispatcherMiddleware(outside, {prefix: app.wsgi_app})
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    return app


def _stamp(moment, fmt):
    """Local time with its UTC offset, e.g. '15:43 (UTC+00:00)', the same on Windows and Linux."""
    offset = moment.strftime("%z")
    return f"{moment.strftime(fmt)} (UTC{offset[:3]}:{offset[3:]})"


def _remove(path):
    path = Path(path)
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass                                                 # removed later by the sweep

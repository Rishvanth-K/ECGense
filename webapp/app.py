"""ECGense - Flask web app for ECG heartbeat classification (CNN-BiLSTM)."""

import os
import socket

from flask import Flask, jsonify, render_template, request

from webapp import inference

app = Flask(__name__)
app.json.sort_keys = False  # keep class order N, S, V, F, Q in responses
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB uploads

SAMPLES = inference.load_samples()


def build_info():
    """Values passed in by Jenkins at docker build time (see Dockerfile)."""
    return {
        "build_number": os.environ.get("BUILD_NUMBER", "dev"),
        "git_commit": os.environ.get("GIT_COMMIT", "local")[:7],
        "pod": socket.gethostname(),  # in Kubernetes this is the pod name
    }


@app.get("/")
def index():
    return render_template("index.html", info=build_info())


@app.get("/health")
def health():
    """Used by the Kubernetes probes and the Jenkins Verify stage."""
    inference.get_model()
    return jsonify(status="ok", model_loaded=True, **build_info())


@app.get("/api/samples")
def samples():
    return jsonify(SAMPLES)


@app.post("/api/predict")
def predict():
    labels = None
    try:
        if request.is_json:
            # {"sample_id": 3}  or  {"signal": [187 numbers]}
            body = request.get_json()
            if "sample_id" in body:
                s = SAMPLES[int(body["sample_id"])]
                signals, labels = [s["signal"]], [s["true_class"]]
            else:
                signal = body.get("signal", [])
                if len(signal) != inference.SIGNAL_LENGTH:
                    raise ValueError(f"signal must have {inference.SIGNAL_LENGTH} values")
                signals = [signal]
        elif "file" in request.files:
            f = request.files["file"]
            name = (f.filename or "").lower()
            raw = f.read()
            if name.endswith((".png", ".jpg", ".jpeg")):
                signals = [inference.extract_signal_from_image(raw)]
            elif name.endswith((".csv", ".txt")):
                signals, labels = inference.parse_csv(raw.decode("utf-8", errors="ignore"))
            else:
                raise ValueError("Upload a .csv file or an ECG image (.png/.jpg)")
        else:
            raise ValueError("No input given")
    except (ValueError, IndexError, KeyError) as exc:
        return jsonify(error=str(exc)), 400

    results = inference.classify(signals)
    for r, lab in zip(results, labels or [None] * len(results)):
        r["true_label"] = inference.CLASS_NAMES.get(lab) if lab is not None else None
    return jsonify(results=results)


if __name__ == "__main__":
    # Local development only; the container uses gunicorn (see Dockerfile)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

"""ECGense tests. These use the REAL trained model (ecg_cnn_bilstm.keras), not a mock."""

import io

import cv2
import numpy as np
import pytest

from webapp.app import SAMPLES, app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_home_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"ECGense" in r.data


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["model_loaded"] is True


def test_samples_cover_all_five_classes(client):
    labels = {s["true_label"] for s in client.get("/api/samples").get_json()}
    assert labels == {"Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"}


def test_prediction_output_is_valid(client):
    r = client.post("/api/predict", json={"sample_id": 0})
    assert r.status_code == 200
    res = r.get_json()["results"][0]
    assert res["predicted_label"] in res["probabilities"]
    assert sum(res["probabilities"].values()) == pytest.approx(100, abs=0.1)
    assert len(res["signal"]) == 187


def test_model_classifies_sample_beats(client):
    """Sanity check that the model file is the trained one (a broken or untrained
    model scores far below this on 20 class-balanced beats)."""
    csv = "\n".join(",".join(map(str, s["signal"] + [s["true_class"]])) for s in SAMPLES)
    r = client.post("/api/predict", data={"file": (io.BytesIO(csv.encode()), "beats.csv")})
    results = r.get_json()["results"]
    correct = sum(x["predicted_label"] == x["true_label"] for x in results)
    assert correct / len(results) >= 0.7


def test_image_upload(client):
    """Draw a real sample beat as a dark line on white, upload it as PNG."""
    sig = np.array(SAMPLES[0]["signal"])
    img = np.full((200, 600), 255, np.uint8)
    pts = np.stack([np.linspace(0, 599, 187), 190 - sig * 180], axis=1).astype(np.int32)
    cv2.polylines(img, [pts], False, 0, 2)
    png = cv2.imencode(".png", img)[1].tobytes()
    r = client.post("/api/predict", data={"file": (io.BytesIO(png), "ecg.png")})
    assert r.status_code == 200
    assert len(r.get_json()["results"]) == 1


def test_wrong_csv_length_is_rejected(client):
    r = client.post("/api/predict", data={"file": (io.BytesIO(b"0.1,0.2,0.3"), "bad.csv")})
    assert r.status_code == 400


def test_unsupported_file_type_is_rejected(client):
    r = client.post("/api/predict", data={"file": (io.BytesIO(b"hello"), "notes.pdf")})
    assert r.status_code == 400

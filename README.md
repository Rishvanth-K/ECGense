# ECGense

AI-based ECG heartbeat analysis. A Flask website serves the trained hybrid CNN–BiLSTM model
(`ecg_cnn_bilstm.keras`, MIT-BIH data, 5 classes), deployed through a Jenkins CI/CD pipeline.

```
Developer → Git → GitHub → Jenkins → Test → Docker Build → Kubernetes (Kind) → ECGense running
```

| Path | Purpose |
|---|---|
| `webapp/` | Flask app (`app.py`), model loading and preprocessing (`inference.py`), HTML/CSS/JS |
| `samples/sample_beats.csv` | 20 real beats from `mitbih_test.csv` (4 per class), used by the demo and the tests |
| `tests/` | pytest tests that run against the real trained model |
| `Dockerfile` | Container image for the web app |
| `k8s/` | Kubernetes Namespace, Deployment (2 replicas) and NodePort Service |
| `Jenkinsfile` | Pipeline: Checkout → Test → Build Docker Image → Deploy to Kubernetes → Verify Deployment |
| `ecg_classifier.py` | Original training + command-line app (with email alerts) |

## Run locally (without Docker)

```bash
python3.12 -m venv .venv-web && source .venv-web/bin/activate
pip install -r webapp/requirements.txt
pytest -v tests/
python -m webapp.app          # http://localhost:5000
```

## Open the deployed app

```bash
kubectl -n ecgense get pods
kubectl -n ecgense port-forward svc/ecgense 8081:80   # then open http://localhost:8081
```

Educational project. Not a medical device.

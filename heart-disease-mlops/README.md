# Heart Disease Prediction — End-to-End MLOps

Predict the risk of heart disease from patient health data and serve the model as a
containerised, monitored API. This repo covers the full MLOps lifecycle: EDA →
modelling → experiment tracking → packaging → CI/CD → containerisation →
Kubernetes deployment → monitoring.

> **Dataset:** Heart Disease UCI Dataset (303 records, 13 features + binary target).
> **Best model in the reference run:** Random Forest, test ROC-AUC ≈ 0.92.

---

## 1. Project structure

```
heart-disease-mlops/
├── data/                     # dataset (downloaded)
├── scripts/
│   └── download_data.py      # UCI download (ucimlrepo) + GitHub mirror fallback
├── src/
│   ├── config.py             # paths, feature schema, training config
│   ├── data_preprocessing.py # load + ColumnTransformer pipeline
│   ├── eda.py                # EDA figures
│   ├── train.py              # train/tune 2 models, MLflow logging, save best
│   └── predict.py            # CLI/batch inference
├── app/
│   ├── main.py               # Flask API: /predict /health /metrics
│   └── schemas.py            # request validation
├── tests/                    # pytest (data + API)
├── models/                   # saved pipeline + metadata (created by training)
├── reports/figures/          # EDA + evaluation plots (created by EDA/training)
├── k8s/                      # Deployment, Service, HPA
├── monitoring/               # Prometheus + Grafana (docker-compose)
├── .github/workflows/ci.yml  # CI/CD pipeline
├── Dockerfile
├── requirements.txt
├── pyproject.toml            # pytest + flake8 config
├── Makefile
└── docs/report_template.md   # fill in for your submission
```

## 2. Setup

Requires **Python 3.12** (any 3.10+ works). Docker + kubectl for the later stages.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run the pipeline (step by step)

```bash
# 1. Download the dataset -> data/heart.csv
python scripts/download_data.py          # or: make data

# 2. Exploratory Data Analysis -> reports/figures/*.png
python -m src.eda                        # or: make eda

# 3. Train + tune both models, log to MLflow, save best pipeline
python -m src.train                      # or: make train

# 4. Inspect experiments in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
#   open http://127.0.0.1:5000

# 5. Run the tests
pytest -v                                # or: make test
```

## 4. Serve the API locally

```bash
# development
python -m app.main
# production
gunicorn -w 2 -b 0.0.0.0:8000 app.main:app     # or: make run
```

Test it:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Example response:

```json
{
  "prediction": 1,
  "label": "disease",
  "confidence": 0.6345,
  "probabilities": {"no_disease": 0.3655, "disease": 0.6345}
}
```

Offline inference without the server:

```bash
python -m src.predict --json "$(cat sample_request.json)"
python -m src.predict --csv data/new_patients.csv
```

## 5. Docker

```bash
docker build -t heart-disease-api:latest .        # or: make docker-build
docker run --rm -p 8000:8000 heart-disease-api:latest
curl http://localhost:8000/health
```

> The image bundles the trained model, so **train before building** (step 3),
> or let CI train inside the build.

## 6. Kubernetes deployment

```bash
# Minikube example
minikube start
minikube image load heart-disease-api:latest     # push local image to the cluster
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml        # optional autoscaling

kubectl get pods,svc
minikube service heart-disease-api   # opens the LoadBalancer URL
```

For a cloud cluster (GKE/EKS/AKS): push the image to a registry, set that image
in `k8s/deployment.yaml`, then `kubectl apply`. The `LoadBalancer` service gives
you an external IP; verify `/health` and `/predict` against it and screenshot.

## 7. Monitoring (Prometheus + Grafana)

```bash
docker compose -f monitoring/docker-compose.yml up --build   # or: make monitor
```

- API: http://localhost:8000
- Prometheus: http://localhost:9090 (query `api_requests_total`, `predictions_total`)
- Grafana: http://localhost:3000 (admin / admin) — Prometheus datasource is
  pre-provisioned; build a panel on `rate(api_requests_total[1m])` and
  `api_request_latency_seconds`.

Exposed metrics: `api_requests_total`, `api_request_latency_seconds`,
`predictions_total`. Every request is also logged as a structured line to stdout.

## 8. CI/CD

`.github/workflows/ci.yml` runs on push/PR to `main`:

1. **lint** (flake8 — build fails on real errors)
2. **test** (pytest — build fails on any failure)
3. **train** and upload model + figures as artifacts
4. **docker-build** and smoke-test the container's `/health` and `/predict`

## 9. Configuration

Edit `src/config.py` to change the feature split, test size, CV folds,
MLflow experiment name, or tracking backend. Keeping it central means training
and serving never drift apart.

## 10. Notes on reproducibility

- Fixed `random_state` everywhere (`src/config.py`).
- Preprocessing lives *inside* the saved pipeline, so inference applies the exact
  same transforms as training.
- `requirements.txt` is fully pinned.
- `model_metadata.json` records metrics, feature lists, and library versions.

---

### Swapping Flask for FastAPI (optional)

The API is Flask (assignment allows either; FAQ prefers FastAPI). To switch,
reimplement `app/main.py` with `FastAPI` + `pydantic` models mirroring
`app/schemas.py`, keep the same `/predict`, `/health`, `/metrics` contract, and
run with `uvicorn app.main:app`. The rest of the repo is unchanged.

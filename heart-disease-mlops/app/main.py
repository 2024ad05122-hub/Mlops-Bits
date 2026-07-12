"""
Flask model-serving API for heart-disease prediction.

Endpoints
---------
GET  /health   -> liveness/readiness probe + model metadata
POST /predict  -> JSON in, {prediction, label, confidence, probabilities} out
GET  /metrics  -> Prometheus metrics (request count, latency, prediction count)
GET  /         -> minimal usage help

Covers assignment tasks 6 (containerisable API) and 8 (logging + monitoring).

Run locally:
    python -m app.main
    # or, production:  gunicorn -w 2 -b 0.0.0.0:8000 app.main:app
"""
from __future__ import annotations

import json
import logging
import time

import joblib
import pandas as pd
from flask import Flask, g, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from app.schemas import ValidationError, validate_payload
from src import config

# --------------------------------------------------------------------------- #
# Logging – structured, one line per request (task 8)
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart-api")

# --------------------------------------------------------------------------- #
# Prometheus metrics (task 8)
# --------------------------------------------------------------------------- #
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency (s)", ["endpoint"],
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Predictions served by class", ["predicted_label"],
)

LABELS = {0: "no_disease", 1: "disease"}


def create_app(model_path=None) -> Flask:
    app = Flask(__name__)
    model_path = model_path or config.MODEL_PATH

    # Load model + metadata once at startup
    app.config["MODEL"] = joblib.load(model_path)
    try:
        with open(config.METADATA_PATH) as f:
            app.config["METADATA"] = json.load(f)
    except FileNotFoundError:
        app.config["METADATA"] = {}
    logger.info("Model loaded from %s", model_path)

    # ---- request timing + logging ----
    @app.before_request
    def _start_timer():
        g.start = time.perf_counter()

    @app.after_request
    def _log_request(response):
        latency = time.perf_counter() - getattr(g, "start", time.perf_counter())
        endpoint = request.path
        REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
        REQUEST_LATENCY.labels(endpoint).observe(latency)
        logger.info(
            "method=%s path=%s status=%s latency_ms=%.1f",
            request.method, endpoint, response.status_code, latency * 1000,
        )
        return response

    # ---- routes ----
    @app.get("/")
    def index():
        return jsonify(
            service="heart-disease-classifier",
            usage="POST /predict with a JSON object of the 13 features",
            features=config.FEATURE_DESCRIPTIONS,
            endpoints=["/health", "/predict", "/metrics"],
        )

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            model_loaded=app.config["MODEL"] is not None,
            model=app.config["METADATA"].get("best_model", "unknown"),
        )

    @app.post("/predict")
    def predict():
        try:
            payload = request.get_json(force=True, silent=False)
        except Exception:  # noqa: BLE001
            return jsonify(error="Body must be valid JSON."), 400

        try:
            clean = validate_payload(payload)
        except ValidationError as exc:
            return jsonify(error=str(exc)), 400

        X = pd.DataFrame([clean], columns=config.ALL_FEATURES)
        model = app.config["MODEL"]
        proba = model.predict_proba(X)[0]
        pred = int(proba.argmax())
        confidence = float(proba[pred])

        PREDICTION_COUNT.labels(LABELS[pred]).inc()

        return jsonify(
            prediction=pred,
            label=LABELS[pred],
            confidence=round(confidence, 4),
            probabilities={
                "no_disease": round(float(proba[0]), 4),
                "disease": round(float(proba[1]), 4),
            },
        )

    @app.get("/metrics")
    def metrics():
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    return app


# Module-level app for gunicorn / flask run
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

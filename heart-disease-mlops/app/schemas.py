"""Request validation for the prediction API."""
from __future__ import annotations

from src import config


class ValidationError(Exception):
    """Raised when the incoming JSON payload is invalid."""


def validate_payload(payload) -> dict:
    """Validate a single-record prediction payload.

    Expects a JSON object with every feature in config.ALL_FEATURES.
    Returns a clean dict of {feature: float}. Raises ValidationError otherwise.
    """
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")

    missing = [f for f in config.ALL_FEATURES if f not in payload]
    if missing:
        raise ValidationError(f"Missing required features: {missing}")

    unknown = [k for k in payload if k not in config.ALL_FEATURES]
    if unknown:
        raise ValidationError(f"Unknown fields not allowed: {unknown}")

    clean = {}
    for f in config.ALL_FEATURES:
        try:
            clean[f] = float(payload[f])
        except (TypeError, ValueError):
            raise ValidationError(
                f"Feature '{f}' must be numeric ({config.FEATURE_DESCRIPTIONS[f]})."
            )
    return clean

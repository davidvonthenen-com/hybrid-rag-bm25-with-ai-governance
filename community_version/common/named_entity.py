"""Named entity recognition helpers."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from .config import Settings, load_settings
from .logging import get_logger


LOGGER = get_logger(__name__)


def post_ner(text: str, settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Call the configured NER service with the given text."""

    if settings is None:
        settings = load_settings()
    url = settings.ner_url
    timeout = settings.ner_timeout_secs

    headers = {"Content-Type": "application/json"}
    payload = {"text": text}

    LOGGER.info("Requesting NER for %d characters from %s", len(text), url)
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        try:
            msg: Any = resp.json()
        except Exception:
            msg = resp.text
        LOGGER.error("NER HTTP %s: %s", resp.status_code, msg)
        raise exc

    try:
        data: Dict[str, Any] = resp.json()
    except Exception:
        LOGGER.error("NER non-JSON response (first 800 chars): %s", resp.text[:800])
        raise

    return data


def normalize_entities(ner_result: Dict[str, Any]) -> List[str]:
    """Return normalized (lowercased, de-duplicated) entity strings."""

    seen = set()
    out: List[str] = []

    for raw in ner_result.get("entities", []):
        if not isinstance(raw, str):
            continue
        name = raw.strip().lower()
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)

    return out


def extract_entities(question: str) -> List[str]:
    try:
        ner_payload = post_ner(question)
        return normalize_entities(ner_payload)
    except Exception as exc:
        LOGGER.warning("NER failed (%s). Proceeding without entities.", exc)
        return []


__all__ = ["post_ner", "normalize_entities", "extract_entities"]

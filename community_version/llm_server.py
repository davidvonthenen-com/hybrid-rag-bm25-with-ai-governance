"""OpenAI-compatible REST server that fronts a local llama.cpp model."""
from __future__ import annotations

import os
import platform
import time
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request
from llama_cpp import Llama

from common.config import Settings, load_settings
from common.logging import get_logger

LOGGER = get_logger(__name__)


def _is_apple_silicon() -> bool:
    """Return True when running on Apple Silicon hardware."""

    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _resolve_gpu_layers(settings: Settings) -> int:
    """Pick an appropriate GPU offload value for the local runtime."""

    n_gpu_layers = settings.llama_n_gpu_layers
    if _is_apple_silicon() and n_gpu_layers == 0:
        return -1
    return n_gpu_layers


@lru_cache(maxsize=1)
def _load_local_llm(settings: Optional[Settings] = None) -> Llama:
    """Load and cache the local llama.cpp model for serving."""
    if settings is None:
        settings = load_settings()

    model_path = os.path.expanduser(settings.llama_model_path)
    LOGGER.info("Loading LLaMA model from %s", model_path)

    n_gpu_layers = _resolve_gpu_layers(settings)
    if _is_apple_silicon() and n_gpu_layers != 0:
        LOGGER.info(
            "Apple Silicon detected; using Metal GPU offload with n_gpu_layers=%s.",
            n_gpu_layers,
        )

    kwargs: Dict[str, Any] = {
        "model_path": model_path,
        "n_ctx": settings.llama_ctx,
        "n_threads": settings.llama_n_threads,
        "n_gpu_layers": n_gpu_layers,
        "n_batch": settings.llama_n_batch,
        "chat_format": "chatml",
        "verbose": False,
    }

    if settings.llama_n_ubatch is not None:
        kwargs["n_ubatch"] = settings.llama_n_ubatch
    if settings.llama_low_vram:
        kwargs["low_vram"] = True

    return Llama(**kwargs)


def _error(status: int, message: str) -> tuple[Dict[str, Any], int]:
    """Return a JSON API error payload."""
    return {"error": {"message": message, "type": "invalid_request_error"}}, status


def _normalize_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Validate and normalize chat messages from the request body."""
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected non-empty 'messages' list.")
    normalized: List[Dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("Each message must be a JSON object.")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Each message requires string 'role' and 'content'.")
        normalized.append({"role": role, "content": content})
    return normalized


def _build_chat_response(
    *,
    model: str,
    content: str,
    usage: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Format the response payload to match the OpenAI chat completion schema."""
    payload: Dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if usage:
        payload["usage"] = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
    return payload


def create_app() -> Flask:
    """Create the Flask app that serves OpenAI-compatible endpoints."""
    app = Flask(__name__)
    settings = load_settings()
    _load_local_llm(settings)

    @app.route("/health", methods=["GET"])
    def health() -> tuple[Dict[str, Any], int]:
        return jsonify(
            {
                "status": "ok",
                "model": settings.llm_server_model,
                "server": {
                    "host": os.getenv("LLM_SERVER_HOST", "0.0.0.0"),
                    "port": int(os.getenv("LLM_SERVER_PORT", "8001")),
                },
            }
        ), 200

    @app.route("/v1/models", methods=["GET"])
    def models() -> tuple[Dict[str, Any], int]:
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": settings.llm_server_model,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                    }
                ],
            }
        ), 200

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions() -> tuple[Dict[str, Any], int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error(400, "Expected JSON object payload.")

        if payload.get("stream") is True:
            return _error(400, "Streaming responses are not supported.")

        try:
            messages = _normalize_messages(payload)
        except ValueError as exc:
            return _error(400, str(exc))

        temperature = float(payload.get("temperature", 0.2))
        top_p = float(payload.get("top_p", 0.9))
        max_tokens = int(payload.get("max_tokens", 512))
        model = str(payload.get("model") or settings.llm_server_model)

        llm = _load_local_llm(settings)
        response = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        content = ""
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                content = str(message.get("content") or "")
        usage = response.get("usage") if isinstance(response, dict) else None
        return jsonify(_build_chat_response(model=model, content=content, usage=usage)), 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        host=os.getenv("LLM_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("LLM_SERVER_PORT", "8001")),
        debug=False,
    )

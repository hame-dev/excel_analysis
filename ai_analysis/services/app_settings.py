"""Runtime-configurable settings for prompts and Ollama connectivity."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from django.conf import settings
from django.utils import timezone

from ai_analysis.models import RuntimeConfig


@dataclass
class RuntimeSettings:
    ollama_base_url: str
    ollama_qa_model: str
    ollama_chat_model: str
    ollama_embed_model: str
    custom_prompt: str = ""


def _default_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        ollama_base_url=str(settings.OLLAMA_BASE_URL),
        ollama_qa_model=str(settings.OLLAMA_QA_MODEL),
        ollama_chat_model=str(settings.OLLAMA_CHAT_MODEL),
        ollama_embed_model=str(settings.OLLAMA_EMBED_MODEL),
        custom_prompt="",
    )


def get_runtime_config() -> RuntimeConfig:
    defaults = _default_runtime_settings()
    config = RuntimeConfig.objects.order_by("-updated_at", "-id").first()
    if config is None:
        config = RuntimeConfig.objects.create(
            ollama_base_url=defaults.ollama_base_url,
            ollama_qa_model=defaults.ollama_qa_model,
            ollama_chat_model=defaults.ollama_chat_model,
            ollama_embed_model=defaults.ollama_embed_model,
            custom_prompt="",
        )
    return config


def get_runtime_settings() -> RuntimeSettings:
    defaults = _default_runtime_settings()
    config = get_runtime_config()
    return RuntimeSettings(
        ollama_base_url=(config.ollama_base_url or defaults.ollama_base_url).strip(),
        ollama_qa_model=(config.ollama_qa_model or defaults.ollama_qa_model).strip(),
        ollama_chat_model=(config.ollama_chat_model or defaults.ollama_chat_model).strip(),
        ollama_embed_model=(config.ollama_embed_model or defaults.ollama_embed_model).strip(),
        custom_prompt=(config.custom_prompt or "").strip(),
    )


def save_runtime_settings(payload: dict[str, Any]) -> RuntimeConfig:
    defaults = _default_runtime_settings()
    config = get_runtime_config()

    config.ollama_base_url = str(payload.get("ollama_base_url") or defaults.ollama_base_url).strip()
    config.ollama_qa_model = str(payload.get("ollama_qa_model") or defaults.ollama_qa_model).strip()
    config.ollama_chat_model = str(payload.get("ollama_chat_model") or defaults.ollama_chat_model).strip()
    config.ollama_embed_model = str(payload.get("ollama_embed_model") or defaults.ollama_embed_model).strip()
    config.custom_prompt = str(payload.get("custom_prompt") or "").strip()

    if bool(payload.get("mark_verified")):
        config.verified_at = timezone.now()
    config.save()
    return config


def serialize_runtime_config(config: RuntimeConfig | None = None) -> dict[str, Any]:
    if config is None:
        config = get_runtime_config()
    return {
        "id": config.id,
        "ollama_base_url": config.ollama_base_url,
        "ollama_qa_model": config.ollama_qa_model,
        "ollama_chat_model": config.ollama_chat_model,
        "ollama_embed_model": config.ollama_embed_model,
        "custom_prompt": config.custom_prompt,
        "verified_at": config.verified_at.isoformat() if config.verified_at else None,
        "updated_at": config.updated_at.isoformat() if config.updated_at else None,
    }


def _json_request(url: str, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 6) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def verify_ollama_settings(
    ollama_base_url: str,
    ollama_qa_model: str,
    ollama_chat_model: str,
    ollama_embed_model: str,
) -> dict[str, Any]:
    base = str(ollama_base_url or "").strip().rstrip("/")
    qa_model = str(ollama_qa_model or "").strip()
    chat_model = str(ollama_chat_model or "").strip()
    embed_model = str(ollama_embed_model or "").strip()

    if not base or not qa_model or not chat_model or not embed_model:
        return {
            "ok": False,
            "message": "All Ollama fields are required for verification.",
            "missing_models": [],
            "available_models": [],
        }

    try:
        tags_payload = _json_request(f"{base}/api/tags", method="GET", timeout=5)
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "message": f"Could not reach Ollama at {base}: {exc}",
            "missing_models": [],
            "available_models": [],
        }
    except Exception as exc:
        return {
            "ok": False,
            "message": f"Ollama tags check failed: {exc}",
            "missing_models": [],
            "available_models": [],
        }

    available_models = [
        str(item.get("name", "")).strip()
        for item in (tags_payload.get("models") or [])
        if isinstance(item, dict)
    ]
    requested_models = [qa_model, chat_model, embed_model]
    missing_models = [m for m in requested_models if m not in set(available_models)]
    if missing_models:
        return {
            "ok": False,
            "message": "Some selected models are not available in Ollama.",
            "missing_models": missing_models,
            "available_models": available_models,
        }

    try:
        _json_request(
            f"{base}/api/generate",
            method="POST",
            payload={"model": qa_model, "prompt": "Reply exactly with OK", "stream": False},
            timeout=8,
        )
    except Exception as exc:
        return {
            "ok": False,
            "message": f"Model generation test failed for '{qa_model}': {exc}",
            "missing_models": [],
            "available_models": available_models,
        }

    return {
        "ok": True,
        "message": "Ollama connection and model checks passed.",
        "missing_models": [],
        "available_models": available_models,
    }

from __future__ import annotations

import pytest

from ai_analysis.services.app_settings import verify_ollama_settings


def test_verify_ollama_accepts_latest_tag_for_embed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_json_request(url: str, method: str = "GET", payload=None, timeout: int = 6):
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {"name": "qwen3.5:9b"},
                    {"name": "nomic-embed-text:latest"},
                ]
            }
        if url.endswith("/api/generate"):
            return {"response": "OK"}
        raise AssertionError(f"Unexpected URL in test: {url}")

    monkeypatch.setattr("ai_analysis.services.app_settings._json_request", fake_json_request)

    result = verify_ollama_settings(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_qa_model="qwen3.5:9b",
        ollama_chat_model="qwen3.5:9b",
        ollama_embed_model="nomic-embed-text",
    )

    assert result["ok"] is True
    assert result["missing_models"] == []


def test_verify_ollama_reads_model_key_from_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_json_request(url: str, method: str = "GET", payload=None, timeout: int = 6):
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {"model": "qwen3.5:9b"},
                    {"model": "nomic-embed-text:latest"},
                ]
            }
        if url.endswith("/api/generate"):
            return {"response": "OK"}
        raise AssertionError(f"Unexpected URL in test: {url}")

    monkeypatch.setattr("ai_analysis.services.app_settings._json_request", fake_json_request)

    result = verify_ollama_settings(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_qa_model="qwen3.5:9b",
        ollama_chat_model="qwen3.5:9b",
        ollama_embed_model="nomic-embed-text",
    )

    assert result["ok"] is True
    assert result["missing_models"] == []


def test_verify_ollama_does_not_require_generation_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_json_request(url: str, method: str = "GET", payload=None, timeout: int = 6):
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {"name": "qwen3.5:9b"},
                    {"name": "nomic-embed-text:latest"},
                ]
            }
        raise AssertionError(f"Unexpected non-tags request in verify path: {url}")

    monkeypatch.setattr("ai_analysis.services.app_settings._json_request", fake_json_request)

    result = verify_ollama_settings(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_qa_model="qwen3.5:9b",
        ollama_chat_model="qwen3.5:9b",
        ollama_embed_model="nomic-embed-text",
    )

    assert result["ok"] is True
    assert result["missing_models"] == []

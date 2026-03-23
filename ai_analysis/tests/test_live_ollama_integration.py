from __future__ import annotations

import pytest
import requests
from django.conf import settings
from langchain_ollama.llms import OllamaLLM


@pytest.mark.integration
@pytest.mark.django_db
def test_live_ollama_roundtrip_if_available() -> None:
    base_url = settings.OLLAMA_BASE_URL.rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        response.raise_for_status()
    except Exception:
        pytest.skip("Ollama not reachable; skipping live integration test.")

    llm = OllamaLLM(
        model=settings.OLLAMA_QA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.0,
    )
    text = llm.invoke("Reply with exactly this token: OK")
    assert "OK" in str(text).upper()


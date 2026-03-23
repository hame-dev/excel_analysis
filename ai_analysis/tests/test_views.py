from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile

from ai_analysis.models import AgentTraceEvent, ChatMessage, ChatSession, RuntimeConfig, UploadRecord
from ai_analysis.tests.conftest import make_excel_upload


@pytest.mark.django_db
def test_upload_endpoint_and_data_apis(client, disable_vector_build: None) -> None:
    csv_path = Path(settings.BASE_DIR) / "data" / "disoccupazione.csv"
    upload = SimpleUploadedFile(
        name="disoccupazione.csv",
        content=csv_path.read_bytes(),
        content_type="text/csv",
    )
    response = client.post("/api/upload", {"excel_file": upload, "null_strategy": "none"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["upload"]["rows"] > 0

    filters_resp = client.get("/api/data/filters")
    assert filters_resp.status_code == 200
    filters_payload = filters_resp.json()
    assert "filters" in filters_payload
    assert len(filters_payload["filters"]) > 0

    first_categorical = next((f for f in filters_payload["filters"] if f["type"] == "categorical"), None)
    if first_categorical and first_categorical.get("options"):
        filter_payload = {first_categorical["column"]: {"type": "categorical", "values": [first_categorical["options"][0]]}}
    else:
        filter_payload = {}

    query_resp = client.get("/api/data/query", {"filters": json.dumps(filter_payload)})
    assert query_resp.status_code == 200
    query_payload = query_resp.json()
    assert query_payload["total"] >= 0
    assert "charts" in query_payload


@pytest.mark.django_db
def test_upload_preview_nulls_returns_summary(client) -> None:
    df = pd.DataFrame(
        {
            "country": ["IT", None, "DE"],
            "rate": [5.0, None, 8.1],
        }
    )
    upload = make_excel_upload(df, "nulls.xlsx")
    response = client.post("/api/upload/preview-nulls", {"excel_file": upload})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["preview"]["has_nulls"] is True
    assert payload["preview"]["total_nulls"] >= 2


@pytest.mark.django_db
def test_chat_stream_sends_trace_token_final_done_in_order(client, monkeypatch: pytest.MonkeyPatch) -> None:
    upload = UploadRecord.objects.create(
        original_filename="active.xlsx",
        file_hash="hash",
        row_count=2,
        column_count=2,
        columns_json=["country", "unemployment_rate"],
        is_active=True,
    )

    def fake_runner(*, assistant_message_id: int, user_text: str) -> None:
        assistant = ChatMessage.objects.get(id=assistant_message_id)
        AgentTraceEvent.objects.create(
            message=assistant,
            step_index=1,
            payload={
                "step": 1,
                "tool": "run_sql",
                "input": user_text,
                "observation": "ok",
                "status": "completed",
            },
        )
        assistant.content = "Result text [Order:O1, Row:1, File:active.xlsx]"
        assistant.citations = [{"order": "O1", "row": "1", "file": "active.xlsx"}]
        assistant.status = ChatMessage.STATUS_DONE
        assistant.save(update_fields=["content", "citations", "status", "updated_at"])

    class ImmediateThread:
        def __init__(self, *, target, kwargs, daemon):
            self.target = target
            self.kwargs = kwargs
            self.daemon = daemon

        def start(self):
            self.target(**self.kwargs)

    monkeypatch.setattr("ai_analysis.views.run_assistant_message", fake_runner)
    monkeypatch.setattr("ai_analysis.views.threading.Thread", ImmediateThread)
    monkeypatch.setattr("ai_analysis.views.time.sleep", lambda _x: None)

    send_resp = client.post(
        "/api/chat/messages",
        data=json.dumps({"message": "show orders"}),
        content_type="application/json",
    )
    assert send_resp.status_code == 200
    stream_url = send_resp.json()["stream_url"]

    stream_resp = client.get(stream_url)
    assert stream_resp.status_code == 200
    body = b"".join(stream_resp.streaming_content).decode("utf-8")

    trace_index = body.find("event: trace")
    token_index = body.find("event: token")
    final_index = body.find("event: final")
    done_index = body.find("event: done")

    assert trace_index != -1
    assert token_index != -1
    assert final_index != -1
    assert done_index != -1
    assert trace_index < token_index < final_index < done_index


@pytest.mark.django_db
def test_api_chat_sessions_creates_session(client) -> None:
    response = client.post("/api/chat/sessions")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["session"]["id"] > 0
    assert payload["chat_url"].startswith("/chat?session=")


@pytest.mark.django_db
def test_api_chat_session_delete_removes_session_messages_and_traces(client) -> None:
    session_to_delete = ChatSession.objects.create(title="Delete Me")
    keep_session = ChatSession.objects.create(title="Keep Me")

    user_message = ChatMessage.objects.create(
        role=ChatMessage.ROLE_USER,
        session=session_to_delete,
        content="hello",
        status=ChatMessage.STATUS_DONE,
    )
    assistant_message = ChatMessage.objects.create(
        role=ChatMessage.ROLE_ASSISTANT,
        session=session_to_delete,
        content="world",
        status=ChatMessage.STATUS_DONE,
    )
    AgentTraceEvent.objects.create(
        message=assistant_message,
        step_index=1,
        payload={"step": 1, "tool": "run_sql", "status": "completed"},
    )

    response = client.post(f"/api/chat/sessions/{session_to_delete.id}/delete")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["deleted_session_id"] == session_to_delete.id
    assert payload["next_session_id"] == keep_session.id
    assert payload["chat_url"] == f"/chat?session={keep_session.id}"

    assert not ChatSession.objects.filter(id=session_to_delete.id).exists()
    assert not ChatMessage.objects.filter(id=user_message.id).exists()
    assert not ChatMessage.objects.filter(id=assistant_message.id).exists()
    assert AgentTraceEvent.objects.count() == 0
    assert ChatSession.objects.filter(id=keep_session.id).exists()


@pytest.mark.django_db
def test_api_chat_sessions_clear_removes_all_history(client) -> None:
    session_one = ChatSession.objects.create(title="Chat One")
    session_two = ChatSession.objects.create(title="Chat Two")

    assistant_one = ChatMessage.objects.create(
        role=ChatMessage.ROLE_ASSISTANT,
        session=session_one,
        content="one",
        status=ChatMessage.STATUS_DONE,
    )
    ChatMessage.objects.create(
        role=ChatMessage.ROLE_USER,
        session=session_two,
        content="two",
        status=ChatMessage.STATUS_DONE,
    )
    AgentTraceEvent.objects.create(
        message=assistant_one,
        step_index=1,
        payload={"step": 1, "tool": "run_sql", "status": "completed"},
    )

    response = client.post("/api/chat/sessions/clear")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["deleted_sessions"] == 2
    assert payload["chat_url"] == "/chat"

    assert ChatSession.objects.count() == 0
    assert ChatMessage.objects.count() == 0
    assert AgentTraceEvent.objects.count() == 0


@pytest.mark.django_db
def test_api_chat_messages_uses_provided_session_id(client, monkeypatch: pytest.MonkeyPatch) -> None:
    upload = UploadRecord.objects.create(
        original_filename="active.xlsx",
        file_hash="hash",
        row_count=2,
        column_count=2,
        columns_json=["country", "unemployment_rate"],
        is_active=True,
    )
    session = ChatSession.objects.create(title="My Chat")

    def fake_runner(*, assistant_message_id: int, user_text: str) -> None:
        assistant = ChatMessage.objects.get(id=assistant_message_id)
        assistant.content = "ok"
        assistant.status = ChatMessage.STATUS_DONE
        assistant.save(update_fields=["content", "status", "updated_at"])

    class ImmediateThread:
        def __init__(self, *, target, kwargs, daemon):
            self.target = target
            self.kwargs = kwargs
            self.daemon = daemon

        def start(self):
            self.target(**self.kwargs)

    monkeypatch.setattr("ai_analysis.views.run_assistant_message", fake_runner)
    monkeypatch.setattr("ai_analysis.views.threading.Thread", ImmediateThread)

    response = client.post(
        "/api/chat/messages",
        data=json.dumps({"message": "hello", "session_id": session.id}),
        content_type="application/json",
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session.id
    assert ChatMessage.objects.filter(session_id=session.id, role=ChatMessage.ROLE_USER).exists()


@pytest.mark.django_db
def test_upload_rejected_when_another_upload_is_processing(client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_analysis.views._start_upload_processing", lambda _filename: False)
    monkeypatch.setattr(
        "ai_analysis.views._upload_processing_payload",
        lambda: {"is_processing": True, "filename": "busy.xlsx", "elapsed_seconds": 3},
    )

    csv_path = Path(settings.BASE_DIR) / "data" / "disoccupazione.csv"
    upload = SimpleUploadedFile(
        name="disoccupazione.csv",
        content=csv_path.read_bytes(),
        content_type="text/csv",
    )
    response = client.post("/api/upload", {"excel_file": upload, "null_strategy": "none"})
    assert response.status_code == 409
    payload = response.json()
    assert "already in progress" in payload["error"]
    assert payload["processing"]["is_processing"] is True


@pytest.mark.django_db
def test_settings_page_and_runtime_apis(client, monkeypatch: pytest.MonkeyPatch) -> None:
    page = client.get("/settings")
    assert page.status_code == 200

    def fake_verify(**kwargs):
        assert kwargs["ollama_base_url"] == "http://127.0.0.1:11434"
        return {
            "ok": True,
            "message": "ok",
            "missing_models": [],
            "available_models": ["qwen3.5:9b", "nomic-embed-text"],
        }

    monkeypatch.setattr("ai_analysis.views.verify_ollama_settings", fake_verify)

    verify_resp = client.post(
        "/api/settings/verify",
        data=json.dumps(
            {
                "ollama_base_url": "http://127.0.0.1:11434",
                "ollama_qa_model": "qwen3.5:9b",
                "ollama_chat_model": "qwen3.5:9b",
                "ollama_embed_model": "nomic-embed-text",
            }
        ),
        content_type="application/json",
    )
    assert verify_resp.status_code == 200
    assert verify_resp.json()["ok"] is True

    save_resp = client.post(
        "/api/settings/save",
        data=json.dumps(
            {
                "ollama_base_url": "http://127.0.0.1:11434",
                "ollama_qa_model": "qwen3.5:9b",
                "ollama_chat_model": "qwen3.5:9b",
                "ollama_embed_model": "nomic-embed-text",
                "custom_prompt": "Use markdown tables when useful.",
                "mark_verified": True,
            }
        ),
        content_type="application/json",
    )
    assert save_resp.status_code == 200
    payload = save_resp.json()
    assert payload["ok"] is True
    assert payload["config"]["custom_prompt"] == "Use markdown tables when useful."

    config = RuntimeConfig.objects.order_by("-updated_at").first()
    assert config is not None
    assert config.verified_at is not None
    assert config.custom_prompt == "Use markdown tables when useful."


@pytest.mark.django_db
def test_settings_verify_failure_returns_200_with_ok_false(client, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_verify(**_kwargs):
        return {
            "ok": False,
            "message": "Missing models.",
            "missing_models": ["qwen3.5:9b"],
            "available_models": ["nomic-embed-text"],
        }

    monkeypatch.setattr("ai_analysis.views.verify_ollama_settings", fake_verify)

    response = client.post(
        "/api/settings/verify",
        data=json.dumps(
            {
                "ollama_base_url": "http://127.0.0.1:11434",
                "ollama_qa_model": "qwen3.5:9b",
                "ollama_chat_model": "qwen3.5:9b",
                "ollama_embed_model": "nomic-embed-text",
            }
        ),
        content_type="application/json",
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["result"]["missing_models"] == ["qwen3.5:9b"]

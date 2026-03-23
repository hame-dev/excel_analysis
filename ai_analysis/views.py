from __future__ import annotations

import json
import threading
import time
from typing import Any, Iterator

from django.db.models import Count, Max
from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from ai_analysis.models import AgentTraceEvent, ChatMessage, ChatSession, UploadRecord
from ai_analysis.services.agent import run_assistant_message
from ai_analysis.services.app_settings import (
    get_runtime_config,
    save_runtime_settings,
    serialize_runtime_config,
    verify_ollama_settings,
)
from ai_analysis.services.data_ingestion import (
    analyze_upload_nulls,
    load_active_dataset_into_memory,
    replace_active_dataset,
)
from ai_analysis.services.data_query import (
    build_filter_schema,
    build_query_payload,
    get_active_dataframe,
    parse_filters,
)
from ai_analysis.services.runtime import RUNTIME_STATE

SUPPORTED_UPLOAD_EXTENSIONS = (".xlsx", ".xls", ".csv")
UPLOAD_STATE_LOCK = threading.Lock()
UPLOAD_STATE: dict[str, Any] = {
    "is_processing": False,
    "filename": "",
    "started_at": None,
}


def _active_upload() -> UploadRecord | None:
    return UploadRecord.objects.filter(is_active=True).first()


def _serialize_message(message: ChatMessage) -> dict[str, Any]:
    return {
        "id": message.id,
        "session_id": message.session_id,
        "role": message.role,
        "content": message.content,
        "citations": message.citations,
        "status": message.status,
        "upload_label": message.upload_record.original_filename if message.upload_record else None,
        "created_at": message.created_at.isoformat(),
    }


def _session_title_from_text(text: str) -> str:
    compact = " ".join((text or "").strip().split())
    if not compact:
        return "New Chat"
    return compact[:80]


def _serialize_session(session: ChatSession) -> dict[str, Any]:
    last_message_text = (
        session.messages.order_by("-created_at", "-id").values_list("content", flat=True).first() or ""
    )
    return {
        "id": session.id,
        "title": session.title,
        "message_count": int(getattr(session, "message_count", 0) or 0),
        "last_message": str(last_message_text)[:120],
        "updated_at": session.updated_at.isoformat(),
    }


def _resolve_chat_session(session_id: Any) -> ChatSession | None:
    try:
        if session_id is None:
            return None
        session_id_int = int(session_id)
        if session_id_int <= 0:
            return None
    except (TypeError, ValueError):
        return None
    return ChatSession.objects.filter(id=session_id_int).first()


def _create_chat_session(title: str = "New Chat") -> ChatSession:
    return ChatSession.objects.create(title=title)


def _touch_session(session: ChatSession) -> None:
    session.save(update_fields=["updated_at"])


def _start_upload_processing(filename: str) -> bool:
    with UPLOAD_STATE_LOCK:
        if UPLOAD_STATE["is_processing"]:
            return False
        UPLOAD_STATE["is_processing"] = True
        UPLOAD_STATE["filename"] = filename
        UPLOAD_STATE["started_at"] = time.time()
        return True


def _finish_upload_processing() -> None:
    with UPLOAD_STATE_LOCK:
        UPLOAD_STATE["is_processing"] = False
        UPLOAD_STATE["filename"] = ""
        UPLOAD_STATE["started_at"] = None


def _upload_processing_payload() -> dict[str, Any]:
    with UPLOAD_STATE_LOCK:
        started_at = UPLOAD_STATE.get("started_at")
        elapsed_seconds = int(time.time() - float(started_at)) if started_at else 0
        return {
            "is_processing": bool(UPLOAD_STATE.get("is_processing")),
            "filename": str(UPLOAD_STATE.get("filename") or ""),
            "elapsed_seconds": elapsed_seconds,
        }


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _token_chunks(text: str) -> Iterator[str]:
    if not text:
        return
    words = text.split(" ")
    for i, word in enumerate(words):
        sep = " " if i < len(words) - 1 else ""
        yield f"{word}{sep}"


def home(request: HttpRequest) -> HttpResponse:
    return redirect("upload_page")


@require_GET
def upload_page(request: HttpRequest) -> HttpResponse:
    context = {
        "active_upload": _active_upload(),
        "upload_history": UploadRecord.objects.order_by("-uploaded_at")[:25],
        "upload_processing": _upload_processing_payload(),
    }
    return render(request, "ai_analysis/upload.html", context)


@require_GET
def chat_page(request: HttpRequest) -> HttpResponse:
    sessions_qs = ChatSession.objects.annotate(
        message_count=Count("messages"),
        last_message_at=Max("messages__created_at"),
    ).order_by("-updated_at", "-id")

    selected = _resolve_chat_session(request.GET.get("session"))
    if selected is None:
        selected = sessions_qs.first()

    messages_qs = ChatMessage.objects.select_related("upload_record").order_by("created_at", "id")
    if selected is not None:
        messages_qs = messages_qs.filter(session_id=selected.id)
    else:
        messages_qs = messages_qs.none()

    context = {
        "active_upload": _active_upload(),
        "messages": list(messages_qs[:500]),
        "chat_sessions": [_serialize_session(s) for s in sessions_qs[:100]],
        "selected_session_id": selected.id if selected else None,
    }
    return render(request, "ai_analysis/chat.html", context)


@require_GET
def visualization_page(request: HttpRequest) -> HttpResponse:
    context = {
        "active_upload": _active_upload(),
    }
    return render(request, "ai_analysis/visualization.html", context)


@require_GET
def settings_page(request: HttpRequest) -> HttpResponse:
    config = get_runtime_config()
    context = {
        "runtime_config": serialize_runtime_config(config),
    }
    return render(request, "ai_analysis/settings.html", context)


@require_POST
def api_upload(request: HttpRequest) -> JsonResponse:
    upload = request.FILES.get("excel_file")
    if upload is None:
        return JsonResponse({"error": "Missing file under key 'excel_file'."}, status=400)

    if not upload.name.lower().endswith(SUPPORTED_UPLOAD_EXTENSIONS):
        return JsonResponse({"error": "Only .xlsx/.xls/.csv files are supported."}, status=400)

    null_strategy = request.POST.get("null_strategy", UploadRecord.NULL_STRATEGY_NONE)
    valid_strategies = {choice[0] for choice in UploadRecord.NULL_STRATEGY_CHOICES}
    if null_strategy not in valid_strategies:
        return JsonResponse({"error": f"Invalid null_strategy '{null_strategy}'."}, status=400)

    if not _start_upload_processing(upload.name):
        return JsonResponse(
            {
                "error": "Another upload is already in progress. Please wait.",
                "processing": _upload_processing_payload(),
            },
            status=409,
        )

    try:
        record = replace_active_dataset(upload, null_strategy=null_strategy)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    finally:
        _finish_upload_processing()

    return JsonResponse(
        {
            "ok": True,
            "upload": {
                "id": record.id,
                "filename": record.original_filename,
                "rows": record.row_count,
                "columns": record.column_count,
                "null_strategy": record.null_strategy,
                "uploaded_at": record.uploaded_at.isoformat(),
            },
        }
    )


@require_POST
def api_upload_preview_nulls(request: HttpRequest) -> JsonResponse:
    upload = request.FILES.get("excel_file")
    if upload is None:
        return JsonResponse({"error": "Missing file under key 'excel_file'."}, status=400)
    if not upload.name.lower().endswith(SUPPORTED_UPLOAD_EXTENSIONS):
        return JsonResponse({"error": "Only .xlsx/.xls/.csv files are supported."}, status=400)

    try:
        payload = analyze_upload_nulls(upload)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    return JsonResponse({"ok": True, "preview": payload})


@require_GET
def api_upload_status(request: HttpRequest) -> JsonResponse:
    return JsonResponse({"ok": True, "processing": _upload_processing_payload()})


@require_POST
def api_settings_verify(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        payload = {}

    result = verify_ollama_settings(
        ollama_base_url=str(payload.get("ollama_base_url", "")).strip(),
        ollama_qa_model=str(payload.get("ollama_qa_model", "")).strip(),
        ollama_chat_model=str(payload.get("ollama_chat_model", "")).strip(),
        ollama_embed_model=str(payload.get("ollama_embed_model", "")).strip(),
    )
    status = 200 if result.get("ok") else 400
    return JsonResponse({"ok": bool(result.get("ok")), "result": result}, status=status)


@require_POST
def api_settings_save(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        payload = {}

    required_fields = ("ollama_base_url", "ollama_qa_model", "ollama_chat_model", "ollama_embed_model")
    for field in required_fields:
        if not str(payload.get(field, "")).strip():
            return JsonResponse({"error": f"Field '{field}' is required."}, status=400)

    config = save_runtime_settings(payload)
    return JsonResponse({"ok": True, "config": serialize_runtime_config(config)})


@require_POST
def api_chat_sessions(request: HttpRequest) -> JsonResponse:
    session = _create_chat_session()
    return JsonResponse(
        {
            "ok": True,
            "session": _serialize_session(session),
            "chat_url": f"/chat?session={session.id}",
        }
    )


@require_POST
def api_chat_messages(request: HttpRequest) -> JsonResponse:
    active_upload = _active_upload()
    if active_upload is None:
        return JsonResponse({"error": "Upload a dataset first."}, status=400)

    if RUNTIME_STATE.get_dataframe() is None:
        load_active_dataset_into_memory()

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        payload = {}
    user_text = str(payload.get("message", "")).strip()
    if not user_text:
        return JsonResponse({"error": "Message is required."}, status=400)

    session = _resolve_chat_session(payload.get("session_id"))
    created_session = False
    if session is None:
        session = _create_chat_session(_session_title_from_text(user_text))
        created_session = True

    had_messages = session.messages.exists()

    user_message = ChatMessage.objects.create(
        role=ChatMessage.ROLE_USER,
        session=session,
        content=user_text,
        upload_record=active_upload,
        status=ChatMessage.STATUS_DONE,
    )
    assistant_message = ChatMessage.objects.create(
        role=ChatMessage.ROLE_ASSISTANT,
        session=session,
        content="",
        upload_record=active_upload,
        status=ChatMessage.STATUS_PENDING,
    )

    if (not had_messages and session.title == "New Chat") or created_session:
        session.title = _session_title_from_text(user_text)
        session.save(update_fields=["title", "updated_at"])
    else:
        _touch_session(session)

    thread = threading.Thread(
        target=run_assistant_message,
        kwargs={"assistant_message_id": assistant_message.id, "user_text": user_text},
        daemon=True,
    )
    thread.start()

    return JsonResponse(
        {
            "ok": True,
            "user_message": _serialize_message(user_message),
            "assistant_message_id": assistant_message.id,
            "session_id": session.id,
            "session_title": session.title,
            "created_session": created_session,
            "stream_url": f"/api/chat/stream/{assistant_message.id}",
        }
    )


@require_http_methods(["GET"])
def api_chat_stream(request: HttpRequest, message_id: int) -> StreamingHttpResponse:
    def event_stream() -> Iterator[str]:
        last_trace_id = 0
        idle_rounds = 0
        while True:
            message = ChatMessage.objects.select_related("upload_record").filter(id=message_id).first()
            if message is None:
                yield _sse_event("error", {"message": "Assistant message not found."})
                yield _sse_event("done", {"message_id": message_id})
                break

            trace_events = AgentTraceEvent.objects.filter(message_id=message_id, id__gt=last_trace_id).order_by("id")
            for event in trace_events:
                last_trace_id = event.id
                yield _sse_event("trace", event.payload)

            if message.status == ChatMessage.STATUS_ERROR:
                yield _sse_event("error", {"message": message.error_text or "Agent failed."})
                yield _sse_event("done", {"message_id": message.id})
                break

            if message.status == ChatMessage.STATUS_DONE:
                for chunk in _token_chunks(message.content):
                    yield _sse_event("token", {"text": chunk})
                    time.sleep(0.01)
                yield _sse_event(
                    "final",
                    {
                        "answer": message.content,
                        "upload_label": (
                            message.upload_record.original_filename if message.upload_record else "unknown_upload.xlsx"
                        ),
                        "message_id": message.id,
                        "session_id": message.session_id,
                    },
                )
                yield _sse_event("done", {"message_id": message.id})
                break

            idle_rounds += 1
            if idle_rounds % 8 == 0:
                yield ": heartbeat\n\n"
            time.sleep(0.35)

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@require_GET
def api_data_filters(request: HttpRequest) -> JsonResponse:
    df = get_active_dataframe()
    if df.empty:
        return JsonResponse({"filters": [], "columns": []})
    return JsonResponse({"filters": build_filter_schema(df), "columns": [str(c) for c in df.columns.tolist()]})


@require_GET
def api_data_query(request: HttpRequest) -> JsonResponse:
    df = get_active_dataframe()
    if df.empty:
        return JsonResponse({"total": 0, "rows": [], "columns": [], "charts": {}}, status=200)

    filters = parse_filters(request.GET.get("filters"))
    page = max(int(request.GET.get("page", 1)), 1)
    page_size = int(request.GET.get("page_size", 50))
    page_size = max(1, min(page_size, 200))
    payload = build_query_payload(df, filters=filters, page=page, page_size=page_size)
    return JsonResponse(payload, safe=False)

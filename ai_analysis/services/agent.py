"""Agent runtime preserving Ollama + SQLite + DataFrame + tool architecture."""

from __future__ import annotations

import ast
import html
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from django.conf import settings
from django.db import close_old_connections
from langchain_classic.agents import AgentType, initialize_agent
from langchain_classic.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_ollama.llms import OllamaLLM

from ai_analysis.models import AgentTraceEvent, ChatMessage
from ai_analysis.services.app_settings import RuntimeSettings, get_runtime_settings
from ai_analysis.services.prompting import build_hub_prompt, build_sql_agent_prefix
from ai_analysis.services.runtime import RUNTIME_STATE

logger = logging.getLogger(__name__)

PYTHON_EXEC_ALLOWED_IMPORTS: dict[str, Any] = {"pd": pd, "np": np}
ALLOWED_IMPORT_ROOTS = {"pandas", "numpy", "matplotlib", "seaborn", "PIL"}
FOLLOWUP_HINT_TOKENS = {
    "top",
    "same",
    "those",
    "them",
    "again",
    "more",
    "also",
    "list",
    "show",
    "give",
    "first",
    "second",
    "five",
    "ten",
}


def escape_xml_text(text: Any) -> str:
    return html.escape(str(text), quote=False)


def _sqlite_uri() -> str:
    db_path = Path(settings.DATABASES["default"]["NAME"]).resolve()
    return f"sqlite:///{db_path}"


def _validate_python_imports(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Let exec return the syntax error later with full context.
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    return f"Import '{alias.name}' is not allowed."
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0] if module else ""
            if root not in ALLOWED_IMPORT_ROOTS:
                return f"Import from '{module}' is not allowed."
    return None


def _capture_matplotlib_plot() -> tuple[str | None, str | None]:
    try:
        _ensure_non_interactive_matplotlib_backend()
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None, None

    fignums = plt.get_fignums()
    if not fignums:
        return None, None

    plot_dir = Path(settings.MEDIA_ROOT) / "generated_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = f"plot_{int(time.time() * 1000)}_{uuid4().hex[:8]}.png"
    file_path = plot_dir / filename

    fig = plt.figure(fignums[-1])
    fig.savefig(file_path, dpi=160, bbox_inches="tight")
    plt.close("all")
    return str(file_path), f"{settings.MEDIA_URL}generated_plots/{filename}"


def _ensure_non_interactive_matplotlib_backend() -> None:
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib

        current_backend = str(matplotlib.get_backend() or "").lower()
        if "agg" not in current_backend:
            matplotlib.use("Agg", force=True)
    except Exception:
        # Keep execution resilient even if matplotlib is not available.
        return


def _build_llms(runtime_settings: RuntimeSettings) -> tuple[OllamaLLM, OllamaLLM]:
    qa_llm = OllamaLLM(
        model=runtime_settings.ollama_qa_model,
        base_url=runtime_settings.ollama_base_url,
        temperature=0.0,
    )
    chat_llm = OllamaLLM(
        model=runtime_settings.ollama_chat_model,
        base_url=runtime_settings.ollama_base_url,
        temperature=0.7,
    )
    return qa_llm, chat_llm


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    match = re.match(r"^\s*```(?:python)?\s*\n(.*?)\n\s*```\s*$", cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^\s*```(?:python)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    return cleaned


def _normalize_python_code_input(raw_code: str) -> str:
    cleaned = _strip_code_fences(raw_code)

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        try:
            literal = ast.literal_eval(cleaned)
            if isinstance(literal, str):
                cleaned = literal
        except Exception:
            pass

    cleaned = _strip_code_fences(cleaned)
    if "\\n" in cleaned and "\n" not in cleaned:
        cleaned = cleaned.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return cleaned.strip()


def _parse_csv_candidate(response_text: str) -> pd.DataFrame | None:
    lines = response_text.strip().splitlines()
    if len(lines) <= 1 or not any("," in line for line in lines):
        return None
    try:
        df_candidate = pd.read_csv(StringIO(response_text))
    except Exception:
        return None
    if len(df_candidate.columns) == 1 and len(df_candidate.columns[0]) > 60 and len(df_candidate) == 1:
        return None
    return df_candidate


def _summarize_tool_output(output: Any) -> str:
    text = str(output or "").strip()
    if not text.startswith("<tool_output"):
        return text[:2000]

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return text[:2000]

    output_type = root.attrib.get("type", "text")
    message = (root.findtext("message") or "").strip()
    content = (root.findtext("content") or "").strip()
    printed_output = (root.findtext("printed_output") or "").strip()
    preview = (root.findtext("preview") or "").strip()
    dataframe_key = (root.findtext("dataframe_key") or "").strip()
    plot_url = (root.findtext("./plot_image/url") or "").strip()

    parts: list[str] = []
    if output_type == "error":
        err = (root.findtext("message") or "Tool error").strip()
        return f"Error: {err}"[:2000]
    if output_type == "dataframe":
        if message:
            parts.append(message)
        if dataframe_key:
            parts.append(f"dataframe_key={dataframe_key}")
        if preview:
            preview_lines = preview.splitlines()[:8]
            parts.append("preview:\n" + "\n".join(preview_lines))
    else:
        if content:
            parts.append(content)
        if message:
            parts.append(message)
        if printed_output:
            parts.append(f"printed:\n{printed_output}")
        if preview:
            parts.append("preview:\n" + "\n".join(preview.splitlines()[:8]))

    if plot_url:
        parts.append(f"plot_url={plot_url}")

    if not parts:
        return text[:2000]
    return "\n".join(parts)[:2000]


def _dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in citations:
        key = (str(item.get("order", "")), str(item.get("row", "")), str(item.get("file", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _sanitize_user_answer(answer: str) -> str:
    cleaned = re.sub(r"\[\s*Order:[^\]]+\]", "", answer, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?im)^\s*(references?|citations?)\s*:\s*.*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = cleaned.strip()
    return cleaned or answer.strip()


def _shorten_message_text(text: str, limit: int = 320) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "..."


def _is_short_followup_prompt(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text:
        return False
    tokens = re.findall(r"[a-z0-9]+", text)
    if not tokens:
        return False
    if len(tokens) <= 5:
        return True
    return any(token in FOLLOWUP_HINT_TOKENS for token in tokens) and len(tokens) <= 9


def _previous_relevant_user_text(conversation_messages: list[ChatMessage], current_user_text: str) -> str:
    current_clean = (current_user_text or "").strip()
    skipped_current = False
    for message in reversed(conversation_messages):
        if message.role != ChatMessage.ROLE_USER:
            continue
        content = (message.content or "").strip()
        if not content:
            continue
        if not skipped_current and content == current_clean:
            skipped_current = True
            continue
        return content
    return ""


def _build_conversation_input(user_text: str, conversation_messages: list[ChatMessage] | None = None) -> str:
    if not conversation_messages:
        return user_text

    recent_messages = conversation_messages[-10:]
    history_lines: list[str] = []
    for message in recent_messages:
        role = "User" if message.role == ChatMessage.ROLE_USER else "Assistant"
        history_lines.append(f"{role}: {_shorten_message_text(message.content)}")

    previous_user = _previous_relevant_user_text(recent_messages, user_text)
    followup_hint = _is_short_followup_prompt(user_text)

    parts = [
        "Use this recent conversation context for continuity between turns.",
        "Recent conversation:",
        *history_lines,
        f"Current user message: {user_text}",
    ]

    if followup_hint and previous_user:
        parts.extend(
            [
                f"Previous relevant user question: {previous_user}",
                (
                    "Instruction: The current message is likely a follow-up. Reuse the prior metric/entity/filter "
                    "from the previous question and answer directly with dataset tools."
                ),
            ]
        )
    else:
        parts.append(
            "Instruction: If the current message refers to previous turns, resolve references from the context above."
        )

    return "\n".join(parts)


@dataclass
class RunContext:
    upload_label: str
    shared_dataframes: dict[str, pd.DataFrame]
    citations: list[dict[str, Any]] = field(default_factory=list)
    sql_output_counter: int = 0
    py_output_counter: int = 0

    @staticmethod
    def _identifier_column(df: pd.DataFrame) -> str | None:
        candidates = ("order", "id", "code", "reference", "name")
        for col in df.columns:
            lower = str(col).lower()
            if any(token in lower for token in candidates):
                return str(col)
        for col in df.columns:
            if str(col) != "_row_id":
                return str(col)
        return None

    def add_dataframe_citations(self, df: pd.DataFrame) -> None:
        if df.empty:
            self.citations.append({"order": "N/A", "row": "N/A", "file": self.upload_label})
            return
        identifier_col = self._identifier_column(df)
        max_rows = min(5, len(df))
        for i in range(max_rows):
            row = df.iloc[i]
            order = row.get(identifier_col, "N/A") if identifier_col else "N/A"
            row_id = row.get("_row_id", i + 1)
            self.citations.append(
                {
                    "order": str(order),
                    "row": str(row_id),
                    "file": self.upload_label,
                }
            )

    def add_vector_citation(self, metadata: dict[str, Any]) -> None:
        self.citations.append(
            {
                "order": str(metadata.get("identifier") or metadata.get("order") or "N/A"),
                "row": str(metadata.get("row_id") or "N/A"),
                "file": self.upload_label,
            }
        )


class TraceCallbackHandler(BaseCallbackHandler):
    """Callback adapter used by LangChain classic agent to persist tool trace events."""

    def __init__(self, message_id: int):
        super().__init__()
        self.message_id = message_id
        self.step = 0
        self.last_action: dict[str, Any] = {}

    def _save_trace(self, payload: dict[str, Any]) -> None:
        AgentTraceEvent.objects.create(
            message_id=self.message_id,
            step_index=payload.get("step", 0),
            event_kind=AgentTraceEvent.EVENT_TRACE,
            payload=payload,
        )

    # LangChain classic callback signatures are flexible; keep kwargs permissive.
    def on_agent_action(self, action: Any, **_: Any) -> None:  # pragma: no cover - integration behavior
        self.step += 1
        tool_name = getattr(action, "tool", "unknown")
        tool_input = getattr(action, "tool_input", "")
        self.last_action = {"tool": str(tool_name), "input": str(tool_input), "step": self.step}
        self._save_trace(
            {
                "step": self.step,
                "tool": str(tool_name),
                "action": str(tool_name),
                "input": str(tool_input),
                "observation": "",
                "status": "running",
            }
        )

    def on_tool_end(self, output: Any, **_: Any) -> None:  # pragma: no cover - integration behavior
        step = int(self.last_action.get("step", self.step or 1))
        summarized_output = (_summarize_tool_output(output) or "").strip() or "No observation returned."
        self._save_trace(
            {
                "step": step,
                "tool": self.last_action.get("tool", "unknown"),
                "action": self.last_action.get("tool", "unknown"),
                "input": self.last_action.get("input", ""),
                "observation": summarized_output,
                "status": "completed",
            }
        )

    def on_tool_error(self, error: Any, **_: Any) -> None:  # pragma: no cover - integration behavior
        step = int(self.last_action.get("step", self.step or 1))
        self._save_trace(
            {
                "step": step,
                "tool": self.last_action.get("tool", "unknown"),
                "action": self.last_action.get("tool", "unknown"),
                "input": self.last_action.get("input", ""),
                "observation": str(error),
                "status": "error",
            }
        )


def _execute_python_code(code: str, context: RunContext) -> str:
    cleaned = _normalize_python_code_input(code)
    _ensure_non_interactive_matplotlib_backend()

    invalid_import_error = _validate_python_imports(cleaned)
    if invalid_import_error:
        return (
            "<tool_output type=\"error\">"
            f"<message>{escape_xml_text(invalid_import_error)}</message>"
            "</tool_output>"
        )

    exec_globals = {
        **PYTHON_EXEC_ALLOWED_IMPORTS,
        "shared_dataframes": context.shared_dataframes,
        "pd": pd,
        "np": np,
    }
    for key, value in context.shared_dataframes.items():
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            exec_globals[key] = value
    if "df_dataset" in context.shared_dataframes:
        exec_globals["df_dataset"] = context.shared_dataframes["df_dataset"]
    local_vars: dict[str, Any] = {}
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:
        exec(cleaned, exec_globals, local_vars)
        stdout_text = captured.getvalue().strip()
        printed_xml = f"<printed_output>{escape_xml_text(stdout_text)}</printed_output>" if stdout_text else ""
        plot_file_path, plot_url = _capture_matplotlib_plot()
        plot_xml = (
            f"<plot_image><url>{escape_xml_text(plot_url)}</url><file>{escape_xml_text(plot_file_path)}</file></plot_image>"
            if plot_url and plot_file_path
            else ""
        )

        if "result" in local_vars:
            result = local_vars["result"]
            if isinstance(result, pd.DataFrame):
                key = f"df_py_output_{context.py_output_counter}"
                context.py_output_counter += 1
                context.shared_dataframes[key] = result
                context.add_dataframe_citations(result)
                preview = result.head().to_csv(index=False).strip()
                return (
                    "<tool_output type=\"dataframe\">"
                    f"<message>DataFrame saved as shared_dataframes['{key}'].</message>"
                    f"<dataframe_key>{key}</dataframe_key>"
                    f"{printed_xml}"
                    f"{plot_xml}"
                    f"<preview format=\"csv\">\n{escape_xml_text(preview)}\n</preview>"
                    "</tool_output>"
                )

            text = str(result)[:2000]
            return (
                "<tool_output type=\"text\">"
                f"<content>{escape_xml_text(text)}</content>{printed_xml}{plot_xml}"
                "</tool_output>"
            )

        if printed_xml:
            return f"<tool_output type=\"text\">{printed_xml}{plot_xml}</tool_output>"
        if plot_xml:
            return (
                "<tool_output type=\"image\">"
                "<content>Matplotlib plot generated.</content>"
                f"{plot_xml}"
                "</tool_output>"
            )
        return "<tool_output type=\"text\"><content>Code executed. No output.</content></tool_output>"
    except Exception as exc:
        logger.exception("Python tool execution failed.")
        return (
            "<tool_output type=\"error\">"
            f"<message>{escape_xml_text(f'{type(exc).__name__}: {exc}')}</message>"
            "</tool_output>"
        )
    finally:
        sys.stdout = old_stdout


def _get_dataframe_schema(dataframe_name: str, context: RunContext) -> str:
    requested = (dataframe_name or "").strip()
    aliases = {
        "exports": "df_dataset",
        "df_exports": "df_dataset",
        "dataset": "df_dataset",
    }
    normalized = aliases.get(requested.lower(), requested)

    if normalized and normalized != requested:
        dataframe_name = normalized

    if dataframe_name not in context.shared_dataframes:
        available = ", ".join(context.shared_dataframes.keys()) or "none"
        return f"Error: '{dataframe_name}' not found. Available: {available}"

    df = context.shared_dataframes[dataframe_name]
    buf = StringIO()
    df.info(buf=buf)
    preview = df.head().to_string()
    return (
        f"Schema for '{dataframe_name}': {len(df)} rows, {len(df.columns)} cols\n\n"
        f"{buf.getvalue()}\n\nFirst 5 rows:\n{preview}"
    )


def _vector_search(query: str, context: RunContext) -> str:
    if RUNTIME_STATE.vector_store is None:
        return "Vector index is not available."
    try:
        docs = RUNTIME_STATE.vector_store.similarity_search(query, k=10)
    except Exception as exc:  # pragma: no cover - runtime/network dependent
        return f"Vector search error: {exc}"

    if not docs:
        return "No relevant documents found."
    rows = []
    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        context.add_vector_citation(metadata)
        rows.append(f"[{idx}] {doc.page_content}")
    return "\n\n".join(rows)


def _build_sql_tool(context: RunContext, sql_agent_executor: Any) -> Tool:
    def run_sql(query: str) -> str:
        try:
            normalized_query = query
            if isinstance(normalized_query, str):
                normalized_query = re.sub(r"\bdf_dataset\b", "exports", normalized_query, flags=re.IGNORECASE)
                normalized_query = re.sub(r"\bdf_exports\b", "exports", normalized_query, flags=re.IGNORECASE)

            response_text: str
            if hasattr(sql_agent_executor, "invoke"):
                invoke_result = sql_agent_executor.invoke({"input": normalized_query})
                if isinstance(invoke_result, dict):
                    response_text = str(
                        invoke_result.get("output")
                        or invoke_result.get("result")
                        or invoke_result.get("text")
                        or ""
                    )
                else:
                    response_text = str(invoke_result)
            else:
                response_text = str(sql_agent_executor.run(normalized_query))

            parsed_df = _parse_csv_candidate(response_text)
            if parsed_df is not None and not parsed_df.empty:
                key = f"df_sql_output_{context.sql_output_counter}"
                context.sql_output_counter += 1
                context.shared_dataframes[key] = parsed_df
                context.add_dataframe_citations(parsed_df)
                preview = parsed_df.head().to_csv(index=False).strip()
                return (
                    "<tool_output type=\"dataframe\">"
                    f"<message>SQL result saved to shared_dataframes['{key}'].</message>"
                    f"<dataframe_key>{key}</dataframe_key>"
                    f"<preview format=\"csv\">\n{escape_xml_text(preview)}\n</preview>"
                    f"<original_llm_response>{escape_xml_text(response_text)}</original_llm_response>"
                    "</tool_output>"
                )
            return f"<tool_output type=\"text\"><content>{escape_xml_text(response_text)}</content></tool_output>"
        except Exception as exc:
            logger.exception("SQL tool failed.")
            return f"<tool_output type=\"error\"><message>{escape_xml_text(str(exc))}</message></tool_output>"

    return Tool.from_function(
        run_sql,
        name="run_sql",
        description="Execute SQL on SQLite table `exports` for aggregations, grouping, filtering, and comparisons.",
    )


def _build_agent_and_tools(
    df: pd.DataFrame,
    context: RunContext,
    trace_handler: TraceCallbackHandler,
    runtime_settings: RuntimeSettings,
) -> Any:
    qa_llm, _chat_llm = _build_llms(runtime_settings)

    db = SQLDatabase.from_uri(_sqlite_uri(), include_tables=["exports"])
    sql_agent = create_sql_agent(
        llm=qa_llm,
        toolkit=SQLDatabaseToolkit(llm=qa_llm, db=db),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prefix=build_sql_agent_prefix(df, runtime_settings.custom_prompt),
        verbose=True,
        max_iterations=8,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )

    sql_tool = _build_sql_tool(context, sql_agent)
    schema_tool = Tool.from_function(
        lambda name: _get_dataframe_schema(name, context),
        name="get_dataframe_schema",
        description="Inspect DataFrame columns, dtypes, and sample rows. Main dataset key is `df_dataset` (alias: exports).",
    )
    pandas_tool = Tool.from_function(
        lambda code: _execute_python_code(code, context),
        name="run_pandas",
        description="Run pandas/numpy/matplotlib code with shared_dataframes, pd, np. Main DataFrame key: df_dataset.",
    )
    python_tool = Tool.from_function(
        lambda code: _execute_python_code(code, context),
        name="python_interpreter",
        description="Run Python code for calculations and matplotlib plotting with DataFrame access via shared_dataframes.",
    )
    vector_tool = Tool.from_function(
        lambda query: _vector_search(query, context),
        name="vector_search",
        description="Semantic search over uploaded rows and record text.",
    )

    hub_prompt = build_hub_prompt(df, runtime_settings.custom_prompt)
    return initialize_agent(
        tools=[sql_tool, schema_tool, pandas_tool, python_tool, vector_tool],
        llm=qa_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={"prefix": hub_prompt},
        callbacks=[trace_handler],
        handle_parsing_errors=True,
        max_iterations=8,
        early_stopping_method="generate",
    )


def run_agent_query(
    user_text: str,
    assistant_message_id: int,
    upload_label: str,
    conversation_messages: list[ChatMessage] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    close_old_connections()
    df = RUNTIME_STATE.get_dataframe()
    if df is None or df.empty:
        raise RuntimeError("No active dataset available. Upload a file first.")

    shared = dict(RUNTIME_STATE.shared_dataframes)
    context = RunContext(upload_label=upload_label, shared_dataframes=shared)
    runtime_settings = get_runtime_settings()
    trace_handler = TraceCallbackHandler(message_id=assistant_message_id)
    agent_executor = _build_agent_and_tools(df, context, trace_handler, runtime_settings)
    agent_input = _build_conversation_input(user_text, conversation_messages)

    try:
        result = agent_executor.invoke({"input": agent_input})
        raw_answer = result["output"] if isinstance(result, dict) and "output" in result else str(result)
        final_answer = _sanitize_user_answer(raw_answer)
    except Exception as exc:
        logger.exception("Agent invocation failed.")
        raise RuntimeError(str(exc)) from exc

    if not context.citations:
        context.citations.append({"order": "N/A", "row": "N/A", "file": upload_label})

    return final_answer, _dedupe_citations(context.citations)


def run_assistant_message(assistant_message_id: int, user_text: str) -> None:
    close_old_connections()
    try:
        assistant = ChatMessage.objects.select_related("upload_record").get(id=assistant_message_id)
    except ChatMessage.DoesNotExist:
        return

    assistant.status = ChatMessage.STATUS_RUNNING
    assistant.save(update_fields=["status", "updated_at"])

    upload_label = (
        assistant.upload_record.original_filename if assistant.upload_record else "unknown_upload.xlsx"
    )
    conversation_messages: list[ChatMessage] = []
    if assistant.session_id:
        conversation_messages = list(
            ChatMessage.objects.filter(session_id=assistant.session_id)
            .exclude(id=assistant.id)
            .order_by("created_at", "id")[:30]
        )

    try:
        answer, citations = run_agent_query(
            user_text=user_text,
            assistant_message_id=assistant_message_id,
            upload_label=upload_label,
            conversation_messages=conversation_messages,
        )
        assistant.content = answer
        assistant.citations = citations
        assistant.status = ChatMessage.STATUS_DONE
        assistant.error_text = ""
        assistant.save(update_fields=["content", "citations", "status", "error_text", "updated_at"])
    except Exception as exc:
        assistant.status = ChatMessage.STATUS_ERROR
        assistant.error_text = str(exc)
        assistant.content = ""
        assistant.save(update_fields=["status", "error_text", "content", "updated_at"])
    finally:
        close_old_connections()

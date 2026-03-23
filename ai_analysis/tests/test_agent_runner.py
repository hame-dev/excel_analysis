from __future__ import annotations

import pandas as pd
import pytest

from ai_analysis.models import ChatMessage, UploadRecord
from ai_analysis.services.agent import (
    RunContext,
    _build_sql_tool,
    _build_conversation_input,
    _ensure_non_interactive_matplotlib_backend,
    _execute_python_code,
    _get_dataframe_schema,
    _is_short_followup_prompt,
    _normalize_python_code_input,
    run_assistant_message,
)


@pytest.mark.django_db
def test_run_assistant_message_updates_message_with_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    upload = UploadRecord.objects.create(
        original_filename="sample.xlsx",
        file_hash="abc",
        row_count=1,
        column_count=1,
        columns_json=["A"],
        is_active=True,
    )
    assistant = ChatMessage.objects.create(
        role=ChatMessage.ROLE_ASSISTANT,
        status=ChatMessage.STATUS_PENDING,
        upload_record=upload,
    )

    def fake_run_agent_query(
        user_text: str,
        assistant_message_id: int,
        upload_label: str,
        conversation_messages=None,
    ):
        assert user_text == "hello"
        assert assistant_message_id == assistant.id
        assert upload_label == "sample.xlsx"
        assert isinstance(conversation_messages, list)
        return ("answer", [{"order": "ROW-1", "row": "1", "file": "sample.xlsx"}])

    monkeypatch.setattr("ai_analysis.services.agent.run_agent_query", fake_run_agent_query)
    run_assistant_message(assistant.id, "hello")

    assistant.refresh_from_db()
    assert assistant.status == ChatMessage.STATUS_DONE
    assert assistant.content == "answer"
    assert assistant.citations[0]["file"] == "sample.xlsx"


def test_get_dataframe_schema_accepts_exports_alias() -> None:
    df = pd.DataFrame({"country": ["US", "IT"], "value": [1, 2]})
    context = RunContext(upload_label="sample.xlsx", shared_dataframes={"df_dataset": df})

    schema = _get_dataframe_schema("exports", context)
    assert "Schema for 'df_dataset'" in schema
    assert "country" in schema
    assert "value" in schema


def test_sql_tool_rewrites_df_dataset_table_name() -> None:
    class FakeSQLExecutor:
        def __init__(self) -> None:
            self.last_query = ""

        def run(self, query: str) -> str:
            self.last_query = query
            return "ok"

    fake = FakeSQLExecutor()
    context = RunContext(upload_label="sample.xlsx", shared_dataframes={"df_dataset": pd.DataFrame({"a": [1]})})
    sql_tool = _build_sql_tool(context, fake)

    output = sql_tool.func("SELECT * FROM df_dataset WHERE a > 0")
    assert "tool_output" in output
    assert "SELECT * FROM exports WHERE a > 0" == fake.last_query


def test_sql_tool_uses_invoke_when_available() -> None:
    class FakeInvokeSQLExecutor:
        def __init__(self) -> None:
            self.last_payload = None

        def invoke(self, payload):
            self.last_payload = payload
            return {"output": "ok"}

    fake = FakeInvokeSQLExecutor()
    context = RunContext(upload_label="sample.xlsx", shared_dataframes={"df_dataset": pd.DataFrame({"a": [1]})})
    sql_tool = _build_sql_tool(context, fake)

    output = sql_tool.func("SELECT * FROM df_dataset")
    assert "tool_output" in output
    assert fake.last_payload == {"input": "SELECT * FROM exports"}


def test_normalize_python_code_input_unescapes_single_string_payload() -> None:
    payload = "\"import pandas as pd\\nresult = 1 + 2\""
    normalized = _normalize_python_code_input(payload)
    assert "import pandas as pd" in normalized
    assert "\n" in normalized
    assert "\\n" not in normalized


def test_execute_python_code_supports_df_dataset_direct_name() -> None:
    df = pd.DataFrame({"obs_value": [1.0, 2.5, 3.5]})
    context = RunContext(upload_label="sample.xlsx", shared_dataframes={"df_dataset": df})
    result = _execute_python_code("\"result = df_dataset['obs_value'].sum()\\nprint(result)\"", context)
    assert "tool_output" in result
    assert "7.0" in result or "7" in result
    assert "SyntaxError" not in result


def test_matplotlib_backend_forced_to_agg() -> None:
    _ensure_non_interactive_matplotlib_backend()
    import matplotlib

    assert "agg" in str(matplotlib.get_backend()).lower()


@pytest.mark.django_db
def test_build_conversation_input_handles_short_followup() -> None:
    session_upload = UploadRecord.objects.create(
        original_filename="sample2.xlsx",
        file_hash="def",
        row_count=1,
        column_count=1,
        columns_json=["A"],
        is_active=True,
    )
    user_1 = ChatMessage.objects.create(
        role=ChatMessage.ROLE_USER,
        content="what is the top charging location type",
        upload_record=session_upload,
        status=ChatMessage.STATUS_DONE,
    )
    assistant_1 = ChatMessage.objects.create(
        role=ChatMessage.ROLE_ASSISTANT,
        content="The top charging location type is Urban.",
        upload_record=session_upload,
        status=ChatMessage.STATUS_DONE,
    )
    user_2 = ChatMessage.objects.create(
        role=ChatMessage.ROLE_USER,
        content="give me top 5",
        upload_record=session_upload,
        status=ChatMessage.STATUS_DONE,
    )
    composed = _build_conversation_input("give me top 5", [user_1, assistant_1, user_2])
    assert "Previous relevant user question:" in composed
    assert "what is the top charging location type" in composed


def test_is_short_followup_prompt() -> None:
    assert _is_short_followup_prompt("give me top 5")
    assert _is_short_followup_prompt("same for last month")
    assert not _is_short_followup_prompt("show average obs_value by country and sex for 2024")

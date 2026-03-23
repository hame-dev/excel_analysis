"""Dynamic prompt construction based on uploaded dataset schema and stats."""

from __future__ import annotations

from typing import Any

import pandas as pd


def pandas_dtype_to_sql(dtype: Any) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TEXT"
    return "TEXT"


def _sample_values(series: pd.Series, limit: int = 5) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "[]"
    values = [str(v) for v in non_null.unique()[:limit]]
    return "[" + ", ".join(values) + "]"


def _numeric_summary(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "n/a"
    return f"min={numeric.min():.4g}, max={numeric.max():.4g}, mean={numeric.mean():.4g}"


def _date_summary(series: pd.Series) -> str:
    dt = pd.to_datetime(series, errors="coerce").dropna()
    if dt.empty:
        return "n/a"
    return f"min={dt.min()}, max={dt.max()}"


def build_dataset_profile_prompt(df: pd.DataFrame) -> str:
    column_lines: list[str] = []
    numeric_lines: list[str] = []
    date_lines: list[str] = []

    for col in df.columns:
        series = df[col]
        null_ratio = float(series.isna().mean()) if len(series) else 0.0
        dtype = str(series.dtype)
        sql_type = pandas_dtype_to_sql(series.dtype)
        sample = _sample_values(series)
        column_lines.append(
            f"- {col}: pandas={dtype}, sqlite={sql_type}, null_ratio={null_ratio:.2%}, sample_values={sample}"
        )
        if pd.api.types.is_numeric_dtype(series):
            numeric_lines.append(f"- {col}: {_numeric_summary(series)}")
        elif pd.api.types.is_datetime64_any_dtype(series) or "date" in str(col).lower() or "time" in str(col).lower():
            date_lines.append(f"- {col}: {_date_summary(series)}")

    profile = [
        f"Dataset rows: {len(df)}",
        f"Dataset columns: {len(df.columns)}",
        "Columns:",
        *column_lines,
        "Numeric summaries:",
        *(numeric_lines or ["- n/a"]),
        "Date summaries:",
        *(date_lines or ["- n/a"]),
    ]
    return "\n".join(profile)


def build_sql_agent_prefix(df: pd.DataFrame, custom_prompt: str | None = None) -> str:
    dataset_profile = build_dataset_profile_prompt(df)
    custom_section = (
        f"\nCustom Instructions (user-provided):\n{custom_prompt.strip()}\n"
        if custom_prompt and custom_prompt.strip()
        else ""
    )
    return f"""
You are SQLTool, a precise data analyst for SQLite table `exports`.
Use ReAct with tool calls and return a direct final answer.
The table schema and stats are dynamic and provided below.

{dataset_profile}
{custom_section}

Rules:
1. Always query table `exports`.
2. Use case-insensitive matching with LOWER(...) for text filters.
3. For identifiers (id/code/order-like columns), avoid dropping duplicates unless explicitly asked.
4. For date/time columns, use strftime grouping when users ask monthly/yearly trends.
5. For averages, protect against divide-by-zero with NULLIF(..., 0).
6. If query returns empty, retry with broader LIKE-based filters.
7. Never query `df_dataset` in SQL; `df_dataset` is only for Python tools.

Response format:
Thought: ...
Action: ...
Action Input: ...
Observation: ...
Thought: ...
Final Answer: direct answer. Use CSV for tabular outputs.
""".strip()


def build_hub_prompt(df: pd.DataFrame, custom_prompt: str | None = None) -> str:
    dataset_profile = build_dataset_profile_prompt(df)
    custom_section = (
        f"\nCustom Instructions (user-provided):\n{custom_prompt.strip()}\n"
        if custom_prompt and custom_prompt.strip()
        else ""
    )
    return f"""
You are AIAnalysisBot, an assistant for uploaded tabular datasets.
You have the following dataset profile:

{dataset_profile}
{custom_section}

Tools:
- run_sql: SQL queries on SQLite table `exports`.
- get_dataframe_schema: inspect DataFrame schema in shared_dataframes (`df_dataset` is main key).
- run_pandas: execute pandas/numpy/matplotlib code against shared_dataframes.
- python_interpreter: general Python execution (pd/np/matplotlib).
- vector_search: semantic search over indexed rows.

Behavior:
1. Prefer run_sql for aggregations, joins, counts, and filtering.
2. Use get_dataframe_schema before pandas code when schema is unclear.
3. In SQL always use table name `exports`.
4. In Python tools the main DataFrame name is `df_dataset`.
5. Use recent conversation context to resolve follow-up references like "top 5", "same", "those", or "again".
6. For follow-up prompts, inherit metric/entity/filter from the prior relevant user question whenever reasonable.
7. Do not ask clarification if prior-turn context is sufficient to infer intent.
8. Do not reveal private chain-of-thought; provide concise answers.
""".strip()

"""Dataframe-backed filter schema, table query, and chart payload generation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from django.conf import settings

from ai_analysis.services.runtime import RUNTIME_STATE


def _db_path() -> Path:
    return Path(settings.DATABASES["default"]["NAME"]).resolve()


def get_active_dataframe() -> pd.DataFrame:
    cached = RUNTIME_STATE.get_dataframe()
    if cached is not None and not cached.empty:
        return cached

    db_path = _db_path()
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        try:
            return pd.read_sql_query("SELECT * FROM exports", conn)
        except Exception:
            return pd.DataFrame()


def _is_datetime_column(series: pd.Series, column_name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    lower = column_name.lower()
    return any(k in lower for k in ("date", "time", "year", "month"))


def _is_numeric_column(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna().mean() > 0.8


def build_filter_schema(df: pd.DataFrame) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            continue

        if _is_numeric_column(series):
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            if numeric.empty:
                continue
            filters.append({"column": column, "type": "numeric", "min": float(numeric.min()), "max": float(numeric.max())})
            continue

        if _is_datetime_column(series, str(column)):
            dt = pd.to_datetime(non_null, errors="coerce").dropna()
            if not dt.empty:
                filters.append(
                    {
                        "column": column,
                        "type": "datetime",
                        "min": dt.min().isoformat(),
                        "max": dt.max().isoformat(),
                    }
                )
                continue

        unique_count = non_null.nunique()
        if unique_count <= 60:
            sample_vals = [str(v) for v in non_null.astype(str).value_counts().head(60).index.tolist()]
            filters.append({"column": column, "type": "categorical", "options": sample_vals})
        else:
            filters.append({"column": column, "type": "text"})
    return filters


def parse_filters(raw_filters: str | None) -> dict[str, Any]:
    if not raw_filters:
        return {}
    try:
        parsed = json.loads(raw_filters)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    filtered = df.copy()
    for column, rule in filters.items():
        if column not in filtered.columns:
            continue

        series = filtered[column]
        filter_type = rule.get("type")
        if filter_type == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            min_val = rule.get("min")
            max_val = rule.get("max")
            if min_val not in (None, ""):
                filtered = filtered[numeric >= float(min_val)]
            if max_val not in (None, ""):
                filtered = filtered[numeric <= float(max_val)]
        elif filter_type == "datetime":
            dt = pd.to_datetime(series, errors="coerce")
            start = rule.get("start")
            end = rule.get("end")
            if start:
                filtered = filtered[dt >= pd.to_datetime(start)]
            if end:
                filtered = filtered[dt <= pd.to_datetime(end)]
        elif filter_type == "categorical":
            values = [str(v) for v in rule.get("values", [])]
            if values:
                filtered = filtered[series.astype(str).isin(values)]
        elif filter_type == "text":
            contains = str(rule.get("contains", "")).strip()
            if contains:
                filtered = filtered[series.astype(str).str.contains(contains, case=False, na=False)]
    return filtered


def _pick_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if _is_datetime_column(df[col], str(col)):
            return str(col)
    return None


def _pick_numeric_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col == "_row_id":
            continue
        if _is_numeric_column(df[col]):
            return str(col)
    return None


def _pick_categorical_column(df: pd.DataFrame, exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    for col in df.columns:
        col_name = str(col)
        if col_name in exclude or col_name == "_row_id":
            continue
        if _is_numeric_column(df[col]):
            continue
        if _is_datetime_column(df[col], col_name):
            continue
        return col_name
    return None


def _chart_time_trend(df: pd.DataFrame) -> dict[str, Any]:
    date_col = _pick_date_column(df)
    if not date_col:
        return {"labels": [], "values": []}
    dt = pd.to_datetime(df[date_col], errors="coerce")
    temp = df.copy()
    temp["_bucket"] = dt.dt.to_period("M").astype(str)
    temp = temp[temp["_bucket"].notna()]
    if temp.empty:
        return {"labels": [], "values": []}

    grouped = temp.groupby("_bucket").size().sort_index()
    return {"labels": grouped.index.tolist(), "values": grouped.values.tolist()}


def _chart_top_categories(df: pd.DataFrame) -> dict[str, Any]:
    category_col = _pick_categorical_column(df)
    if not category_col:
        return {"labels": [], "values": []}
    value_col = _pick_numeric_column(df)
    temp = df.copy()
    temp[category_col] = temp[category_col].astype(str)
    if value_col:
        values = pd.to_numeric(temp[value_col], errors="coerce")
        temp[value_col] = values
        grouped = temp.groupby(category_col)[value_col].sum().sort_values(ascending=False).head(10)
    else:
        grouped = temp[category_col].value_counts().head(10)
    return {"labels": grouped.index.tolist(), "values": grouped.values.tolist(), "label": category_col}


def _chart_top_entities(df: pd.DataFrame) -> dict[str, Any]:
    category_col = _pick_categorical_column(df)
    if not category_col:
        return {"labels": [], "values": []}
    alt_col = _pick_categorical_column(df, exclude={category_col})
    entity_col = alt_col or category_col
    value_col = _pick_numeric_column(df)
    temp = df.copy()
    temp[entity_col] = temp[entity_col].astype(str)
    if value_col:
        values = pd.to_numeric(temp[value_col], errors="coerce")
        temp[value_col] = values
        grouped = temp.groupby(entity_col)[value_col].sum().sort_values(ascending=False).head(10)
    else:
        grouped = temp[entity_col].value_counts().head(10)
    return {"labels": grouped.index.tolist(), "values": grouped.values.tolist(), "label": entity_col}


def build_query_payload(df: pd.DataFrame, filters: dict[str, Any], page: int, page_size: int) -> dict[str, Any]:
    filtered = apply_filters(df, filters)
    total = len(filtered)
    start = max(page - 1, 0) * page_size
    end = start + page_size
    page_df = filtered.iloc[start:end]

    rows = page_df.where(pd.notnull(page_df), None).to_dict(orient="records")
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "columns": [str(c) for c in filtered.columns.tolist()],
        "rows": rows,
        "charts": {
            "time_trend": _chart_time_trend(filtered),
            "top_categories": _chart_top_categories(filtered),
            "top_entities": _chart_top_entities(filtered),
        },
    }

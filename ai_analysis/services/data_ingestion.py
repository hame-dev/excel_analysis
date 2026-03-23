"""Dataset ingestion, null handling, SQLite replacement, and vector indexing."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from langchain_classic.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from ai_analysis.models import UploadRecord
from ai_analysis.services.app_settings import get_runtime_settings
from ai_analysis.services.prompting import pandas_dtype_to_sql
from ai_analysis.services.runtime import RUNTIME_STATE

logger = logging.getLogger(__name__)

NULL_STRATEGIES = {
    UploadRecord.NULL_STRATEGY_NONE,
    UploadRecord.NULL_STRATEGY_FRONT,
    UploadRecord.NULL_STRATEGY_BACK,
    UploadRecord.NULL_STRATEGY_BOTH,
    UploadRecord.NULL_STRATEGY_MEAN,
    UploadRecord.NULL_STRATEGY_ZERO,
}


def get_sqlite_path() -> Path:
    return Path(settings.DATABASES["default"]["NAME"]).resolve()


def _read_tabular(uploaded_file: Any) -> pd.DataFrame:
    filename = str(uploaded_file.name).lower()
    uploaded_file.seek(0)
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, sheet_name=0)
    uploaded_file.seek(0)
    return df


def analyze_upload_nulls(uploaded_file: Any) -> dict[str, Any]:
    df = _read_tabular(uploaded_file)
    if df.empty:
        return {"has_nulls": False, "total_nulls": 0, "columns": []}

    null_counts = df.isna().sum()
    columns = []
    for col in df.columns:
        count = int(null_counts[col])
        if count <= 0:
            continue
        ratio = float(count / len(df)) if len(df) else 0.0
        columns.append({"column": str(col), "null_count": count, "null_ratio": ratio})

    return {
        "has_nulls": len(columns) > 0,
        "total_nulls": int(null_counts.sum()),
        "columns": columns,
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
    }


def _safe_datetime_convert(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df.copy()
    for col in parsed.columns:
        series = parsed[col]
        col_upper = str(col).upper()
        should_try_datetime = (
            pd.api.types.is_datetime64_any_dtype(series)
            or col_upper.endswith("_DT")
            or "DATE" in col_upper
            or "TIME" in col_upper
        )
        if should_try_datetime:
            try:
                # Use mixed parsing to avoid per-element fallback warnings on heterogeneous date strings.
                parsed[col] = pd.to_datetime(series, errors="coerce", format="mixed")
            except TypeError:
                parsed[col] = pd.to_datetime(series, errors="coerce")
        if pd.api.types.is_datetime64_any_dtype(parsed[col]):
            parsed[col] = parsed[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return parsed


def _apply_null_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    strategy = strategy if strategy in NULL_STRATEGIES else UploadRecord.NULL_STRATEGY_NONE
    transformed = df.copy()

    if strategy == UploadRecord.NULL_STRATEGY_FRONT:
        transformed = transformed.ffill()
    elif strategy == UploadRecord.NULL_STRATEGY_BACK:
        transformed = transformed.bfill()
    elif strategy == UploadRecord.NULL_STRATEGY_BOTH:
        transformed = transformed.ffill().bfill()
    elif strategy == UploadRecord.NULL_STRATEGY_MEAN:
        for col in transformed.columns:
            if pd.api.types.is_numeric_dtype(transformed[col]):
                numeric = pd.to_numeric(transformed[col], errors="coerce")
                transformed[col] = numeric.fillna(float(numeric.mean()) if not numeric.dropna().empty else 0.0)
    elif strategy == UploadRecord.NULL_STRATEGY_ZERO:
        for col in transformed.columns:
            if pd.api.types.is_numeric_dtype(transformed[col]):
                transformed[col] = pd.to_numeric(transformed[col], errors="coerce").fillna(0)
            else:
                transformed[col] = transformed[col].fillna("0")

    return transformed


def _prepare_dataframe(df: pd.DataFrame, null_strategy: str = UploadRecord.NULL_STRATEGY_NONE) -> pd.DataFrame:
    prepared = _safe_datetime_convert(df)
    prepared = _apply_null_strategy(prepared, null_strategy)
    prepared = prepared.replace({np.nan: None})
    if "_row_id" in prepared.columns:
        prepared = prepared.drop(columns=["_row_id"])
    prepared.insert(0, "_row_id", range(1, len(prepared) + 1))
    return prepared


def _quote_identifier(name: str) -> str:
    escaped = str(name).replace('"', '""')
    return f'"{escaped}"'


def _safe_index_name(column_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", column_name.lower()).strip("_")
    return f"idx_exports_{slug or 'col'}"


def _guess_index_columns(df: pd.DataFrame) -> list[str]:
    hints = ("id", "code", "name", "date", "time", "year", "month", "region", "country")
    ranked: list[str] = []
    for col in df.columns:
        lower = str(col).lower()
        if any(h in lower for h in hints):
            ranked.append(str(col))
    if len(ranked) < 4:
        for col in df.columns:
            if str(col) not in ranked:
                ranked.append(str(col))
            if len(ranked) >= 4:
                break
    return ranked[:4]


def _create_exports_table(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    col_defs = []
    for col_name, dtype in df.dtypes.items():
        col_type = "INTEGER" if col_name == "_row_id" else pandas_dtype_to_sql(dtype)
        col_defs.append(f"{_quote_identifier(str(col_name))} {col_type}")

    create_sql = f'CREATE TABLE IF NOT EXISTS "exports" (\n    {", ".join(col_defs)}\n);'
    conn.executescript(
        f"""
        DROP TABLE IF EXISTS exports;
        {create_sql}
        """
    )
    df.to_sql("exports", conn, if_exists="append", index=False)

    for col in _guess_index_columns(df):
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {_safe_index_name(col)} ON exports({_quote_identifier(col)});"
        )
    conn.execute('CREATE INDEX IF NOT EXISTS idx_exports_row_id ON exports("_row_id");')
    conn.commit()


def _hash_uploaded_file(uploaded_file: Any) -> str:
    digest = hashlib.sha256()
    uploaded_file.seek(0)
    for chunk in uploaded_file.chunks():
        digest.update(chunk)
    uploaded_file.seek(0)
    return digest.hexdigest()


def _build_docs(df: pd.DataFrame) -> list[Document]:
    docs: list[Document] = []
    id_col = "_row_id"
    identifier_col = next((c for c in df.columns if c != id_col and "id" in str(c).lower()), None)

    for _, row in df.iterrows():
        pieces = [f"{col}={row.get(col)}" for col in df.columns[:20]]
        metadata = {
            "row_id": str(row.get("_row_id", "")),
            "identifier": str(row.get(identifier_col, "")) if identifier_col else "",
        }
        docs.append(Document(page_content=" | ".join(pieces), metadata=metadata))
    return docs


def _build_vector_store(df: pd.DataFrame) -> Chroma | None:
    docs = _build_docs(df)
    if settings.CHROMA_DIR.exists():
        shutil.rmtree(settings.CHROMA_DIR, ignore_errors=True)
    if not docs:
        return None

    runtime_settings = get_runtime_settings()
    embedder = OllamaEmbeddings(
        model=runtime_settings.ollama_embed_model,
        base_url=runtime_settings.ollama_base_url,
    )
    return Chroma.from_documents(docs, embedder, persist_directory=str(settings.CHROMA_DIR))


def replace_active_dataset(
    uploaded_file: Any,
    null_strategy: str = UploadRecord.NULL_STRATEGY_NONE,
) -> UploadRecord:
    normalized_strategy = null_strategy if null_strategy in NULL_STRATEGIES else UploadRecord.NULL_STRATEGY_NONE
    file_hash = _hash_uploaded_file(uploaded_file)
    raw_df = _read_tabular(uploaded_file)
    df = _prepare_dataframe(raw_df, null_strategy=normalized_strategy)

    sqlite_path = get_sqlite_path()
    with sqlite3.connect(sqlite_path) as conn:
        _create_exports_table(conn, df)

    original_filename = os.path.basename(uploaded_file.name)
    stored_name = f"{uuid.uuid4().hex}_{original_filename}"

    with transaction.atomic():
        previous_uploads = list(UploadRecord.objects.select_for_update().filter(is_active=True))
        for previous in previous_uploads:
            previous.is_active = False
            previous.file_deleted = True
            previous.deactivated_at = timezone.now()
            previous.save(update_fields=["is_active", "file_deleted", "deactivated_at"])

        new_record = UploadRecord(
            original_filename=original_filename,
            file_hash=file_hash,
            row_count=len(df),
            column_count=len(df.columns),
            columns_json=[str(c) for c in df.columns],
            null_strategy=normalized_strategy,
            is_active=True,
        )
        new_record.stored_file.save(stored_name, uploaded_file, save=False)
        new_record.save()

        for previous in previous_uploads:
            if previous.stored_file and previous.stored_file.name:
                previous.stored_file.delete(save=False)

    vector_store = None
    try:
        vector_store = _build_vector_store(df)
    except Exception:  # pragma: no cover - network/runtime dependent
        logger.exception("Vector index build failed. Continuing without vector store.")

    RUNTIME_STATE.set_dataset(df, upload_id=new_record.id, vector_store=vector_store)
    return new_record


def load_active_dataset_into_memory() -> None:
    active = UploadRecord.objects.filter(is_active=True).first()
    if not active:
        RUNTIME_STATE.clear_dataset()
        return

    sqlite_path = get_sqlite_path()
    if not sqlite_path.exists():
        RUNTIME_STATE.clear_dataset()
        return

    with sqlite3.connect(sqlite_path) as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM exports", conn)
        except Exception:
            logger.exception("Could not load active dataset table into runtime cache.")
            RUNTIME_STATE.clear_dataset()
            return

    vector_store = None
    try:
        if settings.CHROMA_DIR.exists():
            runtime_settings = get_runtime_settings()
            embedder = OllamaEmbeddings(
                model=runtime_settings.ollama_embed_model,
                base_url=runtime_settings.ollama_base_url,
            )
            vector_store = Chroma(
                persist_directory=str(settings.CHROMA_DIR),
                embedding_function=embedder,
            )
        else:
            vector_store = _build_vector_store(df)
    except Exception:  # pragma: no cover - network/runtime dependent
        logger.exception("Could not initialize vector store on startup.")

    RUNTIME_STATE.set_dataset(df, upload_id=active.id, vector_store=vector_store)

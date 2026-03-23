from __future__ import annotations

import sqlite3
import warnings

import pandas as pd
import pytest

from ai_analysis.models import UploadRecord
from ai_analysis.services.data_ingestion import _safe_datetime_convert, get_sqlite_path, replace_active_dataset
from ai_analysis.tests.conftest import make_excel_upload


@pytest.mark.django_db
def test_replace_dataset_recreates_exports_and_indexes(disable_vector_build: None) -> None:
    first_df = pd.DataFrame(
        {
            "record_id": ["R-1", "R-2"],
            "region_name": ["A", "B"],
            "metric_group": ["alpha", "beta"],
            "observation_date": ["2025-01-01", "2025-02-01"],
            "metric_value": [100.0, 200.0],
        }
    )
    second_df = pd.DataFrame(
        {
            "record_id": ["R-9"],
            "region_name": ["X"],
            "metric_group": ["gamma"],
            "observation_date": ["2026-01-03"],
            "metric_value": [999.0],
        }
    )

    first_upload = replace_active_dataset(make_excel_upload(first_df, "first.xlsx"))
    assert first_upload.is_active

    sqlite_path = get_sqlite_path()
    with sqlite3.connect(sqlite_path) as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM exports").fetchone()[0]
        assert row_count == 2
        indexes = conn.execute("PRAGMA index_list('exports')").fetchall()
        index_names = {idx[1] for idx in indexes}
        assert any("idx_exports_record_id" in name for name in index_names)
        assert any("idx_exports_row_id" in name for name in index_names)

    second_upload = replace_active_dataset(make_excel_upload(second_df, "second.xlsx"))
    assert second_upload.is_active

    with sqlite3.connect(sqlite_path) as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM exports").fetchone()[0]
        assert row_count == 1
        record_id = conn.execute("SELECT record_id FROM exports LIMIT 1").fetchone()[0]
        assert record_id == "R-9"

    first_upload.refresh_from_db()
    second_upload.refresh_from_db()
    assert not first_upload.is_active
    assert first_upload.file_deleted
    assert second_upload.is_active

    active_count = UploadRecord.objects.filter(is_active=True).count()
    assert active_count == 1


@pytest.mark.django_db
def test_replace_dataset_applies_null_strategy(disable_vector_build: None) -> None:
    df = pd.DataFrame(
        {
            "record_id": ["A", "B", "C"],
            "value_num": [1.0, None, 3.0],
            "region_name": ["north", None, "south"],
        }
    )
    upload = replace_active_dataset(
        make_excel_upload(df, "nulls.xlsx"),
        null_strategy=UploadRecord.NULL_STRATEGY_ZERO,
    )
    assert upload.null_strategy == UploadRecord.NULL_STRATEGY_ZERO

    with sqlite3.connect(get_sqlite_path()) as conn:
        row = conn.execute("SELECT value_num, region_name FROM exports WHERE record_id='B'").fetchone()
        assert row[0] == 0
        assert row[1] == "0"


def test_safe_datetime_convert_avoids_format_inference_warning() -> None:
    df = pd.DataFrame(
        {
            "event_date": ["2026-01-01", "01/02/2026", "2026-03-15T10:22:00"],
            "value": [1, 2, 3],
        }
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        parsed = _safe_datetime_convert(df)
    assert not any("Could not infer format" in str(w.message) for w in captured)
    assert "event_date" in parsed.columns

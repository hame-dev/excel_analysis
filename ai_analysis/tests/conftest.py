from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile


def make_excel_upload(df: pd.DataFrame, filename: str = "dataset.xlsx") -> SimpleUploadedFile:
    if filename.lower().endswith(".csv"):
        content = df.to_csv(index=False).encode("utf-8")
        content_type = "text/csv"
    else:
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        content = buffer.getvalue()
        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    return SimpleUploadedFile(name=filename, content=content, content_type=content_type)


@pytest.fixture
def disable_vector_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_analysis.services.data_ingestion._build_vector_store", lambda _df: None)

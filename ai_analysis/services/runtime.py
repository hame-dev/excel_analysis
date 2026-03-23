"""In-memory runtime state shared by request handlers and agent tasks."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class RuntimeState:
    dataframe: Optional[pd.DataFrame] = None
    shared_dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    vector_store: Any = None
    active_upload_id: Optional[int] = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def set_dataset(self, df: pd.DataFrame, upload_id: int, vector_store: Any) -> None:
        with self.lock:
            self.dataframe = df
            self.shared_dataframes = {"df_dataset": df.copy()}
            self.vector_store = vector_store
            self.active_upload_id = upload_id

    def clear_dataset(self) -> None:
        with self.lock:
            self.dataframe = None
            self.shared_dataframes = {}
            self.vector_store = None
            self.active_upload_id = None

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        with self.lock:
            return None if self.dataframe is None else self.dataframe.copy()


RUNTIME_STATE = RuntimeState()

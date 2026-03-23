from __future__ import annotations

import pandas as pd

from ai_analysis.services.prompting import build_dataset_profile_prompt, build_hub_prompt, build_sql_agent_prefix


def test_dynamic_prompt_includes_schema_samples_and_stats() -> None:
    df = pd.DataFrame(
        {
            "country": ["A1", "A2", "A3"],
            "unemployment_rate": [10.0, 20.5, 30.1],
            "observation_date": pd.to_datetime(["2025-01-01", "2025-01-05", "2025-01-09"]),
        }
    )

    profile = build_dataset_profile_prompt(df)
    assert "Dataset rows: 3" in profile
    assert "country" in profile
    assert "sample_values=[A1, A2, A3]" in profile
    assert "unemployment_rate" in profile and "mean=" in profile
    assert "observation_date" in profile and "Date summaries" in profile


def test_dynamic_prompt_changes_when_schema_changes() -> None:
    df_a = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    df_b = pd.DataFrame({"NEW_COL": [1.5, 3.0], "FLAG": [True, False]})

    prompt_a = build_hub_prompt(df_a)
    prompt_b = build_hub_prompt(df_b)

    assert "NEW_COL" not in prompt_a
    assert "NEW_COL" in prompt_b
    assert "A" in prompt_a

    sql_prefix = build_sql_agent_prefix(df_b)
    assert "table `exports`" in sql_prefix
    assert "NEW_COL" in sql_prefix


def test_custom_prompt_is_injected_when_provided() -> None:
    df = pd.DataFrame({"country": ["A"], "value": [1]})
    custom = "Always answer with concise bullets and include a quick SQL explanation."
    hub_prompt = build_hub_prompt(df, custom_prompt=custom)
    sql_prompt = build_sql_agent_prefix(df, custom_prompt=custom)

    assert "Custom Instructions" in hub_prompt
    assert custom in hub_prompt
    assert "Custom Instructions" in sql_prompt
    assert custom in sql_prompt

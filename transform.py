# transform.py
# Reshape raw FRED records into one clean table: one row per date

import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Rename FRED's series codes to readable column names
COLUMN_NAMES = {
    "CPIAUCSL": "cpi",
    "UNRATE":   "unemployment",
    "FEDFUNDS": "fed_funds",
    "GDP":      "gdp",
    "UMCSENT":  "consumer_sentiment",
}


def transform_data(records):
    """Turn raw FRED records into one wide table: date, cpi, unemployment, fed_funds, gdp, consumer_sentiment."""
    logger.info(f"Transform starting: {len(records)} raw records")

    df = pd.DataFrame(records)

    # Keep only the 3 fields that matter - drop FRED's envelope metadata
    df = df[["date", "value", "series_id"]]

    # FRED uses "." as a text placeholder for missing data - convert to real NaN
    df["value"] = df["value"].replace(".", None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Rename series codes to readable names
    df["series_id"] = df["series_id"].map(COLUMN_NAMES)

    # Pivot: one row per date, one column per indicator
    # dropna=False: keep every expected column even if it were entirely
    # empty in a given run (proven necessary by test_missing_value_becomes_nan)
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="series_id", values="value", dropna=False)
    wide = wide.reset_index()
    wide.columns.name = None   # cosmetic: remove leftover "series_id" axis label

    # GDP reports quarterly - forward-fill it so every month has a value
    wide["gdp"] = wide["gdp"].ffill()

    # Drop rows before all 5 indicators overlap (early years miss some series)
    before = len(wide)
    wide = wide.dropna(subset=["cpi", "unemployment", "fed_funds"])
    logger.info(f"Dropped {before - len(wide)} rows missing core indicators")

    wide = wide.sort_values("date").reset_index(drop=True)

    logger.info(f"Transform complete: {len(wide)} rows, columns: {wide.columns.tolist()}")
    return wide
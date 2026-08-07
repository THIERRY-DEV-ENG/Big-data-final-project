import pandas as pd
import logging

logger = logging.getLogger(__name__)

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

    df = df[["date", "value", "series_id"]]

    df["value"] = df["value"].replace(".", None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df["series_id"] = df["series_id"].map(COLUMN_NAMES)

    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="series_id", values="value", dropna=False)
    wide = wide.reset_index()
    wide.columns.name = None

    wide["gdp"] = wide["gdp"].ffill()

    before = len(wide)
    wide = wide.dropna(subset=["cpi", "unemployment", "fed_funds"])
    logger.info(f"Dropped {before - len(wide)} rows missing core indicators")

    wide = wide.sort_values("date").reset_index(drop=True)

    logger.info(f"Transform complete: {len(wide)} rows, columns: {wide.columns.tolist()}")
    return wide
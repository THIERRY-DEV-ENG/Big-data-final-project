# validate.py
# Check the cleaned DataFrame for problems before loading to the database

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def validate_data(df):
    """Validate the wide FRED table. Returns (clean_df, issues_dict)."""
    logger.info(f"Validation starting: {len(df)} rows")
    issues = {}
    original_count = len(df)

    # Check 1: required columns exist
    required = ["date", "cpi", "unemployment", "fed_funds", "gdp", "consumer_sentiment"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    logger.info("All required columns present")

    # Check 2: nulls in core columns (these must never be missing - we already
    # enforced this in transform, this is a second safety check)
    core = ["cpi", "unemployment", "fed_funds"]
    null_counts = df[core].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Nulls found in core columns: {null_counts.to_dict()}")
        issues["core_nulls"] = null_counts.to_dict()
        df = df.dropna(subset=core)

    # Check 3: consumer_sentiment nulls are EXPECTED (early history) - count, don't drop
    sentiment_nulls = df["consumer_sentiment"].isnull().sum()
    if sentiment_nulls > 0:
        logger.info(f"consumer_sentiment missing in {sentiment_nulls} rows (expected - kept as-is)")
        issues["consumer_sentiment_nulls"] = int(sentiment_nulls)

    # Check 4: unemployment must be a realistic percentage (0-100)
    bad_unemployment = df[(df["unemployment"] < 0) | (df["unemployment"] > 100)]
    if len(bad_unemployment) > 0:
        logger.warning(f"{len(bad_unemployment)} rows with unemployment outside 0-100")
        issues["bad_unemployment"] = len(bad_unemployment)
        df = df[(df["unemployment"] >= 0) & (df["unemployment"] <= 100)]

    # Check 5: fed_funds must be non-negative (rates below 0% are not realistic here)
    bad_rate = df[df["fed_funds"] < 0]
    if len(bad_rate) > 0:
        logger.warning(f"{len(bad_rate)} rows with negative fed_funds")
        issues["bad_fed_funds"] = len(bad_rate)
        df = df[df["fed_funds"] >= 0]

    # Check 6: cpi and gdp must be positive numbers
    bad_positive = df[(df["cpi"] <= 0) | (df["gdp"] <= 0)]
    if len(bad_positive) > 0:
        logger.warning(f"{len(bad_positive)} rows with non-positive cpi or gdp")
        issues["bad_positive_values"] = len(bad_positive)
        df = df[(df["cpi"] > 0) & (df["gdp"] > 0)]

    # Check 7: date must be realistic
    bad_dates = df[(df["date"].dt.year < 1900) | (df["date"].dt.year > 2100)]
    if len(bad_dates) > 0:
        logger.warning(f"{len(bad_dates)} rows with unrealistic dates")
        issues["bad_dates"] = len(bad_dates)
        df = df[(df["date"].dt.year >= 1900) & (df["date"].dt.year <= 2100)]

    removed = original_count - len(df)
    logger.info(f"Validation complete: removed {removed} rows, {len(df)} rows remain")

    return df.reset_index(drop=True), issues
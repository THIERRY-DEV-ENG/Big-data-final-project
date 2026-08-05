# tests/test_pipeline.py
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transform import transform_data
from validate import validate_data


class TestTransform:

    def sample_records(self):
        """Fake raw FRED-shaped records - no internet needed."""
        return [
            {"date": "2020-01-01", "value": "100.0", "series_id": "CPIAUCSL"},
            {"date": "2020-01-01", "value": "5.0",   "series_id": "UNRATE"},
            {"date": "2020-01-01", "value": "1.5",   "series_id": "FEDFUNDS"},
            {"date": "2020-01-01", "value": "21000", "series_id": "GDP"},
            {"date": "2020-01-01", "value": ".",     "series_id": "UMCSENT"},  # missing
        ]

    def test_returns_dataframe(self):
        result = transform_data(self.sample_records())
        assert isinstance(result, pd.DataFrame)

    def test_columns_renamed(self):
        result = transform_data(self.sample_records())
        expected = {"date", "cpi", "unemployment", "fed_funds", "gdp", "consumer_sentiment"}
        assert expected.issubset(set(result.columns))

    def test_missing_value_becomes_nan(self):
        """The '.' placeholder for UMCSENT should become NaN, not stay as text."""
        result = transform_data(self.sample_records())
        assert pd.isna(result["consumer_sentiment"].iloc[0])

    def test_value_is_numeric(self):
        result = transform_data(self.sample_records())
        assert result["cpi"].dtype == "float64"


class TestValidate:

    def clean_df(self):
        return pd.DataFrame({
            "date":               pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "cpi":                [100.0, 101.0],
            "unemployment":       [5.0, 5.2],
            "fed_funds":          [1.5, 1.5],
            "gdp":                [21000.0, 21000.0],
            "consumer_sentiment": [88.0, None],
        })

    def test_clean_data_passes(self):
        result, issues = validate_data(self.clean_df())
        assert len(result) == 2

    def test_sentiment_null_is_flagged_not_dropped(self):
        """A null in consumer_sentiment should stay in the data but appear in issues."""
        result, issues = validate_data(self.clean_df())
        assert len(result) == 2                          # row NOT dropped
        assert "consumer_sentiment_nulls" in issues        # but IS flagged

    def test_negative_unemployment_removed(self):
        df = self.clean_df()
        df.loc[0, "unemployment"] = -5.0
        result, issues = validate_data(df)
        assert len(result) == 1
        assert "bad_unemployment" in issues

    def test_negative_fed_funds_removed(self):
        df = self.clean_df()
        df.loc[0, "fed_funds"] = -1.0
        result, issues = validate_data(df)
        assert len(result) == 1

    def test_zero_cpi_removed(self):
        df = self.clean_df()
        df.loc[0, "cpi"] = 0
        result, issues = validate_data(df)
        assert len(result) == 1

    def test_missing_required_column_raises(self):
        df = self.clean_df()
        df.drop(columns=["cpi"], inplace=True)
        with pytest.raises(ValueError):
            validate_data(df)
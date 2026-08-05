# extract.py
# Get economic indicators from the FRED API

import requests
import json
import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

API_KEY   = os.getenv("FRED_API_KEY")
BASE_URL  = "https://api.stlouisfed.org/fred/series/observations"
SERIES    = ["CPIAUCSL", "UNRATE", "FEDFUNDS", "GDP", "UMCSENT"]
RAW_FILE  = "data/raw_response.json"


def fetch_series(series_id, max_retries=3):
    """Fetch one series from FRED, retrying on failure."""
    params = {"series_id": series_id, "api_key": API_KEY, "file_type": "json"}
    delay = 1

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()["observations"]
                for row in data:
                    row["series_id"] = series_id
                logger.info(f"{series_id}: fetched {len(data)} rows")
                return data
            logger.warning(f"{series_id}: status {response.status_code} on attempt {attempt}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"{series_id}: error on attempt {attempt} - {e}")

        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2

    raise Exception(f"Failed to fetch {series_id} after {max_retries} attempts")


def extract_data(force_download=False):
    """Extract all 5 series, or load from disk if already saved."""
    if not force_download and os.path.exists(RAW_FILE):
        logger.info(f"Loading saved data from {RAW_FILE}")
        with open(RAW_FILE) as f:
            return json.load(f)

    all_records = []
    for series_id in SERIES:
        all_records.extend(fetch_series(series_id))

    os.makedirs("data", exist_ok=True)
    with open(RAW_FILE, "w") as f:
        json.dump(all_records, f, indent=2)

    logger.info(f"Saved {len(all_records)} total records to {RAW_FILE}")
    return all_records
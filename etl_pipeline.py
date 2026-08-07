import logging
import os
import sys
from datetime import datetime

os.makedirs("logs", exist_ok=True)
log_file = f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

from extract import extract_data
from transform import transform_data
from validate import validate_data
from load import get_engine, load_raw_table, load_clean_table


def run_pipeline(force_download=False):
    """Run the full pipeline. Returns True on success, False on failure."""
    logger.info("=" * 50)
    logger.info("PIPELINE STARTED - US Economic Indicators (FRED)")
    logger.info("=" * 50)

    try:
        records = extract_data(force_download=force_download)
        logger.info(f"Extract complete: {len(records)} raw records")
    except Exception as e:
        logger.error(f"Extract failed: {e}")
        return False

    try:
        df = transform_data(records)
        logger.info(f"Transform complete: {len(df)} rows")
    except Exception as e:
        logger.error(f"Transform failed: {e}")
        return False

    try:
        df_valid, issues = validate_data(df)
        if issues:
            logger.warning(f"Validation issues: {issues}")
        logger.info(f"Validate complete: {len(df_valid)} valid rows")
    except ValueError as e:
        logger.error(f"Validate failed: {e}")
        return False

    try:
        engine = get_engine()
        load_raw_table(records, engine)
        load_clean_table(df_valid, engine)
        logger.info("Load complete")
    except Exception as e:
        logger.error(f"Load failed: {e}")
        return False

    logger.info("=" * 50)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY - {len(df_valid)} rows loaded")
    logger.info(f"Log saved to: {log_file}")
    logger.info("=" * 50)
    return True


if __name__ == "__main__":
    success = run_pipeline(force_download=False)
    sys.exit(0 if success else 1)
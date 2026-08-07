import pandas as pd
import sqlalchemy
import json
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    engine = sqlalchemy.create_engine(url)
    logger.info(f"Engine created for database: {os.getenv('DB_NAME')}")
    return engine


def load_raw_table(records, engine):
    df_raw = pd.DataFrame([{"raw_json": json.dumps(r)} for r in records])
    df_raw.to_sql("raw_economic_data", engine, if_exists="replace", index=False)
    logger.info(f"raw_economic_data loaded: {len(df_raw)} rows")


def load_clean_table(df, engine):
    df.to_sql(
        "cleaned_economic_data",
        engine,
        if_exists="replace",
        index=False,
        dtype={
            "date":               sqlalchemy.types.DATE(),
            "cpi":                sqlalchemy.types.FLOAT(),
            "unemployment":       sqlalchemy.types.FLOAT(),
            "fed_funds":          sqlalchemy.types.FLOAT(),
            "gdp":                sqlalchemy.types.FLOAT(),
            "consumer_sentiment": sqlalchemy.types.FLOAT(),
        },
    )
    logger.info(f"cleaned_economic_data loaded: {len(df)} rows")
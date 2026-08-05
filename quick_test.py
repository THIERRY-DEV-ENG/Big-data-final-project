import logging
logging.basicConfig(level=logging.INFO)
from extract import extract_data
from transform import transform_data
from validate import validate_data
from load import get_engine, load_raw_table, load_clean_table

records = extract_data()
df = transform_data(records)
df_valid, issues = validate_data(df)

engine = get_engine()
load_raw_table(records, engine)
load_clean_table(df_valid, engine)

print("Done. Issues found:", issues)
import pandas as pd
import sqlalchemy
import os
from dotenv import load_dotenv

load_dotenv()
url = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
engine = sqlalchemy.create_engine(url)
df = pd.read_sql("SELECT * FROM cleaned_economic_data ORDER BY date", engine)
df.to_csv("data/tableau_economic_data.csv", index=False)
print(f"Exported {len(df)} rows to data/tableau_economic_data.csv")
import duckdb
import pandas as pd

from setting import setting

DATA_PATH = setting.DATASET_PATH / "arxiv-metadata-oai-snapshot.json"
DB_PATH = setting.DATASET_PATH / "arxiv.db"

con = duckdb.connect(DB_PATH)
df = con.execute("""
    SELECT *
    FROM arxiv 
    WHERE categories LIKE '%cs.LG%' 
      AND update_date >= '2020-01-01'
    LIMIT 10
""").fetchdf()

print(df)
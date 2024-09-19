import sqlite3
import pandas as pd

from tqdm import tqdm
from setting import setting

TABLE_NAME = "arxiv"
DB_FILE = setting.DATASET_PATH / "arxiv.db"
IGNORED_COLUMNS = ["license", "versions", "authors_parsed"]

def create_arxiv_db(filename, chunksize=10000):
    # - sqlite only supports NULL, INTEGER, REAL, EXT, BLOB
    #   therefore, not all dtypes supported by pandas could be used in sqlite
    # - schema of arxiv dataset could be found here: https://www.kaggle.com/datasets/Cornell-University/arxiv
    #   we could safely ditch "license", "versions", and "authors_parsed" columns

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # read a single line is very fast
    sample_df = pd.read_json(filename, lines=True, nrows=1).fillna("NONE")
    columns = [column for column in sample_df.columns if column not in IGNORED_COLUMNS]

    # create a table with only 1 line
    sample_df[columns].to_sql(TABLE_NAME, conn, if_exists="append", index=False)

    def get_existing_ids():
        cursor.execute(f"SELECT id FROM {TABLE_NAME}")
        return set(row[0] for row in cursor.fetchall())

    existing_ids = get_existing_ids()

    total_processed_lines = 0
    for chunk in pd.read_json(filename, lines=True, chunksize=chunksize):
        filtered_chunk = chunk[~chunk["id"].isin(existing_ids)][columns].fillna("NONE")
        existing_ids.update(filtered_chunk["id"])

        if not filtered_chunk.empty:
            filtered_chunk.to_sql(TABLE_NAME, conn, if_exists="append", index=False)

        total_processed_lines += chunksize
        print(f"Processed {total_processed_lines}")

    conn.close()


def query_arxiv_db(query):
    # for example, query all papers of cs.SE
    # query = """SELECT * FROM arxiv WHERE categories LIKE '%cs.SE%'"""
    conn = sqlite3.connect(DB_FILE)

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df

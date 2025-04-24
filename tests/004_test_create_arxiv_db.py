from fetcher.arxiv import (
    create_arxiv_db,
    query_arxiv_db
)
from setting import setting


# create_arxiv_db(setting.DATASET_PATH / "arxiv-metadata-oai-snapshot.json")
query = """SELECT * FROM arxiv WHERE categories LIKE '%cs.SE%'"""
df = query_arxiv_db(
    query=query,
)
df.to_pickle(
    setting.DATASET_PATH / "se.pkl"
)

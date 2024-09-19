from fetcher.arxiv import create_arxiv_db
from setting import setting


create_arxiv_db(setting.DATASET_PATH / "arxiv-metadata-oai-snapshot.json")
import faiss
import pandas as pd

from search.sentence_transformers import (
    embed_documents,
    create_index,
)
from utils.common import (
    save_pickle_file,
    load_pickle_file,
)
from setting import setting

EMBEDDING_FILE = setting.DATASET_PATH / "se_embedding.pkl"
INDEX_FILE = setting.DATASET_PATH / "se_index.bin"

df = pd.read_pickle(setting.DATASET_PATH / "se.pkl")
records = df.to_dict("records")

if INDEX_FILE.exists():
    index = faiss.read_index(str(INDEX_FILE))
else:
    if EMBEDDING_FILE.exists():
        embeddings = load_pickle_file(setting.DATASET_PATH / "se_embedding.pkl")
    else:
        embeddings = embed_documents(records, batch_size=4)
        save_pickle_file(embeddings, EMBEDDING_FILE)

    index = create_index(embeddings, batch_size=4)
    faiss.write_index(index, str(INDEX_FILE))
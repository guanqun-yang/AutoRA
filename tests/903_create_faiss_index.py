import duckdb
import faiss
import torch
import numpy as np
import os

from tqdm import trange
from setting import setting
from utils.embedder import (
    get_hf_embedding,
    get_model_and_tokenizer,
)

##################################################

DB_PATH = setting.DATASET_PATH / "arxiv.db"
FAISS_PATH = setting.DATASET_PATH / "arxiv.faiss"
EMBEDDING_MODEL = "intfloat/e5-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256

##################################################

con = duckdb.connect(DB_PATH)
all_ids_df = con.execute("SELECT int_id FROM arxiv").fetchdf()
all_ids = set(all_ids_df["int_id"].tolist())

index = None
if os.path.exists(FAISS_PATH):
    index = faiss.read_index(str(FAISS_PATH))
    existing_ids = set([index.id_map.at(i) for i in range(index.id_map.size())]) if hasattr(index, "id_map") else set()
else:
    print("Creating new FAISS index...")
    index = faiss.IndexIDMap(faiss.IndexFlatIP(768))  # 768 for base models
    existing_ids = set()


missing_ids = sorted(list(all_ids - existing_ids))
print(f"{len(missing_ids)} missing entries to embed.")

model, tokenizer = get_model_and_tokenizer(EMBEDDING_MODEL, DEVICE)

for i in trange(0, len(missing_ids), BATCH_SIZE, desc="Embedding Documents..."):
    batch_ids = missing_ids[i:i + BATCH_SIZE]
    id_list_str = f"({','.join(map(str, batch_ids))})"
    query = f"SELECT int_id, title, abstract FROM arxiv WHERE int_id IN {id_list_str}"
    batch_df = con.execute(query).fetchdf().fillna("")

    texts = list(map(lambda title, abstract: f"{title} {abstract}", batch_df["title"].tolist(), batch_df["abstract"].tolist()))
    int_ids = batch_df["int_id"].tolist()

    embeddings = get_hf_embedding(model, tokenizer, texts, device=DEVICE, batch_size=32)
    embeddings = np.array(embeddings, dtype=np.float32)
    int_ids = np.array(int_ids, dtype=np.int64)

    index.add_with_ids(embeddings, int_ids)

faiss.write_index(index, str(FAISS_PATH))
print("✅ All missing vectors embedded and FAISS index updated.")
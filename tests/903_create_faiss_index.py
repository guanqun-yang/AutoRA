import duckdb
import faiss
import torch
import numpy as np
import os

from tqdm import trange
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from setting import setting

##################################################

DB_PATH = setting.DATASET_PATH / "arxiv.db"
FAISS_PATH = setting.DATASET_PATH / "arxiv.faiss"
EMBEDDING_MODEL = "intfloat/e5-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256

##################################################

def get_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    return model, tokenizer


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()  # (1, seq_len, hidden_size)
    summed = (last_hidden_state * mask).sum(1)  # (1, hidden_size)
    counts = mask.sum(1).clamp(min=1e-9) # (1, hidden_size); avoid zero-division by setting the minimal possible value to 1e-9
    embedding = summed / counts  # (1, hidden_size)

    return embedding


def get_hf_embedding(
        model,
        tokenizer,
        texts,
        pooling_func=mean_pooling,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        batch_size=2,
        max_length=512
):
    is_cuda_available = device.type == "cuda"
    model.eval()
    if is_cuda_available:
        model = model.half()

    dataloader = DataLoader(
        texts,
        batch_size=batch_size,
        collate_fn=lambda batch: tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    )

    embeddings = list()
    with torch.inference_mode():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            output = model(**inputs).last_hidden_state
            batch_embeddings = pooling_func(output, inputs["attention_mask"])
            embeddings.extend(batch_embeddings.cpu())

    return embeddings

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
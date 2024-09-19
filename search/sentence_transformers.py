import faiss
import torch

import numpy as np
import pandas as pd

from tqdm import trange
from sentence_transformers import SentenceTransformer

# models: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer("multi-qa-mpnet-base-dot-v1").to(DEVICE)


def embed_documents(records, batch_size=512):
    embeddings = list()
    for start_idx in trange(0, len(records), batch_size, desc="embedding"):
        texts = [
            "{} {}".format(record["title"], record["abstract"])
            for record in records
        ]

        batch_embeddings = MODEL.encode(texts[start_idx: start_idx+batch_size], device=DEVICE)
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


def create_index(embeddings, batch_size=32):
    faiss.normalize_L2(embeddings)

    embedding_dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(embedding_dim)

    if torch.cuda.is_available():
        index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            0,
            index,
        )

    for start_index in trange(0, embeddings.shape[0], batch_size, desc="indexing"):
        index.add(embeddings[start_index: start_index + batch_size])

    return index


def search_index(queries, records, index, top_k=10):
    query_embedding = MODEL.encode(" ".join(queries), device=DEVICE)
    query_embedding = np.array([query_embedding])
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)
    return pd.DataFrame([records[idx] for idx in indices[0]])


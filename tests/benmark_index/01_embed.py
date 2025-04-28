import torch
import ir_datasets

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from utils.common import save_pickle_file

dataset = ir_datasets.load("beir/fever/test")
qrel = pd.DataFrame([q._asdict() for q in list(dataset.qrels_iter())])
query_df = pd.DataFrame([q._asdict() for q in list(dataset.queries_iter())])
doc_df = pd.DataFrame([d._asdict() for d in list(dataset.docs_iter())])

queries = query_df.text.tolist()
docs = doc_df.text.tolist()
query_ids = query_df.query_id.tolist()
doc_ids = doc_df.doc_id.tolist()

all_string_ids = query_ids + doc_ids
string_id_to_int_id = {sid: i for i, sid in enumerate(all_string_ids)}
query_int_ids = np.array([string_id_to_int_id[sid] for sid in query_ids])
doc_int_ids = np.array([string_id_to_int_id[sid] for sid in doc_ids])

qrel["query_id"] = qrel["query_id"].map(string_id_to_int_id).astype(str)
qrel["doc_id"] = qrel["doc_id"].map(string_id_to_int_id).astype(str)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
query_embeddings = model.encode(queries, batch_size=256, show_progress_bar=True).astype("float32")
doc_embeddings = model.encode(docs, batch_size=256, show_progress_bar=True).astype("float32")

save_pickle_file(qrel, "qrel.pkl")
save_pickle_file(query_int_ids, "query_int_ids.pkl")
save_pickle_file(doc_int_ids, "doc_int_ids.pkl")
save_pickle_file(string_id_to_int_id, "string_id_to_int_id.pkl")
save_pickle_file(query_embeddings, "query_embeddings.pkl")
save_pickle_file(doc_embeddings, "doc_embeddings.pkl")

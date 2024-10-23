import uuid
import torch
import chromadb

import numpy as np
import pandas as pd

from tqdm import trange
from datetime import datetime
from search.sentence_transformers import embed_documents
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from setting import setting

##################################################
# VARIABLES
##################################################

# models: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer("multi-qa-mpnet-base-dot-v1").to(DEVICE)
CLIENT = chromadb.PersistentClient(path=str(setting.DATASET_PATH / "chromadb"))

EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-mpnet-base-dot-v1",
    device=DEVICE,
)
COLLECTION = CLIENT.get_or_create_collection(
    "autora",
    embedding_function=EMBEDDING_FUNCTION,
)
CHUNKSIZE = 10000
##################################################
# CREATE INDEX FOR USS AND ICSE 2014 - 2024
##################################################
def preprocess_chunk(chunk):
    if "abstract" not in chunk.columns:
        chunk["abstract"] = ""

    if "title" not in chunk.columns:
        chunk["title"] = ""

    return chunk


filename = setting.DATASET_PATH / "usenix_icse_2015_2024.json"

for chunk in pd.read_json(
    filename,
    lines=True,
    chunksize=CHUNKSIZE
):
    # chunk is indeed a df with CHUNKSIZE rows
    df = preprocess_chunk(chunk)

    # documents
    documents = df.apply(lambda x: "{} {}".format(x["title"], x["abstract"]), axis="columns").tolist()

    # ids and sources
    ids = [str(uuid.uuid4()) for _ in range(len(documents))]

    # embeddings
    embeddings = embed_documents(documents, batch_size=16)

    # the columns that are not integer, string, or float need to be removed
    COLLECTION.add(
        ids=ids,
        documents=documents,
        metadatas=df.assign(uuid=ids).fillna("NONE").to_dict(orient="records"),
        embeddings=embeddings
    )

##################################################
# MAKE A QUERY ON USS AND ICSE 2014 - 2024
##################################################

# QUERY = "llm language model explain verify interpret security patch detection"
#
# or_conditions = []
# and_conditions = [
#     {"year": {"$lte": 2024}},
#     {"year": {"$gte": 2018}}
# ]
# result_dict = COLLECTION.query(
#     query_texts=[QUERY],
#     n_results=10,
#     where={
#         "$and": and_conditions,
#     }
# )
#
# searched_df = pd.DataFrame(result_dict["metadatas"][0])
# searched_df["distance"] = result_dict["distances"][0]
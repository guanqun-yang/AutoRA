import uuid
import copy
import faiss
import chromadb

from chromadb.utils import embedding_functions

from search.sentence_transformers import (
    embed_documents,
)

import pandas as pd

from utils.schema import (
    align_dataset_schema
)
from utils.common import (
    get_current_datetime,
)

from setting import setting

QUERY = "explain, interpret, verify"
client = chromadb.PersistentClient(path=str(setting.DATASET_PATH / "chromadb"))

##################################################
# NOTE: does not seem to be able to batch encode documents
# https://github.com/chroma-core/chroma/blob/main/chromadb/utils/embedding_functions/sentence_transformer_embedding_function.py
##################################################

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-mpnet-base-dot-v1",
    device="cuda:0",
)

collection = client.get_or_create_collection(
    "autora",
    embedding_function=embedding_function,
)
##################################################
# CREATE INDEX
##################################################

# dataset_name = "acl_small.json"
# dataset_name = "arxiv-metadata-oai-snapshot.json",

DATASET_NAME_DICT = {
    "arxiv": "arxiv-metadata-oai-snapshot.json",
    "acl": "acl.json"
}

# process all data sources at the same time
chunksize = 10000

for dataset_name, filename in DATASET_NAME_DICT.items():
    n_processed = 0
    for chunk in pd.read_json(
        setting.DATASET_PATH / filename,
        lines=True,
        chunksize=chunksize
    ):
        # special processing required by each dataset
        df = align_dataset_schema(chunk, dataset_name=dataset_name)

        # documents
        documents = df.apply(lambda x: "{} {}".format(x["title"], x["abstract"]), axis="columns").tolist()

        # ids and sources
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        # embeddings
        embeddings = embed_documents(documents, batch_size=16)

        # the columns that are not integer, string, or float need to be removed
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=df.assign(uuid=ids).fillna("NONE").to_dict(orient="records"),
            embeddings=embeddings
        )

        n_processed += chunksize
        print(f"{dataset_name}: {n_processed}")

##################################################
# QUERY INDEX
##################################################
# each query in the "query_texts" will be considered an independent query
# keys in the returned dict: ['ids', 'distances', 'metadatas', 'embeddings', 'documents', 'uris', 'data', 'included']
# https://github.com/chroma-core/chroma/blob/48c517d1d8979943b5ae42a4504c44b58e7521eb/chromadb/api/models/Collection.py#L140

# $eq - equal to (string, int, float)
# $ne - not equal to (string, int, float)
# $gt - greater than (int, float)
# $gte - greater than or equal to (int, float)
# $lt - less than (int, float)
# $lte - less than or equal to (int, float)

# where: A Where type dict used to filter results by. e.g. `{"$and": [{"color" : "red"}, {"price": {"$gte": 4.20}}]}`
# where_document: A WhereDocument type dict used to filter by the documents. E.g. `{$contains: {"text": "hello"}}`

# start_year = 2022
# end_year = 2024
#
# select_acl = 1
# select_cs_lg = 0
# select_cs_cl = 1
# select_cs_ai = 0
# select_cs_ir = 0
# select_cs_se = 0
# select_cs_ar = 0
#
#
# and_conditions = [
#     {"year": {"$lt": end_year}},
#     {"year": {"$gte": start_year}},
# ]
# or_conditions = [
#     {"acl": {"$eq": select_acl}},
#     {"cs.LG": {"$eq": select_cs_lg}},
#     {"cs.CL": {"$eq": select_cs_cl}},
#     {"cs.AI": {"$eq": select_cs_ai}},
#     {"cs.IR": {"$eq": select_cs_ir}},
#     {"cs.SE": {"$eq": select_cs_se}},
#     {"cs.AR": {"$eq": select_cs_ar}},
# ]
#
#
# result_dict = collection.query(
#     query_texts=[QUERY],
#     n_results=10,
#     where={
#         "$and": and_conditions,
#         "$or": or_conditions,
#     }
# )
#
# searched_df = pd.DataFrame(result_dict["metadatas"][0])
# searched_df["distance"] = result_dict["distances"][0]
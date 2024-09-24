import uuid
import torch
import chromadb

import numpy as np
import pandas as pd

from tqdm import trange
from datetime import datetime
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

##################################################
# VARIABLES
##################################################

# models: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer("multi-qa-mpnet-base-dot-v1").to(DEVICE)
CLIENT = chromadb.PersistentClient(path="chromadb")

# index
ARXIV_CATEGORIES = [
    "cs.CL", "cs.SE", "cs.CR", "cs.LG", "cs.AI", "cs.IR",
]
EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-mpnet-base-dot-v1",
    device=DEVICE,
)
COLLECTION = CLIENT.get_or_create_collection(
    "autora",
    embedding_function=EMBEDDING_FUNCTION,
)
CHUNKSIZE = 10000

# datasets
DATASET_NAME_DICT = {
    "arxiv": "arxiv-metadata-oai-snapshot.json",
    "acl": "acl.json"
}
##################################################
# FUNCTIONS
##################################################

def extract_year(date_string):
    try:
        year = datetime.striptime(date_string, "%Y-%m-%d").year
    except Exception:
        year = 1900

    return year


def create_categories(row):
    shared_categories = set.intersection(
        set(ARXIV_CATEGORIES),
        set([category.strip() for category in row["categories"].split(",")])
    )

    for category in ARXIV_CATEGORIES:
        if category in shared_categories:
            row[category] = 1
        else:
            row[category] = 0

    return row


def align_dataset_schema(df, dataset_name):
    # cs.CL, cs.SE, cs.CR, cs.LG, cs.AI, cs.IR
    # make sure each dataset have the following consistent names, they could have other columns
    # - title (string)
    # - abstract (string)
    # - authors (string)
    # - url (string)
    # - year (int): this should be in the format of YYYY
    # - acl (boolean)
    # - cs.LG (boolean)
    # - cs.CL (boolean)
    # - cs.AI (boolean)
    # - cs.IR (boolean)
    # - cs.SE (boolean)
    # - cs.CR (boolean)

    assert dataset_name in ["acl", "arxiv"]

    if dataset_name == "acl":
        # original acl columns:
        # url, publisher, address, year, month, editor, title, ENTRYTYPE, ID, abstract, pages, doi, booktitle, author
        # volume, journal, language, isbn, note

        df = df[["title", "abstract", "author", "url", "year"]]
        df = df.rename(columns={"author": "authors"})

        # create multi-hot categories
        df["acl"] = 1
        for category in ARXIV_CATEGORIES:
            df[category] = 0
    else:
        # original arxiv columns:
        # string:
        # - "id", "submitter", "authors", "title", "comments",
        # - "journal-ref", "doi", "report-no", "categories", "license",
        # - "abstract", "update_date"
        # list:
        # - "versions", "authors_parsed"

        df = df[["title", "abstract", "authors", "update_date", "id", "categories"]]

        # custom title
        df["title"] = df.apply(lambda x: "[{}] {}".format(x["id"], x["title"]), axis="columns")

        df["update_date"] = df.update_date.apply(extract_year)
        df["id"] = df["id"].apply(lambda x: f"https://www.arxiv.org/abs/{x}")

        df = df.rename(columns={"update_date": "year", "id": "url"})

        # create multi-hot category
        df["acl"] = 0
        df = df.apply(create_categories, axis="columns")

    return df

def embed_documents(documents, batch_size=512):
    embeddings = list()
    for start_idx in trange(0, len(documents), batch_size, desc="embedding"):
        batch_embeddings = MODEL.encode(documents[start_idx: start_idx+batch_size], device=DEVICE)
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

##################################################
# CREATE INDEX
##################################################


# process all data sources at the same time
for dataset_name, filename in DATASET_NAME_DICT.items():
    n_processed = 0
    for chunk in pd.read_json(
        filename,
        lines=True,
        chunksize=CHUNKSIZE
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
        COLLECTION.add(
            ids=ids,
            documents=documents,
            metadatas=df.assign(uuid=ids).fillna("NONE").to_dict(orient="records"),
            embeddings=embeddings
        )

        n_processed += CHUNKSIZE
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
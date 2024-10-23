import faiss
import pandas as pd
import torch
from termcolor import cprint

from search.sentence_transformers import (
    embed_documents,
    create_index,
    search_index,
)
from setting import setting
from utils.common import (
    save_pickle_file,
    load_pickle_file,
)


def search_nlp_papers(query, top_k=20):
    ##################################################
    # search NLP conferences
    ##################################################
    EMBEDDING_FILE = setting.DATASET_PATH / "acl_embedding.pkl"
    INDEX_FILE = setting.DATASET_PATH / "acl_index.bin"

    df = pd.read_pickle(setting.DATASET_PATH / "acl.pkl")
    df = df[df.year.isin(["2023", "2024"])]

    records = df.to_dict(orient="records")

    if INDEX_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
    else:
        if EMBEDDING_FILE.exists():
            embeddings = load_pickle_file(EMBEDDING_FILE)
        else:
            embeddings = embed_documents(records, batch_size=32)
            save_pickle_file(embeddings, EMBEDDING_FILE)

        index = create_index(embeddings, batch_size=32)
        if torch.cuda.is_available():
            index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(index, str(INDEX_FILE))

    cprint(f"SEARCH QUERY: {query}")
    queries = [entry.strip() for entry in query.split(",")]

    searched_df = search_index(queries, records, index, top_k=int(top_k))
    searched_papers = searched_df[["title", "url", "author", "abstract"]].fillna("NONE").to_dict(orient="records")

    return searched_papers

def search_se_papers(query, top_k=20):
    EMBEDDING_FILE = setting.DATASET_PATH / "se_embedding.pkl"
    INDEX_FILE = setting.DATASET_PATH / "se_index.bin"

    df = pd.read_pickle(setting.DATASET_PATH / "se.pkl")
    df = df[df.update_date >= "2022-01-01"]

    records = df.to_dict(orient="records")

    if INDEX_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
    else:
        if EMBEDDING_FILE.exists():
            embeddings = load_pickle_file(EMBEDDING_FILE)
        else:
            embeddings = embed_documents(records, batch_size=32)
            save_pickle_file(embeddings, EMBEDDING_FILE)

        index = create_index(embeddings, batch_size=32)
        if torch.cuda.is_available():
            index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(index, str(INDEX_FILE))

    cprint(f"SEARCH QUERY: {query}")
    queries = [entry.strip() for entry in query.split(",")]

    searched_df = search_index(queries, records, index, top_k=int(top_k))
    # add url
    searched_df["url"] = searched_df.apply(lambda x: "https://www.arxiv.org/abs/{}".format(x["id"]), axis="columns")
    # add arXiv ID to the title
    searched_df["title"] = searched_df.apply(lambda x: "[{}] {}".format(x["id"], x["title"]), axis="columns")
    # rename the columns for consistency with the UI
    searched_df = searched_df.rename({"authors": "author"}, axis="columns")

    searched_papers = searched_df[["title", "url", "author", "abstract"]].fillna("NONE").to_dict(orient="records")

    return searched_papers

import torch
import faiss
import chromadb

import pandas as pd

from chromadb.utils import embedding_functions
from search.sentence_transformers import (
    embed_documents,
    create_index,
    search_index,
)
from flask import (
    Flask,
    render_template,
    request,
    jsonify
)
from termcolor import cprint

from utils.common import (
    add_google_sheet,
    load_google_sheet,
    get_current_datetime,
    save_pickle_file,
    load_pickle_file,
)
from setting import setting


app = Flask(__name__)
client = chromadb.PersistentClient(path=str(setting.DATASET_PATH / "chromadb"))
collection = client.get_or_create_collection(
    "autora",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="multi-qa-mpnet-base-dot-v1",
        device="cuda:0",
    )
)

##################################################
# NOTE: does not seem to be able to batch encode documents
# https://github.com/chroma-core/chroma/blob/main/chromadb/utils/embedding_functions/sentence_transformer_embedding_function.py
##################################################

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


def search_papers(query, start_year, end_year, num_papers, paper_sources):
    result_dict = collection.query(
        query_texts=[query],
        n_results=num_papers,
        where={
            "$and": [
                {"year": {"$lt": int(end_year)}},
                {"year": {"$gte": int(start_year)}},
                # {"source": {"$in": paper_sources}}
            ]
        }
    )

    searched_papers = result_dict["metadatas"][0]
    return searched_papers


@app.route('/')
def index():
    # the index.html stays in templates/ folder
    # app.py and templates/ should be in the same directory
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    num_papers = int(request.form.get('numPapers'))
    start_year = int(request.form.get("startYear"))
    end_year = int(request.form.get("endYear"))
    paper_sources = request.form.get("paperSources[]")

    papers = search_papers(
        query=query,
        start_year=start_year,
        end_year=end_year,
        paper_sources=paper_sources,
        num_papers=num_papers,
    )
    # papers = list()
    # if "ACL" in paper_sources:
    #     papers = search_nlp_papers(queries, top_k=num_papers)
    # if "SE" in paper_sources:
    #     papers = search_se_papers(queries, top_k=num_papers)

    return jsonify(papers)


if __name__ == '__main__':
    app.run(debug=True)
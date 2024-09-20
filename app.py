import torch
import faiss

import pandas as pd

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

def search_papers(query):
    ##################################################
    # search NLP conferences
    ##################################################

    # df = pd.read_pickle(setting.DATASET_PATH / "acl.pkl")
    # df = df[df.year.isin(["2023", "2024"])]

    # records = df.to_dict(orient="records")
    #
    # embeddings = load_pickle_file(setting.DATASET_PATH / "acl_embedding.pkl")
    # index = create_index(embeddings, batch_size=32)
    #
    # cprint(f"SEARCH QUERY: {query}")
    # queries = [entry.strip() for entry in query.split(",")]
    #
    # searched_df = search_index(queries, records, index, top_k=200)
    # searched_papers = searched_df[["title", "url", "author", "abstract"]].fillna("NONE").to_dict(orient="records")

    ##################################################
    # search SE papers
    ##################################################
    EMBEDDING_FILE = setting.DATASET_PATH / "se_embedding.pkl"
    INDEX_FILE = setting.DATASET_PATH / "se_index.bin"

    df = pd.read_pickle(setting.DATASET_PATH / "se.pkl")
    df = df[df.update_date >= "2022-01-01"]

    records = df.to_dict(orient="records")

    if INDEX_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
    else:
        if EMBEDDING_FILE.exists():
            embeddings = load_pickle_file(setting.DATASET_PATH / "se_embedding.pkl")
        else:
            embeddings = embed_documents(records, batch_size=32)
            save_pickle_file(embeddings, EMBEDDING_FILE)

        index = create_index(embeddings, batch_size=32)
        if torch.cuda.is_available():
            index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(index, str(INDEX_FILE))

    cprint(f"SEARCH QUERY: {query}")
    queries = [entry.strip() for entry in query.split(",")]

    searched_df = search_index(queries, records, index, top_k=200)
    searched_df["url"] = searched_df.apply(lambda x: "https://www.arxiv.org/abs/{}".format(x["id"]), axis="columns")
    searched_df = searched_df.rename({"authors": "author"}, axis="columns")

    searched_papers = searched_df[["title", "url", "author", "abstract"]].fillna("NONE").to_dict(orient="records")

    return searched_papers

@app.route('/')
def index():
    # the index.html stays in templates/ folder
    # app.py and templates/ should be in the same directory
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    queries = request.form['query']
    papers = search_papers(queries)
    return jsonify(papers)


if __name__ == '__main__':
    app.run(debug=True)
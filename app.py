import chromadb
import pandas as pd

from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings

from flask import Flask, render_template, request, jsonify, session
from tqdm import trange
from setting import setting

from search.bm25 import (
    create_index,
    search_index,
)
from utils.common import (
    set_env,
)

set_env()

embedder = VoyageAIEmbeddings(
    model="voyage-3"
)

chroma_client = chromadb.PersistentClient(path="embeddings.chroma")
collection = chroma_client.get_or_create_collection(name="paper")

# src: https://docs.trychroma.com/reference/python/collection
existing_ids = collection.get()["ids"]

all_df = pd.read_pickle(setting.DATASET_PATH / "embedding.pkl").drop_duplicates(subset=["dblp_id"])
new_df = all_df[~all_df.dblp_id.isin(existing_ids)]

##################################################
# CREATE BM25 INDEX
##################################################

index = create_index(all_df.title.tolist())

##################################################
# CREATE CHROMADB INDEX
##################################################

batch_size = 1000
number_of_batches = len(new_df) // batch_size
columns = [
    "year", "venue", "title", "authors", "dblp_id",
]

if not new_df.empty:
    for i in trange(number_of_batches, desc="Creating Index..."):
        batch_df = new_df.iloc[i * batch_size:(i + 1) * batch_size]
        collection.add(
            ids=batch_df.dblp_id.tolist(),
            documents=batch_df.title.tolist(),
            metadatas=batch_df[columns].to_dict("records"),
            embeddings=batch_df.title_embedding.tolist(),
        )

vector_store = Chroma(
    client=chroma_client,
    collection_name="paper",
    embedding_function=embedder,
)
##################################################
## API: https://api.python.langchain.com/en/latest/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html

# query = "vulnerability assessment cvss"
# retrieval_results = vector_store.similarity_search_with_relevance_scores(
#     query=query,
#     k=100,
# )
# search_df = pd.DataFrame([{**doc.metadata, **{"relevance": score}} for doc, score in retrieval_results])

##################################################

app = Flask(__name__)
app.secret_key = "dummy-secret-key"

@app.route("/", methods=["GET", "POST"])
def home():
    query = request.form.get("query", "")
    num_papers = int(request.form.get("num_papers", 10))
    start_year = request.form.get("start_year") or min(all_df["year"].dropna()) # Get start year
    end_year = request.form.get("end_year") or max(all_df["year"].dropna()) # Get end year
    all_venues = all_df.venue.unique().tolist()
    selected_venues = request.form.getlist("venues") or all_venues
    algorithm = request.form.get("algorithm", "BM25")

    search_results = []



    if query:
        if algorithm == "BM25":
            search_df = search_index(query, all_df.title.tolist(), index, k=len(all_df))
            search_df = pd.merge(left=all_df, right=search_df, on="title")
        else:
            # ranking the entire corpus by the title's relevance to the query
            retrieval_results = vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=5 * num_papers,
            )
            search_df = pd.DataFrame([{**doc.metadata, **{"score": score}} for doc, score in retrieval_results])

        if selected_venues:
            search_df = search_df[search_df["venue"].isin(selected_venues)]

        if start_year and end_year:
            search_df = search_df[
                (search_df["year"] >= int(start_year)) & (search_df["year"] <= int(end_year))
            ]

        search_df = search_df.sort_values(by="score", ascending=False).head(num_papers)
        search_df = search_df.sort_values(by=["year", "score"], ascending=[False, False])

        search_results = search_df.to_dict("records")
        session["search_df"] = search_df.to_json(orient="records")

    return render_template(
        "results.html",
        papers=search_results,
        start_year=start_year,
        end_year=end_year,
        query=query,
        venues=all_venues,
        selected_venues=selected_venues
    )

@app.route("/export", methods=["POST"])
def export():
    data = request.json
    relevant_ids = data.get("relevant_ids", [])
    irrelevant_ids = data.get("irrelevant_ids", [])

    search_df_json = session.get('search_df', '[]')
    search_df = pd.read_json(search_df_json, orient="records")

    relevant_papers = search_df.iloc[map(int, relevant_ids)] if relevant_ids else pd.DataFrame()
    irrelevant_papers = search_df.iloc[map(int, irrelevant_ids)] if irrelevant_ids else pd.DataFrame()

    return jsonify({
        "relevant": relevant_papers.to_json(orient="records"),
        "irrelevant": irrelevant_papers.to_json(orient="records"),
    })

if __name__ == "__main__":
    app.run(debug=True)

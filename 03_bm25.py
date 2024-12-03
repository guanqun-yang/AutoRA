from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import bm25s
import Stemmer

from setting import setting

# BM25 setup
STEMMER = Stemmer.Stemmer("english")

def create_index(documents):
    tokenized_documents = bm25s.tokenize(documents, stopwords="en", stemmer=STEMMER)
    index = bm25s.BM25()
    index.index(tokenized_documents)
    return index

def search_index(query, documents, index, k):
    tokenized_query = bm25s.tokenize(query, stemmer=STEMMER)
    results, scores = index.retrieve(tokenized_query, documents, k=k)
    return pd.DataFrame(
        [{"title": result, "score": score} for result, score in zip(results[0], scores[0])]
    )

# Sample DataFrame for testing
df = pd.read_json(setting.DATASET_PATH / "2024-11-26-22-56-31.json", lines=True, orient="records")
documents = df.title.tolist()
index = create_index(documents)

app = Flask(__name__)
app.secret_key = "dummy-secret-key"

@app.route("/", methods=["GET", "POST"])
def home():
    query = request.form.get("query", "")
    num_papers = int(request.form.get("num_papers", 10))
    search_results = []

    if query:
        search_df = search_index(query, documents, index, k=num_papers)
        search_df = pd.merge(left=df, right=search_df, on="title")\
                      .sort_values(by=["year", "score"], ascending=[False, False])
        search_df["id"] = search_df.index
        search_results = search_df.to_dict("records")

        session["search_df"] = search_df.to_json(orient="records")

    return render_template("results.html", papers=search_results, query=query)

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

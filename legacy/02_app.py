import chromadb
from chromadb.utils import embedding_functions
from flask import (
    Flask,
    render_template,
    request,
    jsonify
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

def search_papers(query, start_year, end_year, num_papers):
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

    papers = search_papers(
        query=query,
        start_year=start_year,
        end_year=end_year,
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
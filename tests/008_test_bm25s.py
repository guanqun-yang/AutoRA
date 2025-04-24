import bm25s
import Stemmer
import pandas as pd

from setting import setting
from utils.common import load_google_sheet

STEMMER = Stemmer.Stemmer("english")

def create_index(documents):
    """

    :param documents: a list of strings
    :return: the handler to crated index
    """
    tokenized_documents = bm25s.tokenize(documents, stopwords="en", stemmer=STEMMER)

    index = bm25s.BM25()
    index.index(tokenized_documents)

    return index

def search_index(query, documents, index, k):
    """

    :param query:
    :param documents:
    :param index: the index created in create_index()
    :param k: top-k
    :return:
    """

    tokenized_query = bm25s.tokenize(query, stemmer=STEMMER)
    results, scores = index.retrieve(
        tokenized_query,
        documents,
        k=k
    )

    return pd.DataFrame(
        [
            {
                "text": result,
                "score": score
            }
            for result, score in zip(results[0], scores[0])
        ]
    )


# df = pd.read_json(
#     setting.DATASET_PATH / "2015_2024_usenix_icse.json",
#     lines=True,
#     orient="records"
# )

df = load_google_sheet(
    "LiteratureAnalytics",
    "Collection",
)

documents = df.title.tolist()
index = create_index(documents)

query = "vulnerability detection explain interpret reproduce proof-of-concept accept"
search_df = search_index(query, documents, index, k=200)
result_df = pd.merge(left=df, right=search_df, left_on="title", right_on="text")



